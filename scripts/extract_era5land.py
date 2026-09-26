"""Monthly ERA5-Land climate at every panel point, for the climate robustness check.

The design admits climate forcing as a dynamic input, but the main analysis ran on Landsat
alone. This supplies observed monthly 2 m air temperature, total precipitation and
top-layer volumetric soil water over 2010-2017, which covers the 60-month window before
both target campaigns (2010-2014 for 2015 and 2013-2017 for 2018).

ERA5-Land is a land-only product at about 9 km, so a coastal point can fall on a masked
ocean cell. Those cells are filled from the mean of land cells within 20 km before
sampling, and the count of points that needed it is reported.

    python scripts/extract_era5land.py
    python scripts/extract_era5land.py --limit 200      # quick test on a few points
"""
from __future__ import annotations
import argparse, sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, covariates  # noqa: E402

COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
# band -> (channel name, scale, offset): kelvin to degC, metres to millimetres
BANDS = {
    "temperature_2m": ("t2m", 1.0, -273.15),
    "total_precipitation_sum": ("precip", 1000.0, 0.0),
    "volumetric_soil_water_layer_1": ("swvl1", 1.0, 0.0),
}
CHANNELS = [v[0] for v in BANDS.values()]
START, END = "2010-01-01", "2018-01-01"
SCALE = 11132


def panel_points() -> pd.DataFrame:
    pid = np.unique(np.load(config.DATA_PROCESSED / "panel.npz", allow_pickle=True)["point_id"]
                    .astype(str))
    pts = pd.read_csv(config.DATA_PROCESSED / "lucas_points.csv", dtype={"point_id": str})
    pts = pts[pts.point_id.isin(pid)].drop_duplicates("point_id")
    if len(pts) != len(pid):
        raise ValueError(f"{len(pid) - len(pts)} panel points have no coordinates")
    return pts


def stacked_image(ee):
    """One image with a band per variable per month, ocean cells filled from nearby land."""
    months = pd.date_range(START, END, freq="MS")[:-1]
    coll = ee.ImageCollection(COLLECTION).filterDate(START, END).select(list(BANDS))
    imgs = []
    for m in months:
        img = coll.filterDate(m.strftime("%Y-%m-%d"),
                              (m + pd.DateOffset(months=1)).strftime("%Y-%m-%d")).first()
        tag = m.strftime("%Y%m")
        img = img.rename([f"{BANDS[b][0]}_{tag}" for b in BANDS])
        imgs.append(img)
    stack = ee.Image.cat(imgs)
    near = stack.focalMean(radius=20000, units="meters", kernelType="circle")
    return stack.unmask(near), stack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ee = covariates.init()
    pts = panel_points()
    if args.limit:
        pts = pts.iloc[: args.limit]
    filled, raw = stacked_image(ee)

    out_dir = config.DATA_RAW / "era5land"
    part_dir = config.DATA_INTERIM / "era5land_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    chunks = [pts.iloc[i:i + args.chunk] for i in range(0, len(pts), args.chunk)]
    print(f"{len(pts):,} points in {len(chunks)} chunks")
    frames, n_filled = [], 0
    for i, ch in enumerate(chunks):
        part = part_dir / f"part_{i:03d}{'_test' if args.limit else ''}.csv"
        if part.exists():
            frames.append(pd.read_csv(part, dtype={"point_id": str}))
            print(f"  chunk {i+1}/{len(chunks)}: cached")
            continue
        t0 = time.time()
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"point_id": r.point_id})
            for r in ch.itertuples()])
        got = filled.addBands(raw.select(0).rename("land_check")).reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=SCALE).getInfo()
        rows = [f["properties"] for f in got["features"]]
        df = pd.DataFrame(rows)
        n_filled += int(df["land_check"].isna().sum()) if "land_check" in df else 0
        df = df.drop(columns=["land_check"], errors="ignore")
        for b, (name, s, o) in BANDS.items():
            cols = [c for c in df.columns if c.startswith(name + "_")]
            df[cols] = df[cols] * s + o
        df.to_csv(part, index=False)
        frames.append(df)
        print(f"  chunk {i+1}/{len(chunks)}: {len(df)} points, {time.time()-t0:.0f}s")

    full = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in full.columns if c != "point_id"]
    n_missing = int(full[value_cols].isna().any(axis=1).sum())
    out = out_dir / ("era5land_monthly_test.csv" if args.limit else "era5land_monthly.csv")
    full.to_csv(out, index=False)
    print(f"\nwrote {out}  ({full.shape[0]} points x {len(value_cols)} values)")
    print(f"points on a masked ocean cell, filled from land within 20 km: {n_filled}")
    print(f"points still missing a value: {n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
