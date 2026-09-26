"""LUCAS topsoil: load each campaign, normalise columns, and match points across rounds.

Column names differ between the 2009, 2015 and 2018 releases, so every known spelling is
mapped onto one vocabulary. Anything unrecognised is reported rather than dropped
silently, because a column quietly missing would shrink the panel without anyone noticing.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd

# canonical name -> spellings seen across releases (lower-cased, stripped of separators)
ALIASES = {
    "point_id":  ["pointid", "point_id", "pointidgps", "id"],
    "lat":       ["lat", "latitude", "gpslat", "th_lat", "y"],
    "lon":       ["lon", "long", "longitude", "gpslong", "th_long", "x"],
    "soc":       ["oc", "ocgkg", "organiccarbon", "soc", "oc_20", "ec"],
    "ph_h2o":    ["phh2o", "ph_h2o", "phinh2o"],
    "ph_cacl2":  ["phcacl2", "ph_cacl2", "phincacl2"],
    "n_total":   ["n", "ngkg", "totaln", "nitrogen"],
    "clay":      ["clay", "clay_content", "claypercent"],
    "sand":      ["sand", "sandpercent"],
    "silt":      ["silt", "siltpercent"],
    "cec":       ["cec", "cec_cmolkg"],
    "caco3":     ["caco3", "carbonates"],
    "land_cover": ["lc", "lc0desc", "lc1", "landcover", "lc_class"],
    "land_use":  ["lu", "lu1", "landuse", "lu0desc"],
}
def _norm_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# alias keys are normalised too, so "TH_LAT", "th_lat" and "thlat" all resolve
_LOOKUP = {_norm_key(a): canon for canon, alist in ALIASES.items() for a in alist}


_norm = _norm_key


def _coords_from_shapefile(year_dir: Path, id_values: pd.Series) -> pd.DataFrame | None:
    """Pull point coordinates from a shapefile beside the table.

    The LUCAS 2015 CSV carries no latitude or longitude; the accompanying shapefile
    does. Rather than dropping the round, match on the point identifier and take the
    geometry.
    """
    shps = list(year_dir.rglob("*.shp"))
    if not shps:
        return None
    import geopandas as gpd
    g = gpd.read_file(shps[0])
    id_col = next((c for c in g.columns if _norm_key(c) in _LOOKUP
                   and _LOOKUP[_norm_key(c)] == "point_id"), None)
    if id_col is None or g.geometry.isna().all():
        return None
    g = g.to_crs(4326)
    out = pd.DataFrame({"point_id": g[id_col].astype(str).str.strip(),
                        "lat": g.geometry.y, "lon": g.geometry.x})
    return out[out.point_id.isin(set(id_values))]


def load_round(path: Path, year: int) -> pd.DataFrame:
    """Read one campaign table and rename its columns onto the shared vocabulary."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    mapping, unknown = {}, []
    for col in df.columns:
        canon = _LOOKUP.get(_norm(col))
        if canon and canon not in mapping.values():
            mapping[col] = canon
        elif not canon:
            unknown.append(col)
    out = df.rename(columns=mapping)[list(mapping.values())].copy()
    out["year"] = year
    if "point_id" in out.columns:
        out["point_id"] = out["point_id"].astype(str).str.strip()
    if ("lat" not in out.columns or "lon" not in out.columns) and "point_id" in out.columns:
        coords = _coords_from_shapefile(path.parent, out["point_id"])
        if coords is not None:
            out = out.drop(columns=[c for c in ("lat", "lon") if c in out.columns])
            out = out.merge(coords, on="point_id", how="left")
            warnings.warn(f"{path.name}: coordinates taken from the shapefile")
    missing = [c for c in ("point_id", "lat", "lon", "soc") if c not in out.columns]
    if missing:
        raise ValueError(f"{path.name}: required columns not found: {missing}. "
                         f"Unmapped columns were: {unknown[:25]}")
    out["point_id"] = out["point_id"].astype(str).str.strip()
    # LUCAS marks values under the detection limit as "< LOD". Those become missing
    # rather than being guessed at, and the count is reported so the choice is visible.
    for col in ("soc", "ph_h2o", "ph_cacl2", "n_total", "clay", "sand", "silt",
                "cec", "caco3", "lat", "lon"):
        if col in out.columns and out[col].dtype == object:
            num = pd.to_numeric(out[col], errors="coerce")
            lost = int((num.isna() & out[col].notna()).sum())
            if lost:
                warnings.warn(f"{path.name}: {lost} non-numeric values in '{col}' "
                              f"(e.g. below detection limit) set to missing")
            out[col] = num
        elif col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if unknown:
        warnings.warn(f"{path.name}: {len(unknown)} columns unmapped, e.g. {unknown[:6]}")
    return out


def load_all(raw_dir: Path) -> pd.DataFrame:
    """Read every campaign under data/raw/lucas/<year>/ into one long table."""
    raw_dir = Path(raw_dir)
    frames = []
    for year_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        tables = [p for p in year_dir.rglob("*")
                  if p.suffix.lower() in (".csv", ".xlsx", ".xls")
                  and not p.name.startswith((".", "~", "._"))
                  and "readme" not in p.name.lower()]
        # the same data often ships as both .csv and .xlsx; keep one per basename
        best: dict[str, Path] = {}
        rank = {".csv": 0, ".xlsx": 1, ".xls": 2}
        for t in tables:
            cur = best.get(t.stem)
            if cur is None or rank[t.suffix.lower()] < rank[cur.suffix.lower()]:
                best[t.stem] = t
        if not best:
            warnings.warn(f"no table found in {year_dir}")
        for t in sorted(best.values()):
            # A campaign folder often ships auxiliary tables (erosion, organic horizons,
            # bulk density). They lack a carbon value or coordinates, so they are skipped
            # with a warning rather than aborting the load.
            try:
                frames.append(load_round(t, year))
            except ValueError as e:
                warnings.warn(f"skipped {t.name}: {e}")
    if not frames:
        raise FileNotFoundError(f"no LUCAS tables under {raw_dir}")
    return pd.concat(frames, ignore_index=True)


CROP_GRASS_PREFIXES = ("b", "e")   # LUCAS LC codes: B = cropland, E = grassland


def is_cropland_or_grassland(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.str.startswith(CROP_GRASS_PREFIXES)


def repeat_panel(df: pd.DataFrame, cropland_grassland_only: bool = True,
                 require_all_rounds: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Points measured in more than one campaign, with a usable carbon value each time.

    Returns (panel, summary). The number of points surviving here caps what any model
    can be shown to do, so it is established before anything is built on top of it.
    """
    d = df.dropna(subset=["soc"]).copy()
    steps = [("all rows with a carbon value", len(d), d.point_id.nunique())]

    if cropland_grassland_only:
        if "land_cover" not in d.columns or d["land_cover"].isna().all():
            warnings.warn("no land_cover in any campaign; land-cover filter skipped")
        else:
            # Land cover belongs to the point, and not every campaign records it, so it
            # is resolved per point from whichever rounds do. Filtering row by row would
            # silently delete every round that omits the column.
            lc = (d.dropna(subset=["land_cover"])
                    .groupby("point_id")["land_cover"]
                    .agg(lambda s: s.astype(str).mode().iat[0]))
            missing = d.point_id.nunique() - len(lc)
            if missing:
                warnings.warn(f"{missing} points have no land cover in any campaign and are dropped")
            keep_lc = set(lc[is_cropland_or_grassland(lc)].index)
            d = d[d.point_id.isin(keep_lc)]
            steps.append(("cropland or grassland (point-level)", len(d), d.point_id.nunique()))

    rounds = sorted(d.year.unique())
    counts = d.groupby("point_id").year.nunique()
    keep = counts[counts == len(rounds)] if require_all_rounds else counts[counts > 1]
    panel = d[d.point_id.isin(keep.index)].sort_values(["point_id", "year"])
    label = f"present in all {len(rounds)} rounds" if require_all_rounds else "present in >1 round"
    steps.append((label, len(panel), panel.point_id.nunique()))

    summary = pd.DataFrame(steps, columns=["step", "rows", "points"])
    return panel, summary
