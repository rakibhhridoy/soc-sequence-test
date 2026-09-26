"""Assemble the model-ready panel from the LUCAS rounds and the exported covariates.

One observation is a point at a survey round, with the five years of monthly covariates
preceding that survey, the soil properties measured at the *previous* round, and the
previous carbon value. Everything an observation carries was therefore knowable when the
forecast would have been made.

The 2009 round has no predecessor, so it serves only as the previous observation for
2015. Targets are the 2015 and 2018 rounds.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

import config

CHANNELS = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndwi", "mndwi"]
# CEC is recorded in the 2009 release but not in 2015, so it is absent for every
# observation whose previous campaign is 2015. A column present for only half the panel
# cannot be standardised across a mixed fold, so it is dropped rather than imputed.
STATIC = ["clay", "sand", "silt", "ph_h2o", "n_total"]
ROUNDS = [2009, 2015, 2018]
N_MONTHS = 60
ORGANIC_THRESHOLD = 120.0      # g/kg; above this a soil is organic rather than mineral


def load_covariates(raw_dir: Path) -> pd.DataFrame:
    """Wide table of every exported band-month, indexed by point."""
    frames = []
    for f in sorted(Path(raw_dir).glob("*.csv")):
        if f.name.startswith("."):      # macOS resource forks on exFAT
            continue
        d = pd.read_csv(f, low_memory=False)
        keep = [c for c in d.columns if re.match(r".+_\d{6}$", c)]
        d = d[["point_id"] + keep]
        d["point_id"] = d["point_id"].astype(str).str.strip()
        frames.append(d.set_index("point_id"))
    wide = pd.concat(frames, axis=1)
    return wide.loc[:, ~wide.columns.duplicated()]


def _series(wide: pd.DataFrame, points: pd.Index, survey_year: int):
    """(n, C, T) covariates for the N_MONTHS before a survey, plus an observed mask.

    Missing months are interpolated along time and the mask records which values were
    real. Filling without a mask would let the model treat an interpolated month as an
    observation.
    """
    months = pd.date_range(f"{survey_year - N_MONTHS // 12}-01-01",
                           periods=N_MONTHS, freq="MS").strftime("%Y%m")
    arr = np.full((len(points), len(CHANNELS), N_MONTHS), np.nan, np.float32)
    for ci, ch in enumerate(CHANNELS):
        cols = [f"{ch}_{m}" for m in months]
        have = [c for c in cols if c in wide.columns]
        block = wide.reindex(index=points, columns=cols)
        arr[:, ci, :] = block.to_numpy(np.float32)
        if len(have) < len(cols):
            raise ValueError(f"missing exported months for {ch}: {set(cols) - set(have)}")
    mask = (~np.isnan(arr[:, :1, :])).astype(np.float32)      # one mask channel
    filled = np.empty_like(arr)
    for ci in range(arr.shape[1]):                            # interpolate along time
        df = pd.DataFrame(arr[:, ci, :])
        filled[:, ci, :] = (df.interpolate(axis=1, limit_direction="both")
                              .fillna(df.stack().mean()).to_numpy(np.float32))
    return np.concatenate([filled, mask], axis=1)


def build(raw_cov: Path | None = None, panel_path: Path | None = None) -> dict:
    raw_cov = raw_cov or config.DATA_RAW / "covariates"
    panel_path = panel_path or config.DATA_PROCESSED / "lucas_repeat_panel.parquet"

    soil = pd.read_parquet(panel_path)
    soil["point_id"] = soil["point_id"].astype(str)
    wide_soc = soil.pivot_table(index="point_id", columns="year", values="soc")
    wide_soc = wide_soc.dropna(subset=ROUNDS)
    points = wide_soc.index

    statics = {y: soil[soil.year == y].set_index("point_id").reindex(points) for y in ROUNDS}
    coords = (soil.groupby("point_id")[["lat", "lon"]].first().reindex(points))
    xy = (__import__("geopandas")
          .GeoSeries(__import__("geopandas").points_from_xy(coords.lon, coords.lat), crs=4326)
          .to_crs(3035))                                        # metric CRS for blocking
    coords_m = np.c_[xy.x.to_numpy(), xy.y.to_numpy()]

    wide = load_covariates(raw_cov)
    blocks, block_static = [], []
    for target, prev in ((2015, 2009), (2018, 2015)):
        x_dyn = _series(wide, points, target)
        st = statics[prev][STATIC].to_numpy(np.float32)
        block_static.append(st)
        blocks.append(dict(
            x_dyn=x_dyn, x_static=st,
            y=np.log(wide_soc[target].to_numpy(np.float32)),
            y_prev=np.log(wide_soc[prev].to_numpy(np.float32)),
            soc=wide_soc[target].to_numpy(np.float32),
            soc_prev=wide_soc[prev].to_numpy(np.float32),
            coords=coords_m, times=np.full(len(points), float(target)),
            point_id=points.to_numpy(),
        ))
    out = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    # Fill residual gaps from the whole panel, then refuse to return a panel that still
    # carries them: a NaN reaching the network turns its output into NaN silently.
    col_mean = np.nanmean(out["x_static"], axis=0)
    out["x_static"] = np.where(np.isnan(out["x_static"]), col_mean, out["x_static"]).astype(np.float32)
    for key in ("x_dyn", "x_static", "y", "y_prev"):
        n_bad = int(np.isnan(out[key]).sum())
        if n_bad:
            raise ValueError(f"{key} still carries {n_bad} missing values after filling")
    out["mineral"] = (wide_soc[ROUNDS].max(axis=1).to_numpy() < ORGANIC_THRESHOLD)
    out["mineral"] = np.concatenate([out["mineral"], out["mineral"]])
    return out
