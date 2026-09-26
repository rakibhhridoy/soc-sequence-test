"""Append ERA5-Land climate channels to the existing panels.

Each observation gains monthly 2 m temperature, precipitation and top-layer soil water over
the same 60-month window as its Landsat series, ending in December of the year before the
survey. Everything else in the panel is left unchanged, so the only difference between a
run on `panel_mineral.npz` and one on `panel_mineral_climate.npz` is the climate input.

    python scripts/extract_era5land.py          # first
    python scripts/build_climate_panel.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config  # noqa: E402
from build_panel import N_MONTHS  # noqa: E402
from extract_era5land import CHANNELS  # noqa: E402


def climate_block(era: pd.DataFrame, point_id: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.full((len(point_id), len(CHANNELS), N_MONTHS), np.nan, np.float32)
    for year in np.unique(times):
        rows = np.where(times == year)[0]
        months = pd.date_range(f"{int(year) - N_MONTHS // 12}-01-01", periods=N_MONTHS,
                               freq="MS").strftime("%Y%m")
        sub = era.reindex(point_id[rows].astype(str))
        for ci, ch in enumerate(CHANNELS):
            out[rows, ci, :] = sub[[f"{ch}_{m}" for m in months]].to_numpy(np.float32)
    n_bad = int(np.isnan(out).sum())
    if n_bad:
        raise ValueError(f"climate block carries {n_bad} missing values")
    return out


def main() -> int:
    era = pd.read_csv(config.DATA_RAW / "era5land" / "era5land_monthly.csv",
                      dtype={"point_id": str}).set_index("point_id")
    for stem in ("panel_mineral", "panel_organic"):
        z = dict(np.load(config.DATA_PROCESSED / f"{stem}.npz", allow_pickle=True))
        clim = climate_block(era, z["point_id"], z["times"])
        z["x_dyn"] = np.concatenate([z["x_dyn"], clim], axis=1).astype(np.float32)
        out = config.DATA_PROCESSED / f"{stem}_climate.npz"
        np.savez(out, **z)
        print(f"wrote {out}: x_dyn {z['x_dyn'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
