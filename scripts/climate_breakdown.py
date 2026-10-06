"""Skill by Köppen climate zone, from the held-out predictions of the main spatial run.

Each point is classed by the Köppen-Geiger rules of Peel et al. (2007), applied to the mean
monthly ERA5-Land temperature and precipitation over every 60-month window the point has in
the panel. Arid (B) comes first; otherwise a coldest month above 0 degC is temperate (C) and
below it cold (D), with the dry-summer subtypes (Cs, Ds) separated from the rest; polar (E)
covers a warmest month below 10 degC. The hybrid, gradient boosting and the mean-reversion
floor (linear on the previous value and campaign, fitted on the training folds) are scored
within each zone with block-bootstrap intervals, beside the size of the change persistence
misses there. Nothing is retrained.

    python scripts/climate_breakdown.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, evaluate  # noqa: E402
from bootstrap_comparison import PRED_DIR, block_bootstrap  # noqa: E402
from extract_era5land import CHANNELS  # noqa: E402

ZONES = {"Cs": "Mediterranean (Cs)", "Cf": "Temperate, no dry season (Cf, Cw)",
         "D": "Continental and boreal (D)", "B": "Arid and semi-arid (B)", "E": "Polar (E)"}


def koppen(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Main zone from a mean monthly climatology, t in degC and p in mm, both (n, 12)."""
    mat, map_ = t.mean(1), p.sum(1)
    summer = p[:, 3:9].sum(1)                       # April to September, northern hemisphere
    winter = map_ - summer
    pth = np.where(winter >= 0.7 * map_, 2 * mat,
                   np.where(summer >= 0.7 * map_, 2 * mat + 28, 2 * mat + 14))
    ps_dry, pw_wet = p[:, 3:9].min(1), np.r_["1", p[:, :3], p[:, 9:]].max(1)
    dry_summer = (ps_dry < 40) & (ps_dry < pw_wet / 3)
    z = np.where(t.min(1) > 0, np.where(dry_summer, "Cs", "Cf"), "D")
    z = np.where(t.max(1) < 10, "E", z)
    return np.where(map_ < 10 * pth, "B", z)


def main() -> int:
    z = np.load(config.DATA_PROCESSED / "panel_mineral.npz", allow_pickle=True)
    c = np.load(config.DATA_PROCESSED / "panel_mineral_climate.npz", allow_pickle=True)
    for k in ("y", "point_id", "times"):
        assert (z[k] == c[k]).all(), f"climate panel is not aligned with the main panel ({k})"
    pr = np.load(PRED_DIR / "spatial_mineral.npz")
    y, y0, yr = z["y"], z["y_prev"], z["times"]
    pid = z["point_id"].astype(str)

    n_main = z["x_dyn"].shape[1]
    t = c["x_dyn"][:, n_main + CHANNELS.index("t2m")].reshape(len(y), 5, 12).mean(1)
    p = c["x_dyn"][:, n_main + CHANNELS.index("precip")].reshape(len(y), 5, 12).mean(1)
    # one zone per point, from the mean over all of its windows
    tp = pd.DataFrame(np.hstack([t, p]), index=pid).groupby(level=0).mean()
    zone_pt = pd.Series(koppen(tp.iloc[:, :12].to_numpy(), tp.iloc[:, 12:].to_numpy()),
                        index=tp.index)
    zone = zone_pt.reindex(pid).to_numpy()

    floor = np.full(len(y), np.nan)
    fold = pr["fold"]
    X = np.column_stack([np.ones(len(y)), y0, yr == 2018]).astype(float)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        floor[te] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    preds = {"hybrid": pr["hybrid__full"],
             "boosting": pr["gradient_boosting__with_previous_value"],
             "mean-reversion floor": floor}

    rows = []
    for code, label in ZONES.items():
        s = zone == code
        if s.sum() < 200:
            print(f"{label}: {s.sum()} observations, too few to score")
            rows.append(dict(zone=label, n=int(s.sum()), points=int((zone_pt == code).sum())))
            continue
        res = block_bootstrap(pr["groups"][s], y0[s], y[s], {k: v[s] for k, v in preds.items()},
                              [("hybrid", "boosting"), ("hybrid", "mean-reversion floor"),
                               ("boosting", "mean-reversion floor")])
        for r in res:
            rows.append(dict(zone=label, n=int(s.sum()), points=int((zone_pt == code).sum()),
                             persistence_rmse=evaluate.rmse(y[s], y0[s]),
                             blocks=len(np.unique(pr["groups"][s])), **r))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "breakdown_climate.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
