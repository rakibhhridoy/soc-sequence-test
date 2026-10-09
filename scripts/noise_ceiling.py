"""Ceiling on skill against persistence set by the noise in a single measurement.

Each observed log SOC is taken as x_t = s_t + e_t, the true value plus an error from
sampling, relocation and the laboratory, with e_t independent across visits and of
everything a model sees. No forecast can then predict e_t, so its mean squared error is at
least var(e) and its skill against persistence at most 1 - sqrt(var(e) / MSE_persistence).

var(e) is identified from the 8,123 points with all three campaigns. The two changes at a
point share the 2015 measurement with opposite signs, so

    cov(change 2009-2015, change 2015-2018) = cov(true changes) - var(e).

If the true changes at a point are not negatively correlated, var(e) >= -cov of the
observed changes, and replacing var(e) by that lower bound gives an upper bound on the
ceiling. The error variance is assumed equal at every campaign. The skill of each model
from robustness.csv is then expressed as the share of the attainable reduction in mean
squared error, (1 - (1 - skill)^2) / (1 - var(e) / MSE_persistence). Intervals come from
resampling the spatial blocks of the main analysis.

    python scripts/noise_ceiling.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402

N_BOOT = 2000


def ceiling(d1, d2, change):
    """Error variance, correlation of successive changes, MSE of persistence and ceiling."""
    var_e = -np.cov(d1, d2)[0, 1]
    mse = np.mean(change ** 2)
    return dict(var_e=var_e, r=np.corrcoef(d1, d2)[0, 1], mse_persistence=mse,
                ceiling=1 - np.sqrt(max(var_e, 0) / mse))


def main() -> int:
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    y, y0, year, pid = data.y, data.y_prev, data.times, data.point_id
    _, _, groups = train.spatial_folds(data, n_folds=4)
    change = y - y0

    # the points with all three campaigns, one row per point
    a = pd.DataFrame(dict(pid=pid, year=year, y=y, y0=y0, change=change, g=groups))
    w = a.pivot(index="pid", columns="year")
    w = w.dropna(subset=[("change", 2015.0), ("change", 2018.0)])
    assert np.allclose(w[("y", 2015.0)], w[("y0", 2018.0)]), "2018 rows must start from 2015"
    d1, d2 = w[("change", 2015.0)].to_numpy(), w[("change", 2018.0)].to_numpy()
    g_pt = w[("g", 2015.0)].to_numpy()
    print(f"{len(w):,} points with three campaigns, {len(y):,} observations")

    designs = {"all targets": np.ones(len(y), bool), "2015 targets": year == 2015,
               "2018 targets": year == 2018}
    point = {k: ceiling(d1, d2, change[m]) for k, m in designs.items()}

    rng = np.random.default_rng(0)
    ids = np.unique(groups)
    obs = {b: np.where(groups == b)[0] for b in ids}
    pts = {b: np.where(g_pt == b)[0] for b in ids}
    draws = {k: [] for k in designs}
    for _ in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        io = np.concatenate([obs[b] for b in pick])
        ip = np.concatenate([pts[b] for b in pick])
        for k, m in designs.items():
            sel = io[m[io]]
            draws[k].append(ceiling(d1[ip], d2[ip], change[sel]))

    rows = []
    for k in designs:
        for q in ("r", "var_e", "mse_persistence", "ceiling"):
            v = np.array([d[q] for d in draws[k]])
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append(dict(design=k, quantity=q, estimate=point[k][q], ci95_low=lo, ci95_high=hi))

    # the models' skill as a share of the attainable reduction in MSE (spatial blocking)
    rob = pd.read_csv(config.TABLES / "robustness.csv").set_index("quantity")
    attain = 1 - point["all targets"]["var_e"] / point["all targets"]["mse_persistence"]
    for model in ("hybrid", "boosting", "linear"):
        s = rob.loc[model, "estimate"]
        rows.append(dict(design="all targets", quantity=f"share of attainable, {model}",
                         estimate=(1 - (1 - s) ** 2) / attain, ci95_low=np.nan,
                         ci95_high=np.nan))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "noise_ceiling.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
