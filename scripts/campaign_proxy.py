"""Do the Landsat series act as a proxy for the campaign?

The mix of Landsat sensors differs between the covariate windows (TM and ETM+ before
2009, mostly ETM+ and OLI before 2018) and no cross-sensor adjustment is applied, so the
series could carry the identity of the campaign rather than information about the soil.
Neither model is given the campaign. This gives gradient boosting the campaign as an
explicit input, on the same folds as the main spatial results, and asks what the series
add beyond it. It also repeats the new-places temporal design with and without the series,
to see whether the series are what put boosting below the mean-reversion floor there.

    python scripts/campaign_proxy.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, evaluate  # noqa: E402
from run_ablations import load_npz  # noqa: E402

N_BOOT = 2000


def main() -> int:
    d = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    saved = np.load(config.RESULTS / "predictions" / "spatial_mineral.npz")
    fold, block = saved["fold"], saved["groups"]
    y, y0 = d.y, d.y_prev
    series = baselines.summary_features(d.x_dyn)
    static, prev = d.x_static, d.y_prev[:, None]
    campaign = (d.times == 2018).astype(float)[:, None]

    variants = {
        "static + previous": [static, prev],
        "static + previous + campaign": [static, prev, campaign],
        "static + previous + series": [series, static, prev],
        "static + previous + series + campaign": [series, static, prev, campaign],
    }
    pred = {}
    for name, cols in variants.items():
        X, p = np.hstack(cols), np.zeros(len(y))
        for f in np.unique(fold):
            te = fold == f
            p[te] = baselines.fit_gradient_boosting(X[~te], y[~te]).predict(X[te])
        pred[name] = p
    check = np.abs(pred["static + previous + series"]
                   - saved["gradient_boosting__with_previous_value"]).max()
    assert check < 1e-9, f"does not reproduce the saved boosting predictions ({check})"

    rng = np.random.default_rng(config.SEED)
    blocks = np.unique(block)
    members = {b: np.where(block == b)[0] for b in blocks}
    draws = [np.concatenate([members[b] for b in rng.choice(blocks, len(blocks))])
             for _ in range(N_BOOT)]

    def skill(p, s=slice(None)):
        return evaluate.skill_score(y0[s], y[s], p[s])

    rows = [dict(comparison=f"skill: {k}", difference=skill(v)) for k, v in pred.items()]
    for a, b, label in [
        ("static + previous + campaign", "static + previous", "campaign alone adds"),
        ("static + previous + series", "static + previous", "series add, campaign not given"),
        ("static + previous + series + campaign", "static + previous + campaign",
         "series add beyond the campaign"),
        ("static + previous + series + campaign", "static + previous + series",
         "campaign adds beyond the series"),
    ]:
        diff = np.array([skill(pred[a], s) - skill(pred[b], s) for s in draws])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(comparison=label, difference=skill(pred[a]) - skill(pred[b]),
                         lo95=lo, hi95=hi, share_positive=(diff > 0).mean()))

    # New places, single assignment: train on 2015 targets outside a fold, test on 2018
    # targets inside it.
    temporal = {k: np.full(len(y), np.nan) for k in
                ["mean-reversion floor", "boosting, static + previous",
                 "boosting, static + previous + series"]}
    for f in np.unique(fold):
        tr = (fold != f) & (d.times == 2015)
        te = (fold == f) & (d.times == 2018)
        temporal["mean-reversion floor"][te] = (LinearRegression()
                                               .fit(prev[tr], y[tr]).predict(prev[te]))
        for k, cols in [("boosting, static + previous", [static, prev]),
                        ("boosting, static + previous + series", [series, static, prev])]:
            X = np.hstack(cols)
            temporal[k][te] = baselines.fit_gradient_boosting(X[tr], y[tr]).predict(X[te])
    m = ~np.isnan(temporal["mean-reversion floor"])
    rows += [dict(comparison=f"new places skill: {k}",
                  difference=evaluate.skill_score(y0[m], y[m], v[m]))
             for k, v in temporal.items()]

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "campaign_proxy.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
