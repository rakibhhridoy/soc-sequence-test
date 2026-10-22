"""Tuned gradient boosting with observed climate, the variant climate_comparison.py lacked.

With ERA5-Land climate added, the hybrid led boosting by 0.037, but that boosting ran with
fixed settings and lost skill when the fifteen climate summaries were added. This refits
boosting with the nested selection of baselines.fit_tuned_boosting on the climate panel,
on the single assignment of folds used by climate_comparison.py, and pairs it with the
saved hybrid-with-climate predictions and with tuned Landsat-only boosting from
boosting_features.py (replicate 0 is the same assignment).

    python scripts/climate_comparison.py     # first, for the hybrid with climate
    python scripts/boosting_features.py      # first, for tuned Landsat-only boosting
    python scripts/climate_tuned.py
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, skill, block_bootstrap  # noqa: E402


def main() -> int:
    data = load_npz(config.DATA_PROCESSED / "panel_mineral_climate.npz")
    config.seed_everything()
    folds, _, groups = train.spatial_folds(data, n_folds=4)
    ref = np.load(PRED_DIR / "spatial_mineral.npz")
    for i, (_, te) in enumerate(folds):          # the pairing is only valid on identical folds
        assert (ref["fold"][te] == i).all(), "folds differ from the reference run"
    assert (groups == ref["groups"]).all()
    bf = np.load(PRED_DIR / "boosting_features_rep0.npz")
    assert np.allclose(bf["summary__fixed"], ref["gradient_boosting__with_previous_value"])

    X = np.hstack([baselines.summary_features(data.x_dyn), data.x_static, data.y_prev[:, None]])
    tuned, log = np.full(len(data), np.nan), []
    for tr, te in folds:
        m, rec = baselines.fit_tuned_boosting(X, data.y, tr, groups, seed=0)
        log.append(rec)
        tuned[te] = m.predict(X[te])
    clim = np.load(PRED_DIR / "spatial_mineral_climate.npz")
    preds = {
        "hybrid, with climate": clim["hybrid_climate"],
        "hybrid, Landsat only": ref["hybrid__full"],
        "boosting tuned, with climate": tuned,
        "boosting tuned, Landsat only": bf["summary__tuned"],
    }
    for k, p in preds.items():
        print(f"  {k:32s} skill {skill(data.y_prev, data.y, p):+.3f}")
    np.savez(PRED_DIR / "climate_tuned.npz", boosting_tuned_climate=tuned)
    pairs = [("hybrid, with climate", "boosting tuned, with climate"),
             ("boosting tuned, with climate", "boosting tuned, Landsat only"),
             ("hybrid, with climate", "hybrid, Landsat only")]
    df = pd.DataFrame(block_bootstrap(groups, data.y_prev, data.y, preds, pairs))
    out = config.TABLES / "climate_tuned.csv"
    df.to_csv(out, index=False)
    (config.TABLES / "climate_tuned_tuning.json").write_text(json.dumps(log, indent=1))
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
