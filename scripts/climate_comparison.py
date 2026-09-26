"""Does observed climate change the result? Paired against the Landsat-only runs.

Refits the full hybrid and both gradient-boosting variants on `panel_mineral_climate.npz`,
which adds ERA5-Land temperature, precipitation and soil water to the Landsat channels,
under the same blocked folds. The Landsat-only predictions saved by
`bootstrap_comparison.py` are the paired reference, so each difference is bootstrapped
over the same resampled blocks.

    python scripts/bootstrap_comparison.py     # first, for the reference predictions
    python scripts/climate_comparison.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, cfg, gbm_oof, skill, block_bootstrap  # noqa: E402


def main() -> int:
    device = config.device()
    data = load_npz(config.DATA_PROCESSED / "panel_mineral_climate.npz")
    config.seed_everything()
    folds, _ = train.spatial_folds(data, n_folds=4)
    ref = np.load(PRED_DIR / "spatial_mineral.npz")
    ref_folds = ref["fold"]
    for i, (_, te) in enumerate(folds):          # the pairing is only valid on identical folds
        assert (ref_folds[te] == i).all(), "folds differ from the reference run"
    groups = ref["groups"]
    print(f"{len(data):,} observations, {data.x_dyn.shape[1]} dynamic channels")

    summ = baselines.summary_features(data.x_dyn)
    preds = {
        "hybrid, Landsat only": ref["hybrid__full"],
        "gradient boosting with previous value, Landsat only":
            ref["gradient_boosting__with_previous_value"],
        "gradient boosting no previous value, Landsat only":
            ref["gradient_boosting__no_previous_value"],
        "gradient boosting with previous value, with climate": gbm_oof(
            np.hstack([summ, data.x_static, data.y_prev[:, None]]), data.y, folds),
        "gradient boosting no previous value, with climate": gbm_oof(
            np.hstack([summ, data.x_static]), data.y, folds),
        "hybrid, with climate": train.cross_validate(
            train.with_previous(data), cfg(), folds, device, n_ensemble=3)["mu"],
    }
    for k, p in preds.items():
        print(f"  {k:55s} skill {skill(data.y_prev, data.y, p):+.3f}")
    np.savez(PRED_DIR / "spatial_mineral_climate.npz", hybrid_climate=preds["hybrid, with climate"])

    pairs = [("hybrid, with climate", "hybrid, Landsat only"),
             ("gradient boosting with previous value, with climate",
              "gradient boosting with previous value, Landsat only"),
             ("hybrid, with climate", "gradient boosting with previous value, with climate"),
             ("hybrid, with climate", "gradient boosting no previous value, with climate")]
    df = pd.DataFrame(block_bootstrap(groups, data.y_prev, data.y, preds, pairs))
    out = config.TABLES / "climate_comparison.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
