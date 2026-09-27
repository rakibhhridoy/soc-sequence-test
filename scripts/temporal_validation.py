"""Forward temporal validation.

Spatial blocking asks whether a model generalises to new places. This asks whether it
generalises to a later time, by training only on the 2015 targets and testing on the 2018
targets, so every training observation precedes every test observation.

    python scripts/temporal_validation.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, evaluate, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402


def main() -> int:
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    config.seed_everything()
    tr = np.where(data.times == 2015)[0]
    te = np.where(data.times == 2018)[0]
    print(f"train on 2015 targets: {len(tr):,} | test on 2018 targets: {len(te):,}")

    rows = []
    rows.append(dict(model="persistence",
                     **_m(data.y_prev[te], data.y[te], data.y_prev[te])))

    feats = np.hstack([baselines.summary_features(data.x_dyn), data.x_static,
                       data.y_prev[:, None]])
    gbm = baselines.fit_gradient_boosting(feats[tr], data.y[tr])
    rows.append(dict(model="gradient boosting (summary stats)",
                     **_m(data.y_prev[te], data.y[te], gbm.predict(feats[te]))))

    d = train.with_previous(data)
    device = config.device()
    _, _, groups = train.spatial_folds(data, n_folds=4)
    fc = tuning.fold_configs(d, [(tr, te)], groups, device, tag="mineral temporal")
    out = train.cross_validate(d, train.TrainConfig(epochs=200), [(tr, te)], device,
                               n_ensemble=3, groups=groups, fold_cfgs=fc)
    rows.append(dict(model="full model (LSTM, tuned)",
                     **_m(data.y_prev[te], data.y[te], out["mu"][te])))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(config.TABLES / "temporal_validation.csv", index=False)
    print(f"\nwrote {config.TABLES/'temporal_validation.csv'}")
    return 0


def _m(y_prev, y, pred):
    return {"rmse": evaluate.rmse(y, pred), "ccc": evaluate.ccc(y, pred),
            "skill_vs_persistence": evaluate.skill_score(y_prev, y, pred),
            "n": len(y)}


if __name__ == "__main__":
    raise SystemExit(main())
