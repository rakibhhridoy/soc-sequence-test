"""Extend the design forward in time at new places from three replicates to five.

robustness.py runs that design, training on 2015 targets outside a fold and testing on
2018 targets inside it, for the first three of its five spatial replicates only, and
boosting_features.py follows it. This adds replicates 3 and 4, on their saved fold
assignments and cached hyperparameter selections, for the hybrid, the mean-reversion
floor and all four boosting variants, and writes the results into the saved prediction
files of both scripts. The pooled tables are then rebuilt without refitting:

    python scripts/temporal_extend.py
    python scripts/robustness.py --pool-only
    python scripts/boosting_features.py --pool-only
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, blocking, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, cfg, skill  # noqa: E402
from robustness import linear_prev, gbm, groups_in  # noqa: E402
from boosting_features import VARIANTS, predict  # noqa: E402

NEW = (3, 4)


def main() -> int:
    device = config.device()
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    d_prev = train.with_previous(data)
    y, y0, year = data.y, data.y_prev, data.times
    Xg = np.hstack([baselines.summary_features(data.x_dyn), data.x_static, y0[:, None]])
    X = {"summary": Xg,
         "timing": np.hstack([baselines.temporal_features(data.x_dyn), data.x_static, y0[:, None]])}
    _, _, groups = train.spatial_folds(data, n_folds=4)
    for r in NEW:
        rob_path = PRED_DIR / f"robustness_rep{r}.npz"
        bf_path = PRED_DIR / f"boosting_features_rep{r}.npz"
        rob, bf = dict(np.load(rob_path)), dict(np.load(bf_path))
        if "st_hybrid" in rob and "st_summary__fixed" in bf:
            print(f"replicate {r}: already extended", flush=True)
            continue
        t0 = time.time()
        seed = 100 * r
        config.seed_everything(seed)
        folds = list(blocking.spatial_block_folds(groups, n_folds=4, seed=r))
        # the spatial selection of this replicate, read from the cache
        fc = tuning.fold_configs(d_prev, folds, groups, device, tag=f"mineral replicate {r}")
        st = [(np.where((groups_in(folds, f, len(y)) == 0) & (year == 2015))[0],
               np.where((groups_in(folds, f, len(y)) == 1) & (year == 2018))[0])
              for f in range(len(folds))]
        te_all = np.concatenate([te for _, te in st])
        mask = np.zeros(len(y), bool)
        mask[te_all] = True
        if "st_hybrid" not in rob:
            new = {"st_hybrid": train.cross_validate(d_prev, cfg(seed=seed), st, device,
                                                     groups=groups, fold_cfgs=fc,
                                                     n_ensemble=3)["mu"],
                   "st_boosting": gbm(Xg, y, st),
                   "st_linear": linear_prev(y0, y, year, st)}
            rob.update({k: np.where(mask, v, np.nan) for k, v in new.items()})
            np.savez(rob_path, **rob)
        for v in VARIANTS:
            feats, mode = v.split(", ")
            bf["st_" + v.replace(", ", "__")] = predict(X[feats], y, st, mode == "tuned",
                                                       groups, r, [])
        bf["st_hybrid"], bf["st_linear"] = rob["st_hybrid"], rob["st_linear"]
        check = np.nanmax(np.abs(bf["st_summary__fixed"] - rob["st_boosting"]))
        assert check < 1e-9, f"replicate {r}: st boosting differs between scripts ({check})"
        np.savez(bf_path, **bf)
        ok = mask
        print(f"replicate {r}: st hybrid {skill(y0[ok], y[ok], rob['st_hybrid'][ok]):.3f}, "
              f"st linear {skill(y0[ok], y[ok], rob['st_linear'][ok]):.3f}, " + ", ".join(
                  f"st {v} {skill(y0[ok], y[ok], bf['st_' + v.replace(', ', '__')][ok]):.3f}"
                  for v in VARIANTS) + f"  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
