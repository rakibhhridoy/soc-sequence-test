"""Sensitivity of the headline comparison to the spatial block size.

The variogram range of 780 km leaves 18 blocks, so the block bootstrap resamples few units
and a reviewer may ask whether the comparison depends on that choice. This repeats the
central comparison of robustness.py with smaller blocks: the hybrid with its nested
selection, gradient boosting with fixed settings and with nested selection, and the
mean-reversion floor, over several replicates of the assignment of blocks to folds, with
intervals from a bootstrap over the smaller blocks. Smaller blocks hold out less of each
point's neighbourhood, so skill is expected to rise slightly. The question is whether the
difference between the methods moves.

Each replicate is saved as it finishes and skipped on a rerun, so the script can resume.

    python scripts/block_sensitivity.py --block-km 390 --replicates 3
"""
from __future__ import annotations
import argparse, sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, blocking, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, cfg, skill  # noqa: E402
from robustness import N_BOOT, MARGIN, linear_prev, gbm  # noqa: E402


def tuned_gbm(X, y, pairs, groups, seed):
    out = np.full(len(y), np.nan)
    for tr, te in pairs:
        m, _ = baselines.fit_tuned_boosting(X, y, tr, groups, seed=seed)
        out[te] = m.predict(X[te])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-km", type=float, default=390)
    ap.add_argument("--replicates", type=int, default=3)
    args = ap.parse_args()
    device = config.device()
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    d_prev = train.with_previous(data)
    Xg = np.hstack([baselines.summary_features(data.x_dyn), data.x_static, data.y_prev[:, None]])
    groups = blocking.spatial_blocks(data.coords, args.block_km * 1000)
    tag = f"blocks{int(args.block_km)}km"
    print(f"{len(data):,} observations, {len(np.unique(groups))} blocks of {args.block_km:.0f} km",
          flush=True)

    reps = []
    for r in range(args.replicates):
        path = PRED_DIR / f"block_sensitivity_{tag}_rep{r}.npz"
        if path.exists():
            z = np.load(path)
            reps.append({k.replace("_", " "): z[k] for k in z.files})
            print(f"replicate {r}: loaded", flush=True)
            continue
        seed = 100 * r
        config.seed_everything(seed)
        folds = list(blocking.spatial_block_folds(groups, n_folds=4, seed=r))
        t0 = time.time()
        fc = tuning.fold_configs(d_prev, folds, groups, device, tag=f"mineral {tag} replicate {r}")
        p = {"hybrid": train.cross_validate(d_prev, cfg(seed=seed), folds, device, groups=groups,
                                            fold_cfgs=fc, n_ensemble=3)["mu"],
             "boosting": gbm(Xg, data.y, folds),
             "boosting tuned": tuned_gbm(Xg, data.y, folds, groups, r),
             "linear": linear_prev(data.y_prev, data.y, data.times, folds)}
        np.savez(path, **{k.replace(" ", "_"): v for k, v in p.items()})
        reps.append(p)
        print(f"replicate {r}: " + ", ".join(
            f"{k} {skill(data.y_prev, data.y, v):.3f}" for k, v in p.items())
            + f"  ({time.time()-t0:.0f}s)", flush=True)

    quantities = {"hybrid": ("hybrid", None), "boosting": ("boosting", None),
                  "boosting tuned": ("boosting tuned", None), "linear": ("linear", None),
                  "hybrid minus boosting": ("hybrid", "boosting"),
                  "hybrid minus boosting tuned": ("hybrid", "boosting tuned"),
                  "hybrid minus linear": ("hybrid", "linear"),
                  "boosting minus linear": ("boosting", "linear")}
    y0, y = data.y_prev, data.y
    per_rep = [{q: skill(y0, y, p[a]) - (skill(y0, y, p[b]) if b else 0)
                for q, (a, b) in quantities.items()} for p in reps]
    rng = np.random.default_rng(0)
    ids = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ids}
    draws = {q: [] for q in quantities}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.choice(ids, size=len(ids), replace=True)])
        p = reps[rng.integers(len(reps))]
        for q, (a, b) in quantities.items():
            s = skill(y0[idx], y[idx], p[a][idx])
            if b is not None:
                s -= skill(y0[idx], y[idx], p[b][idx])
            draws[q].append(s)
    rows = []
    for q, (a, b) in quantities.items():
        v = np.asarray(draws[q])
        lo95, hi95 = np.percentile(v, [2.5, 97.5])
        lo90, hi90 = np.percentile(v, [5, 95])
        row = dict(block_km=args.block_km, blocks=len(ids), quantity=q,
                   estimate=np.mean([pr[q] for pr in per_rep]), ci95_low=lo95, ci95_high=hi95,
                   ci90_low=lo90, ci90_high=hi90, replicates=len(reps))
        if b is not None:
            row["equivalent_at_0.02"] = bool(lo90 > -MARGIN and hi90 < MARGIN)
            row["share_above_zero"] = float((v > 0).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    out = config.TABLES / f"block_sensitivity_{tag}.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
