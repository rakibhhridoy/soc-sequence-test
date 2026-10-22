"""The climate comparison over the five replicates of robustness.py.

climate_comparison.py and climate_tuned.py ran on the single assignment of folds, where the
hybrid with ERA5-Land climate led tuned boosting with climate by 0.015 (0.003 to 0.028),
the only comparison across space whose interval excluded zero. This repeats it on the five
replicates of robustness.py, each with its own network seed and assignment of blocks to
folds, and pools them with the same block bootstrap. The Landsat-only selection of each
fold is reused from the cache, so the climate channels are the only change, as in
climate_comparison.py. Replicate 0 is the single assignment, so its saved predictions are
reused. Each replicate is saved as it finishes and skipped on a rerun.

    python scripts/climate_replicates.py
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, blocking, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, cfg, skill  # noqa: E402
from robustness import R_SPATIAL, N_BOOT, MARGIN, gbm  # noqa: E402


def tuned_gbm(X, y, pairs, groups, seed):
    out = np.full(len(y), np.nan)
    for tr, te in pairs:
        m, _ = baselines.fit_tuned_boosting(X, y, tr, groups, seed=seed)
        out[te] = m.predict(X[te])
    return out


def main() -> int:
    device = config.device()
    clim = load_npz(config.DATA_PROCESSED / "panel_mineral_climate.npz")
    landsat = train.with_previous(load_npz(config.DATA_PROCESSED / "panel_mineral.npz"))
    assert np.array_equal(clim.y, landsat.y) and np.array_equal(clim.y_prev, landsat.y_prev)
    d_clim = train.with_previous(clim)
    y, y0 = clim.y, clim.y_prev
    Xc = np.hstack([baselines.summary_features(clim.x_dyn), clim.x_static, y0[:, None]])
    _, _, groups = train.spatial_folds(clim, n_folds=4)
    print(f"{len(y):,} observations, {clim.x_dyn.shape[1]} dynamic channels", flush=True)

    reps = []
    for r in range(R_SPATIAL):
        rob = np.load(PRED_DIR / f"robustness_rep{r}.npz")
        bf = np.load(PRED_DIR / f"boosting_features_rep{r}.npz")
        path = PRED_DIR / f"climate_replicates_rep{r}.npz"
        p = {"hybrid": rob["hybrid"], "boosting": rob["boosting"],
             "boosting tuned": bf["summary__tuned"]}
        if path.exists():
            z = np.load(path)
            p.update({k.replace("_", " "): z[k] for k in z.files})
            reps.append(p)
            print(f"replicate {r}: loaded", flush=True)
            continue
        t0 = time.time()
        seed = 100 * r
        config.seed_everything(seed)
        folds = list(blocking.spatial_block_folds(groups, n_folds=4, seed=r))
        if r == 0:
            # the single assignment of climate_comparison.py and climate_tuned.py
            ref = np.load(PRED_DIR / "spatial_mineral.npz")
            for i, (_, te) in enumerate(folds):
                assert (ref["fold"][te] == i).all(), "replicate 0 is not the single assignment"
            new = {"hybrid climate": np.load(PRED_DIR / "spatial_mineral_climate.npz")["hybrid_climate"],
                   "boosting tuned climate": np.load(PRED_DIR / "climate_tuned.npz")["boosting_tuned_climate"]}
        else:
            fc = tuning.fold_configs(landsat, folds, groups, device, tag=f"mineral replicate {r}")
            new = {"hybrid climate": train.cross_validate(
                       d_clim, cfg(seed=seed), folds, device, n_ensemble=3, groups=groups,
                       fold_cfgs=fc)["mu"],
                   "boosting tuned climate": tuned_gbm(Xc, y, folds, groups, r)}
        new["boosting climate"] = gbm(Xc, y, folds)
        np.savez(path, **{k.replace(" ", "_"): v for k, v in new.items()})
        p.update(new)
        reps.append(p)
        print(f"replicate {r}: " + ", ".join(f"{k} {skill(y0, y, v):.3f}" for k, v in p.items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    quantities = {k: (k, None) for k in reps[0]}
    quantities.update({
        "hybrid climate minus boosting tuned climate": ("hybrid climate", "boosting tuned climate"),
        "hybrid climate minus boosting climate": ("hybrid climate", "boosting climate"),
        "hybrid climate minus hybrid": ("hybrid climate", "hybrid"),
        "boosting tuned climate minus boosting tuned": ("boosting tuned climate", "boosting tuned"),
        "boosting climate minus boosting": ("boosting climate", "boosting"),
        "hybrid minus boosting tuned": ("hybrid", "boosting tuned"),
    })
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
        row = dict(quantity=q, estimate=np.mean([pr[q] for pr in per_rep]), ci95_low=lo95,
                   ci95_high=hi95, ci90_low=lo90, ci90_high=hi90, replicates=len(reps))
        if b is not None:
            row["equivalent_at_0.02"] = bool(lo90 > -MARGIN and hi90 < MARGIN)
            row["share_above_zero"] = float((v > 0).mean())
        rows.append(row)
    pd.DataFrame(per_rep).assign(replicate=range(len(reps))).to_csv(
        config.TABLES / "climate_replicates_per_replicate.csv", index=False)
    df = pd.DataFrame(rows)
    out = config.TABLES / "climate_replicates.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
