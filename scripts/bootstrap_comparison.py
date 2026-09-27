"""Uncertainty on the key skill differences, by spatial block bootstrap.

The headline comparisons differ by less than 0.01 skill, so a point estimate alone cannot
say whether they are distinguishable. This refits the models that matter on the same
blocked folds as Table 1, keeps their held-out predictions, and resamples whole spatial
blocks with replacement. Resampling blocks rather than observations keeps the spatial
correlation inside each replicate, so the interval is not made artificially narrow by
treating neighbouring points as independent.

It also fits gradient boosting with the previous carbon value as an input. The baseline in
`run_ablations.py` is given the summary statistics and soil properties but not the previous
value, which the hybrid receives, so that comparison is not like for like.

    python scripts/bootstrap_comparison.py
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, blocking, evaluate, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402

N_BOOT = 2000
PRED_DIR = config.TABLES.parent / "predictions"      # gitignored: keyed to LUCAS observations


def cfg(**kw) -> train.TrainConfig:
    # width, learning rate and dropout come from the nested search of each fold
    return train.TrainConfig(epochs=200, **kw)


def gbm_oof(X, y, folds):
    mu = np.full(len(y), np.nan)
    for tr, te in folds:
        mu[te] = baselines.fit_gradient_boosting(X[tr], y[tr]).predict(X[te])
    return mu


def skill(y_prev, y, pred):
    return 1.0 - evaluate.rmse(y, pred) / evaluate.rmse(y, y_prev)


def block_bootstrap(groups, y_prev, y, preds: dict, pairs, n_boot=N_BOOT, seed=0):
    """Percentile intervals for each model's skill and each paired difference."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ids}
    draws = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ids, size=len(ids), replace=True)])
        for k, p in preds.items():
            draws[k][b] = skill(y_prev[idx], y[idx], p[idx])
    rows = []
    for k, p in preds.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        rows.append(dict(quantity=k, estimate=skill(y_prev, y, p), ci_low=lo, ci_high=hi))
    for a, b in pairs:
        d = draws[a] - draws[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(quantity=f"{a} minus {b}",
                         estimate=skill(y_prev, y, preds[a]) - skill(y_prev, y, preds[b]),
                         ci_low=lo, ci_high=hi, share_above_zero=float((d > 0).mean())))
    return rows


def spatial(device):
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    config.seed_everything()
    folds, block, groups = train.spatial_folds(data, n_folds=4)
    d_prev = train.with_previous(data)
    fc = tuning.fold_configs(d_prev, folds, groups, device, tag="mineral spatial")
    summ = baselines.summary_features(data.x_dyn)
    print(f"spatial: {len(data):,} observations, {len(np.unique(groups))} blocks")

    preds = {}
    preds["gradient boosting, no previous value"] = gbm_oof(np.hstack([summ, data.x_static]), data.y, folds)
    preds["gradient boosting, with previous value"] = gbm_oof(
        np.hstack([summ, data.x_static, data.y_prev[:, None]]), data.y, folds)
    for name, kw, d in (
        ("hybrid, full", {}, d_prev),
        ("hybrid, no recurrent decoder", dict(use_recurrent=False), d_prev),
    ):
        t0 = time.time()
        preds[name] = train.cross_validate(d, cfg(**kw), folds, device, n_ensemble=3,
                                           groups=groups, fold_cfgs=fc)["mu"]
        print(f"  {name:30s} {time.time()-t0:.0f}s")
    zeros = train.with_previous(load_npz(config.DATA_PROCESSED / "panel_mineral.npz"))
    zeros.x_dyn = np.zeros_like(zeros.x_dyn)
    preds["hybrid, series replaced by zeros"] = train.cross_validate(
        zeros, cfg(), folds, device, n_ensemble=3, groups=groups, fold_cfgs=fc)["mu"]

    for k, p in preds.items():
        assert not np.isnan(p).any(), k
        print(f"  {k:40s} skill {skill(data.y_prev, data.y, p):+.3f}")
    PRED_DIR.mkdir(exist_ok=True)
    np.savez(PRED_DIR / "spatial_mineral.npz", groups=groups, fold=_fold_index(folds, len(data)),
             **{k.replace(", ", "__").replace(" ", "_"): v for k, v in preds.items()})

    per_fold = []
    for f, (_, te) in enumerate(folds):
        per_fold.append({"fold": f, "n": len(te),
                         **{k: skill(data.y_prev[te], data.y[te], p[te]) for k, p in preds.items()}})
    pd.DataFrame(per_fold).to_csv(config.TABLES / "bootstrap_per_fold.csv", index=False)

    pairs = [("hybrid, full", "gradient boosting, no previous value"),
             ("hybrid, full", "gradient boosting, with previous value"),
             ("hybrid, no recurrent decoder", "hybrid, full"),
             ("hybrid, full", "hybrid, series replaced by zeros")]
    return [dict(scheme="spatial", **r) for r in block_bootstrap(groups, data.y_prev, data.y, preds, pairs)]


def temporal(device):
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    config.seed_everything()
    tr, te = np.where(data.times == 2015)[0], np.where(data.times == 2018)[0]
    _, _, groups = train.spatial_folds(data, n_folds=4)
    summ = baselines.summary_features(data.x_dyn)
    print(f"temporal: train {len(tr):,}, test {len(te):,}")
    d_prev = train.with_previous(data)
    fc = tuning.fold_configs(d_prev, [(tr, te)], groups, device, tag="mineral temporal")

    preds = {}
    for name, X in (("gradient boosting, no previous value", np.hstack([summ, data.x_static])),
                    ("gradient boosting, with previous value",
                     np.hstack([summ, data.x_static, data.y_prev[:, None]]))):
        preds[name] = baselines.fit_gradient_boosting(X[tr], data.y[tr]).predict(X[te])
    out = train.cross_validate(d_prev, cfg(), [(tr, te)], device, n_ensemble=3,
                               groups=groups, fold_cfgs=fc)
    preds["hybrid, full"] = out["mu"][te]
    for k, p in preds.items():
        print(f"  {k:40s} skill {skill(data.y_prev[te], data.y[te], p):+.3f}")
    np.savez(PRED_DIR / "temporal_mineral.npz", test_idx=te,
             **{k.replace(", ", "__").replace(" ", "_"): v for k, v in preds.items()})
    pairs = [("hybrid, full", "gradient boosting, no previous value"),
             ("hybrid, full", "gradient boosting, with previous value")]
    return [dict(scheme="temporal", **r)
            for r in block_bootstrap(groups[te], data.y_prev[te], data.y[te], preds, pairs)]


def _fold_index(folds, n):
    f = np.full(n, -1)
    for i, (_, te) in enumerate(folds):
        f[te] = i
    return f


def main() -> int:
    device = config.device()
    rows = spatial(device) + temporal(device)
    df = pd.DataFrame(rows)
    out = config.TABLES / "bootstrap_comparison.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
