"""Robustness of the headline comparison to training randomness, fold assignment and design.

Addresses four weaknesses of the single-run comparison, all added after the first results
and labelled as such in the manuscript:

1. Training randomness and fold assignment. Every model is refitted under R replicates, each
   with its own network seed and its own assignment of spatial blocks to folds. Intervals pool
   the replicates with the spatial block bootstrap, so they carry both sources of variation.
2. A stronger floor than persistence. A linear fit of the new value on the previous value and
   the campaign measures how much skill regression towards the mean alone provides.
3. Equivalence. "Matches" in the protocol had no margin. A two one-sided test is reported at
   a margin of 0.02 skill, chosen after the results and stated as such, using the 90 %
   interval: the two methods are equivalent at that margin if it lies inside (-0.02, 0.02).
4. Temporal validation independent in space. Training uses 2015 targets outside a fold and
   testing uses 2018 targets inside it, so no test point contributes to training.

    python scripts/robustness.py
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, blocking, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, cfg, skill  # noqa: E402

R_SPATIAL, R_EXTRA = 5, 3
MARGIN = 0.02
N_BOOT = 2000


def linear_prev(y_prev, y, year, pairs):
    out = np.full(len(y), np.nan)
    X = np.column_stack([np.ones(len(y)), y_prev, year == 2018]).astype(float)
    for tr, te in pairs:
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        out[te] = X[te] @ b
    return out


def gbm(X, y, pairs):
    out = np.full(len(y), np.nan)
    for tr, te in pairs:
        out[te] = baselines.fit_gradient_boosting(X[tr], y[tr]).predict(X[te])
    return out


def main() -> int:
    device = config.device()
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    d_prev = train.with_previous(data)
    Xg = np.hstack([baselines.summary_features(data.x_dyn), data.x_static, data.y_prev[:, None]])
    year = data.times
    _, block = train.spatial_folds(data, n_folds=4)
    groups = blocking.spatial_blocks(data.coords, block)
    print(f"{len(data):,} observations, {len(np.unique(groups))} blocks")

    reps = []
    for r in range(R_SPATIAL):
        seed = 100 * r                       # ensemble members use seed, seed+1, seed+2
        config.seed_everything(seed)
        folds = list(blocking.spatial_block_folds(groups, n_folds=4, seed=r))
        t0 = time.time()
        p = {"hybrid": train.cross_validate(d_prev, cfg(seed=seed), folds, device, n_ensemble=3)["mu"],
             "boosting": gbm(Xg, data.y, folds),
             "linear": linear_prev(data.y_prev, data.y, year, folds)}
        if r < R_EXTRA:
            p["no decoder"] = train.cross_validate(
                d_prev, cfg(seed=seed, use_recurrent=False), folds, device, n_ensemble=3)["mu"]
            # forward in time and held out in space
            st = [(np.where((groups_in(folds, f, len(data)) == 0) & (year == 2015))[0],
                   np.where((groups_in(folds, f, len(data)) == 1) & (year == 2018))[0])
                  for f in range(len(folds))]
            te_all = np.concatenate([te for _, te in st])
            p["st hybrid"] = train.cross_validate(d_prev, cfg(seed=seed, patience=25), st, device,
                                                  n_ensemble=3)["mu"]
            p["st boosting"] = gbm(Xg, data.y, st)
            p["st linear"] = linear_prev(data.y_prev, data.y, year, st)
            for k in ("st hybrid", "st boosting", "st linear"):
                mask = np.zeros(len(data), bool); mask[te_all] = True
                p[k] = np.where(mask, p[k], np.nan)
        reps.append(p)
        print(f"replicate {r}: " + ", ".join(
            f"{k} {skill(data.y_prev[~np.isnan(v)], data.y[~np.isnan(v)], v[~np.isnan(v)]):.3f}"
            for k, v in p.items()) + f"  ({time.time()-t0:.0f}s)")
        np.savez(PRED_DIR / f"robustness_rep{r}.npz",
                 **{k.replace(" ", "_"): v for k, v in p.items()})

    per_rep = []
    for r, p in enumerate(reps):
        row = {"replicate": r}
        for k, v in p.items():
            ok = ~np.isnan(v)
            row[k] = skill(data.y_prev[ok], data.y[ok], v[ok])
        per_rep.append(row)
    pd.DataFrame(per_rep).to_csv(config.TABLES / "robustness_per_replicate.csv", index=False)

    quantities = {
        "hybrid": ("hybrid", None), "boosting": ("boosting", None), "linear": ("linear", None),
        "hybrid minus boosting": ("hybrid", "boosting"),
        "hybrid minus linear": ("hybrid", "linear"),
        "boosting minus linear": ("boosting", "linear"),
        "no decoder minus hybrid": ("no decoder", "hybrid"),
        "spatiotemporal hybrid": ("st hybrid", None),
        "spatiotemporal boosting": ("st boosting", None),
        "spatiotemporal linear": ("st linear", None),
        "spatiotemporal hybrid minus boosting": ("st hybrid", "st boosting"),
        "spatiotemporal hybrid minus linear": ("st hybrid", "st linear"),
    }
    rng = np.random.default_rng(0)
    ids = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ids}
    draws = {q: [] for q in quantities}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.choice(ids, size=len(ids), replace=True)])
        for q, (a, b) in quantities.items():
            avail = [p for p in reps if a in p and (b is None or b in p)]
            p = avail[rng.integers(len(avail))]
            ok = idx[~np.isnan(p[a][idx])]
            s = skill(data.y_prev[ok], data.y[ok], p[a][ok])
            if b is not None:
                s -= skill(data.y_prev[ok], data.y[ok], p[b][ok])
            draws[q].append(s)
    rows = []
    for q, v in draws.items():
        v = np.asarray(v)
        a, b = quantities[q]
        est = np.mean([r[a] - (r[b] if b else 0) for r in per_rep if a in r and (b is None or b in r)])
        lo95, hi95 = np.percentile(v, [2.5, 97.5])
        lo90, hi90 = np.percentile(v, [5, 95])
        row = dict(quantity=q, estimate=est, ci95_low=lo95, ci95_high=hi95,
                   ci90_low=lo90, ci90_high=hi90,
                   replicates=sum(1 for r in per_rep if a in r and (b is None or b in r)))
        if b is not None:
            row["equivalent_at_0.02"] = bool(lo90 > -MARGIN and hi90 < MARGIN)
            row["share_above_zero"] = float((v > 0).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    out = config.TABLES / "robustness.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out}")
    return 0


def groups_in(folds, f, n):
    """1 for the test side of fold f, 0 for its training side."""
    z = np.zeros(n, int)
    z[folds[f][1]] = 1
    return z


if __name__ == "__main__":
    raise SystemExit(main())
