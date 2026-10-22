"""Does the graph's gain come from the architecture or from local information?

The geographic graph extension raised skill by about 0.02 over its edge-free control, and
the manuscript read that gain as local information about the level to which carbon
reverts, without testing it. This gives the tabular models the same information. Each
observation receives the mean previous log carbon and the mean static properties of its
neighbours, taken over exactly the edges the graph used (eight nearest points on the
ground, with edges between observations of one point and edges from a later survey round
removed by graphs.causal_edges), so nothing reaches a model that the graph could not see.

The features are added to the mean-reversion floor and to gradient boosting, fixed and
tuned, on the five spatial replicates of robustness.py and on the three that are also run
forward in time at new places. If boosting with neighbour features gains what the graph
gained, the gain belongs to the information and not to message passing.

    python scripts/neighbour_features.py
"""
from __future__ import annotations
import sys, pathlib, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, blocking, train, graphs  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, skill  # noqa: E402
from robustness import R_SPATIAL, R_EXTRA, N_BOOT, groups_in  # noqa: E402

K = 8


def neighbour_means(data, k=K):
    """Mean previous value and static properties over each observation's causal neighbours."""
    ei = graphs.causal_edges(graphs.geographic_graph(data.coords, k), data.point_id, data.times)
    src, dst = ei.numpy()
    n = len(data)
    count = np.bincount(dst, minlength=n).astype(float)
    feats = np.column_stack([data.y_prev, data.x_static])
    sums = np.zeros((n, feats.shape[1]))
    np.add.at(sums, dst, feats[src])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / count[:, None]
    return means, count


def linear(X, y, pairs):
    """Least squares on the given columns plus an intercept, with column means for gaps."""
    out = np.full(len(y), np.nan)
    for tr, te in pairs:
        Xf = X.copy()
        fill = np.nanmean(Xf[tr], axis=0)
        gaps = np.isnan(Xf)
        Xf[gaps] = np.take(fill, np.where(gaps)[1])
        A = np.column_stack([np.ones(len(y)), Xf])
        b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        out[te] = A[te] @ b
    return out


def boosting(X, y, pairs, tuned, groups, seed, log):
    out = np.full(len(y), np.nan)
    for tr, te in pairs:
        if tuned:
            m, rec = baselines.fit_tuned_boosting(X, y, tr, groups, seed=seed)
            log.append(rec)
        else:
            m = baselines.fit_gradient_boosting(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def main() -> int:
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    y, y0, year = data.y, data.y_prev, data.times
    _, _, groups = train.spatial_folds(data, n_folds=4)
    nb, count = neighbour_means(data)
    print(f"{len(y):,} observations | neighbours per observation: median {np.median(count):.0f}, "
          f"none for {(count == 0).sum()}")

    campaign = (year == 2018).astype(float)[:, None]
    summ = baselines.summary_features(data.x_dyn)
    # the previous value stays the last column, as fit_tuned_boosting requires
    X = {"boosting": np.hstack([summ, data.x_static, y0[:, None]]),
         "boosting + neighbours": np.hstack([summ, data.x_static, nb, y0[:, None]])}
    L = {"linear": np.column_stack([y0, campaign]),
         "linear + neighbour level": np.column_stack([y0, campaign, nb[:, 0]]),
         "linear + neighbours": np.column_stack([y0, campaign, nb])}
    log = {}

    reps = []
    for r in range(R_SPATIAL):
        t0 = time.time()
        saved = np.load(PRED_DIR / f"robustness_rep{r}.npz")
        folds = list(blocking.spatial_block_folds(groups, n_folds=4, seed=r))
        p = {"hybrid": saved["hybrid"]}
        designs = [("", folds)]
        if r < R_EXTRA:
            st = [(np.where((groups_in(folds, f, len(y)) == 0) & (year == 2015))[0],
                   np.where((groups_in(folds, f, len(y)) == 1) & (year == 2018))[0])
                  for f in range(len(folds))]
            designs.append(("st ", st))
            p["st hybrid"] = saved["st_hybrid"]
        for pre, pairs in designs:
            for name, Xl in L.items():
                p[pre + name] = linear(Xl, y, pairs)
            for name, Xb in X.items():
                p[f"{pre}{name}, fixed"] = boosting(Xb, y, pairs, False, groups, r, [])
                p[f"{pre}{name}, tuned"] = boosting(
                    Xb, y, pairs, True, groups, r, log.setdefault(f"{pre}{name} rep{r}", []))
        check = np.nanmax(np.abs(p["linear"] - saved["linear"]))
        assert check < 1e-8, f"replicate {r} does not reproduce the saved floor ({check})"
        check = np.nanmax(np.abs(p["boosting, fixed"] - saved["boosting"]))
        assert check < 1e-9, f"replicate {r} does not reproduce saved boosting ({check})"
        reps.append(p)
        np.savez(PRED_DIR / f"neighbour_features_rep{r}.npz",
                 **{k.replace(", ", "__").replace(" + ", "_plus_").replace(" ", "_"): v
                    for k, v in p.items()})
        print(f"replicate {r}: " + ", ".join(
            f"{k} {skill(y0[~np.isnan(v)], y[~np.isnan(v)], v[~np.isnan(v)]):.3f}"
            for k, v in p.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)

    singles = ["hybrid", *L, *[f"{n}, {m}" for n in X for m in ("fixed", "tuned")]]
    pairs_q = [("linear + neighbour level", "linear"), ("linear + neighbours", "linear"),
               ("boosting + neighbours, fixed", "boosting, fixed"),
               ("boosting + neighbours, tuned", "boosting, tuned"),
               ("boosting + neighbours, fixed", "hybrid"),
               ("boosting + neighbours, tuned", "hybrid"),
               ("linear + neighbours", "hybrid")]
    quantities = {}
    for pre in ("", "st "):
        for s in singles:
            quantities[pre + s] = (pre + s, None)
        for a, b in pairs_q:
            quantities[f"{pre}{a} minus {pre}{b}"] = (pre + a, pre + b)

    # the resampling scheme of robustness.py: blocks, then one replicate per draw
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
            s = skill(y0[ok], y[ok], p[a][ok])
            if b is not None:
                s -= skill(y0[ok], y[ok], p[b][ok])
            draws[q].append(s)
    rows = []
    for q, (a, b) in quantities.items():
        v = np.asarray(draws[q])
        est = []
        for p in reps:
            if a in p and (b is None or b in p):
                ok = ~np.isnan(p[a])
                e = skill(y0[ok], y[ok], p[a][ok])
                if b is not None:
                    e -= skill(y0[ok], y[ok], p[b][ok])
                est.append(e)
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(design="new places, later survey" if q.startswith("st ")
                         else "spatial blocking", quantity=q.replace("st ", ""),
                         estimate=np.mean(est), ci95_low=lo, ci95_high=hi,
                         replicates=len(est),
                         share_above_zero=float((v > 0).mean()) if b else np.nan))

    # replicate 0 is the fold assignment of the graph analysis, so compare directly
    graph = np.load(PRED_DIR / "graph_mineral_geographic.npz")
    g = graph["mu"]
    ref = np.load(PRED_DIR / "spatial_mineral.npz")
    same = (graph["fold"] == ref["fold"]).all() and np.allclose(
        reps[0]["boosting, fixed"], ref["gradient_boosting__with_previous_value"], equal_nan=True)
    if same:
        p0 = reps[0]
        for name in ("boosting + neighbours, fixed", "boosting + neighbours, tuned",
                     "linear + neighbours"):
            ok = ~np.isnan(g) & ~np.isnan(p0[name])
            rows.append(dict(design="spatial blocking, replicate 0",
                             quantity=f"graph, geographic minus {name}",
                             estimate=skill(y0[ok], y[ok], g[ok]) - skill(y0[ok], y[ok], p0[name][ok]),
                             ci95_low=np.nan, ci95_high=np.nan, replicates=1,
                             share_above_zero=np.nan))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "neighbour_features.csv"
    df.to_csv(out, index=False)
    (config.TABLES / "neighbour_features_tuning.json").write_text(json.dumps(log, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
