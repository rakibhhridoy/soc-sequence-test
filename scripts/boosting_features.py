"""Gradient boosting with timing-aware features and its own nested tuning.

The headline comparison gives boosting five statistics per channel (mean, standard
deviation, minimum, maximum, slope), which discard seasonality and timing, and runs it
with fixed settings while the hybrid's are selected inside each training fold. Both choices
favour the hybrid. This refits boosting with the features of baselines.temporal_features
and with the nested selection of baselines.fit_tuned_boosting, in the four combinations,
on exactly the folds of robustness.py (five spatial replicates, three of them also run
forward in time at new places) and of the same-point temporal design, and pairs each
variant with the saved hybrid predictions for block-bootstrap intervals.

    python scripts/boosting_features.py
"""
from __future__ import annotations
import sys, pathlib, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, blocking, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR, skill, block_bootstrap  # noqa: E402
from robustness import R_SPATIAL, R_EXTRA, N_BOOT, groups_in, gbm  # noqa: E402

VARIANTS = ["summary, fixed", "summary, tuned", "timing, fixed", "timing, tuned"]


def predict(X, y, pairs, tuned, groups, seed, log):
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
    tail = [data.x_static, y0[:, None]]
    X = {"summary": np.hstack([baselines.summary_features(data.x_dyn), *tail]),
         "timing": np.hstack([baselines.temporal_features(data.x_dyn), *tail])}
    print(f"{len(y):,} observations | features: summary {X['summary'].shape[1]}, "
          f"timing {X['timing'].shape[1]}")
    log = {}

    # spatial blocking and forward in time at new places, robustness.py's folds
    reps = []
    for r in range(R_SPATIAL):
        t0 = time.time()
        saved = np.load(PRED_DIR / f"robustness_rep{r}.npz")
        folds = list(blocking.spatial_block_folds(groups, n_folds=4, seed=r))
        p = {"hybrid": saved["hybrid"], "linear": saved["linear"]}
        designs = [("", folds)]
        if r < R_EXTRA:
            st = [(np.where((groups_in(folds, f, len(y)) == 0) & (year == 2015))[0],
                   np.where((groups_in(folds, f, len(y)) == 1) & (year == 2018))[0])
                  for f in range(len(folds))]
            designs.append(("st ", st))
            p["st hybrid"], p["st linear"] = saved["st_hybrid"], saved["st_linear"]
        for prefix, pairs in designs:
            for v in VARIANTS:
                feats, mode = v.split(", ")
                lg = log.setdefault(f"{prefix}{v} rep{r}", [])
                p[prefix + v] = predict(X[feats], y, pairs, mode == "tuned", groups, r, lg)
        check = np.nanmax(np.abs(p["summary, fixed"] - saved["boosting"]))
        assert check < 1e-9, f"replicate {r} does not reproduce saved boosting ({check})"
        if r < R_EXTRA:
            check = np.nanmax(np.abs(p["st summary, fixed"] - saved["st_boosting"]))
            assert check < 1e-9, f"replicate {r} st does not reproduce saved boosting ({check})"
        reps.append(p)
        np.savez(PRED_DIR / f"boosting_features_rep{r}.npz",
                 **{k.replace(", ", "__").replace(" ", "_"): v for k, v in p.items()})
        print(f"replicate {r}: " + ", ".join(
            f"{k} {skill(y0[~np.isnan(v)], y[~np.isnan(v)], v[~np.isnan(v)]):.3f}"
            for k, v in p.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)

    quantities = {f"{pre}{v}": (f"{pre}{v}", None) for pre in ("", "st ") for v in VARIANTS}
    quantities.update({"hybrid": ("hybrid", None), "linear": ("linear", None),
                       "st hybrid": ("st hybrid", None), "st linear": ("st linear", None)})
    for pre in ("", "st "):
        for v in VARIANTS:
            quantities[f"{pre}hybrid minus {pre}{v}"] = (f"{pre}hybrid", f"{pre}{v}")
            quantities[f"{pre}{v} minus {pre}linear"] = (f"{pre}{v}", f"{pre}linear")
        quantities[f"{pre}timing, tuned minus {pre}summary, fixed"] = (
            f"{pre}timing, tuned", f"{pre}summary, fixed")

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
                         or " st " in q else "spatial blocking",
                         quantity=q.replace("st ", ""), estimate=np.mean(est), ci95_low=lo,
                         ci95_high=hi, replicates=len(est),
                         share_above_zero=float((v > 0).mean()) if b else np.nan))

    # forward in time at the same points, the single split of temporal_validation.py
    tr, te = np.where(year == 2015)[0], np.where(year == 2018)[0]
    saved = np.load(PRED_DIR / "temporal_mineral.npz")
    assert (saved["test_idx"] == te).all()
    preds = {"hybrid": saved["hybrid__full"]}
    for v in VARIANTS:
        feats, mode = v.split(", ")
        lg = log.setdefault(f"same points {v}", [])
        preds[v] = predict(X[feats], y, [(tr, te)], mode == "tuned", groups, 0, lg)[te]
    check = np.abs(preds["summary, fixed"] - saved["gradient_boosting__with_previous_value"]).max()
    assert check < 1e-9, f"same-point design does not reproduce saved boosting ({check})"
    pairs = [("hybrid", v) for v in VARIANTS] + [("timing, tuned", "summary, fixed")]
    for r in block_bootstrap(groups[te], y0[te], y[te], preds, pairs):
        rows.append(dict(design="same points, later survey", quantity=r["quantity"],
                         estimate=r["estimate"], ci95_low=r["ci_low"], ci95_high=r["ci_high"],
                         replicates=1, share_above_zero=r.get("share_above_zero", np.nan)))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "boosting_features.csv"
    df.to_csv(out, index=False)
    (config.TABLES / "boosting_features_tuning.json").write_text(json.dumps(log, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
