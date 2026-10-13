"""How much of the measured change at a LUCAS point is real, whether it depends on the level.

Uses the error variance identified in noise_ceiling.py, var(e) = -cov(change 2009-2015,
change 2015-2018) at the 8,123 points with three mineral campaigns, for three questions.

1. Signal and noise. With errors independent between visits, the mean squared measured
   change over an interval is the mean squared true change plus 2 var(e), so 2 var(e) /
   mean(change^2) is the share of it that is noise. As var(e) is a lower bound, so is the
   share. Reported for the six-year (2009-2015) and three-year (2015-2018) intervals.

2. Dependence of change on the level. Regressing change on the starting value confounds a
   real dependence with reversion towards the mean, because the starting error enters both.
   The 2009 value carries none of the errors of 2015 and 2018, so it instruments the 2015
   value in a regression of the 2015-2018 change on it,
       b_iv = cov(change 2015-2018, x_2009) / cov(x_2015, x_2009),
   the slope of true change on the true starting level if that change is linear in it and
   otherwise unrelated to the 2009 value. Beside it are the naive slope on x_2015 and the
   slope on the mean of x_2015 and x_2018 (Lark et al. 2006).

3. The ceiling on skill by stratum: land cover in 2015, a change of land-cover class
   between 2015 and 2018, clay and carbon terciles from 2009 (values that carry none of the
   errors of the later two campaigns).

5. Control properties. The same identity applied to pH (CaCl2) and total nitrogen, a
   property that changes slowly and one measured on the same samples as carbon. Particle
   size, which should not change at all, was measured only in 2009 and cannot be used.

Intervals resample the spatial blocks of the main analysis.

    python scripts/point_change.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402

N_BOOT = int(__import__("os").environ.get("N_BOOT", 2000))
LC = {"B": "cropland", "E": "grassland"}


def partition(d1, d2):
    """Error variance and the noise share of mean squared change over each interval."""
    var_e = -np.cov(d1, d2)[0, 1]
    m1, m2 = np.mean(d1 ** 2), np.mean(d2 ** 2)
    return dict(r=np.corrcoef(d1, d2)[0, 1], var_e=var_e,
                noise_share_6yr=2 * var_e / m1, noise_share_3yr=2 * var_e / m2,
                true_change_sd_6yr=np.sqrt(max(m1 - 2 * var_e, 0)),
                true_change_sd_3yr=np.sqrt(max(m2 - 2 * var_e, 0)))


def slopes(x09, x15, x18):
    """Slope of the 2015-2018 change on the 2015 level: naive, mean-based and instrumented."""
    d2 = x18 - x15
    naive = np.cov(d2, x15)[0, 1] / np.var(x15, ddof=1)
    mid = (x15 + x18) / 2
    lark = np.cov(d2, mid)[0, 1] / np.var(mid, ddof=1)
    iv = np.cov(d2, x09)[0, 1] / np.cov(x15, x09)[0, 1]
    return dict(naive=naive, mean_of_two=lark, instrumented=iv)


def ceiling(var_e, change):
    return 1 - np.sqrt(max(var_e, 0) / np.mean(change ** 2))


def main() -> int:
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    y, y0, year, pid = data.y, data.y_prev, data.times, data.point_id.astype(str)
    _, _, groups = train.spatial_folds(data, n_folds=4)
    change = y - y0

    a = pd.DataFrame(dict(pid=pid, year=year, y=y, y0=y0, g=groups))
    w = a.pivot(index="pid", columns="year").dropna(subset=[("y", 2015.0), ("y", 2018.0)])
    x09, x15, x18 = (w[("y0", 2015.0)].to_numpy(), w[("y", 2015.0)].to_numpy(),
                     w[("y", 2018.0)].to_numpy())
    assert np.allclose(w[("y0", 2018.0)], x15)
    g_pt = w[("g", 2015.0)].to_numpy()
    ids_pt = w.index.to_numpy()

    # attributes of each point from the repeat panel
    p = pd.read_parquet(config.DATA_PROCESSED / "lucas_repeat_panel.parquet")
    p["point_id"] = p["point_id"].astype(str)
    q = p.pivot(index="point_id", columns="year").reindex(ids_pt)
    lc15 = q[("land_cover", 2015)].str[0].to_numpy()
    lc18 = q[("land_cover", 2018)].str[0].to_numpy()
    clay = q[("clay", 2009)].to_numpy()
    ctrl = {"pH (CaCl2)": [q[("ph_cacl2", yr)].to_numpy() for yr in (2009, 2015, 2018)],
            "log total nitrogen": [np.log(q[("n_total", yr)].to_numpy().clip(0.05))
                                   for yr in (2009, 2015, 2018)]}
    print(f"{len(w):,} points with three campaigns; land cover 2015 "
          f"{pd.Series(lc15).value_counts().to_dict()}")

    # strata, defined per point and carried to the observation rows
    tert = lambda v: np.digitize(v, np.nanpercentile(v, [100 / 3, 200 / 3]))
    strata = {"all points": np.ones(len(w), bool)}
    for k, name in LC.items():
        strata[f"land cover: {name}"] = lc15 == k
    strata["land cover: other"] = ~np.isin(lc15, list(LC))
    strata["land-cover class unchanged 2015-2018"] = lc15 == lc18
    strata["land-cover class changed 2015-2018"] = lc15 != lc18
    for i, lab in enumerate(("low", "middle", "high")):
        strata[f"clay tercile: {lab}"] = tert(clay) == i
        strata[f"carbon tercile 2009: {lab}"] = tert(x09) == i
    row_of = pd.Series(np.arange(len(w)), index=ids_pt)
    obs_pt = row_of.reindex(pid).to_numpy()            # point row of each observation
    has_pt = ~np.isnan(obs_pt)
    obs_pt = np.where(has_pt, obs_pt, 0).astype(int)

    def stats(ip, io):
        """Every reported quantity for a set of point rows and observation rows."""
        out = {}
        d1, d2 = x15[ip] - x09[ip], x18[ip] - x15[ip]
        out.update({("signal and noise", "all points", k): v for k, v in partition(d1, d2).items()})
        out.update({("level dependence", "all points", k): v
                    for k, v in slopes(x09[ip], x15[ip], x18[ip]).items()})
        for k, name in LC.items():
            m = lc15[ip] == k
            out.update({("level dependence", f"land cover: {name}", kk): v
                        for kk, v in slopes(x09[ip][m], x15[ip][m], x18[ip][m]).items()})
        io = io[has_pt[io]]
        for s, mask in strata.items():
            mp, mo = mask[ip], mask[obs_pt[io]]
            pt = partition(d1[mp], d2[mp])
            out[("ceiling by stratum", s, "points")] = mp.sum()
            out[("ceiling by stratum", s, "r")] = pt["r"]
            out[("ceiling by stratum", s, "var_e")] = pt["var_e"]
            out[("ceiling by stratum", s, "noise_share_6yr")] = pt["noise_share_6yr"]
            out[("ceiling by stratum", s, "ceiling")] = ceiling(pt["var_e"], change[io][mo])
        for name, (c09, c15, c18) in ctrl.items():
            ok = ~(np.isnan(c09[ip]) | np.isnan(c15[ip]) | np.isnan(c18[ip]))
            e1, e2 = (c15[ip] - c09[ip])[ok], (c18[ip] - c15[ip])[ok]
            pt = partition(e1, e2)
            out.update({("control property", name, k): v for k, v in pt.items()})
            out[("control property", name, "ceiling")] = ceiling(pt["var_e"], np.r_[e1, e2])
        return out

    all_pt, all_obs = np.arange(len(w)), np.arange(len(y))
    point = stats(all_pt, all_obs)

    rng = np.random.default_rng(0)
    blocks = np.unique(groups)
    pts = {b: np.where(g_pt == b)[0] for b in blocks}
    obs = {b: np.where(groups == b)[0] for b in blocks}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        draws.append(stats(np.concatenate([pts[b] for b in pick]),
                           np.concatenate([obs[b] for b in pick])))

    rows = []
    for key, est in point.items():
        v = np.array([d[key] for d in draws], float)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        rows.append(dict(analysis=key[0], subset=key[1], quantity=key[2], estimate=est,
                         ci95_low=lo, ci95_high=hi,
                         share_below_zero=np.mean(v < 0) if key[0] == "level dependence" else np.nan))
    df = pd.DataFrame(rows)
    df.loc[df.quantity == "points", ["ci95_low", "ci95_high"]] = np.nan
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "point_change.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
