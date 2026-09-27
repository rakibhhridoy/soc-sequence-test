"""Where carbon changed and where the models succeed: land use, starting level and space.

Uses the held-out predictions saved by `bootstrap_comparison.py`, so nothing is retrained.
Each observation is classed by the land cover recorded at its target campaign (LUCAS codes
beginning B are cropland, E grassland), and by the tertile of its previous carbon
concentration. Skill for the hybrid and the protocol baseline is reported within each
class with block-bootstrap intervals, alongside the observed change itself.

Outputs
    results/tables/breakdown_land_use.csv, breakdown_soc_level.csv, change_by_land_use.csv
    results/figures/fig_change_map.{pdf,png}

    python scripts/soil_breakdown.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config  # noqa: E402
from bootstrap_comparison import PRED_DIR, block_bootstrap, skill  # noqa: E402
from make_figures import INK, INK2, GRID  # noqa: E402  (also applies the rcParams)

MODELS = {"hybrid": "hybrid__full", "boosting": "gradient_boosting__with_previous_value"}
DIVERGING = LinearSegmentedColormap.from_list("loss_gain", ["#c93a39", "#f0efec", "#1c5cab"])
CELL_KM = 100


def load():
    z = np.load(config.DATA_PROCESSED / "panel_mineral.npz", allow_pickle=True)
    pr = np.load(PRED_DIR / "spatial_mineral.npz")
    d = pd.DataFrame({"point_id": z["point_id"].astype(str), "year": z["times"].astype(int),
                      "y": z["y"], "y_prev": z["y_prev"], "soc": z["soc"],
                      "soc_prev": z["soc_prev"], "x": z["coords"][:, 0] / 1000,
                      "yk": z["coords"][:, 1] / 1000, "group": pr["groups"]})
    for k, key in MODELS.items():
        d[k] = pr[key]
    lc = pd.read_parquet(config.DATA_PROCESSED / "lucas_repeat_panel.parquet",
                         columns=["point_id", "year", "land_cover"])
    lc["point_id"] = lc.point_id.astype(str)
    d = d.merge(lc, on=["point_id", "year"], how="left")
    first = d.land_cover.fillna("").str[:1].str.upper()
    d["land_use"] = np.select([first == "B", first == "E"], ["cropland", "grassland"], "other")
    return d


def by_class(d, col, order):
    rows = []
    for cls in order:
        s = d[d[col] == cls]
        preds = {k: s[k].to_numpy() for k in MODELS}
        res = block_bootstrap(s.group.to_numpy(), s.y_prev.to_numpy(), s.y.to_numpy(), preds,
                              [("hybrid", "boosting")])
        for r in res:
            rows.append({col: cls, "n": len(s), **r})
    return pd.DataFrame(rows)


def change_summary(d):
    d = d.assign(change=d.soc - d.soc_prev, rel=100 * (d.soc - d.soc_prev) / d.soc_prev)
    g = d.groupby(["land_use", "year"])
    out = g.agg(n=("change", "size"), mean_change=("change", "mean"),
                median_change=("change", "median"), median_rel_pct=("rel", "median"),
                share_losing=("change", lambda c: float((c < 0).mean())),
                mean_soc_prev=("soc_prev", "mean")).reset_index()
    return out


def simple_baselines(d):
    """How much skill needs no covariates at all, fitted on the training folds only.

    Campaign offsets test whether the models are learning a shift between surveys, and a
    linear fit on the previous value measures pure regression towards the mean.
    """
    fold = np.load(PRED_DIR / "spatial_mineral.npz")["fold"]
    dy = (d.y - d.y_prev).to_numpy()
    y_prev, y, yr = d.y_prev.to_numpy(), d.y.to_numpy(), d.year.to_numpy()
    lu = d.land_use.to_numpy()
    preds = {k: np.full(len(d), np.nan) for k in
             ("campaign shift", "campaign and land-use shift", "linear on previous value",
              "linear on previous value and campaign")}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        m1 = pd.Series(dy[tr]).groupby(yr[tr]).mean()
        m2 = pd.Series(dy[tr]).groupby([yr[tr], lu[tr]]).mean()
        preds["campaign shift"][te] = y_prev[te] + pd.Series(yr[te]).map(m1).to_numpy()
        preds["campaign and land-use shift"][te] = y_prev[te] + np.array(
            [m2.get((a, b), 0.0) for a, b in zip(yr[te], lu[te])])
        for key, cols in (("linear on previous value", [y_prev]),
                          ("linear on previous value and campaign", [y_prev, yr == 2018])):
            X = np.column_stack([np.ones(len(y))] + cols).astype(float)
            b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            preds[key][te] = X[te] @ b
    preds["hybrid"] = d.hybrid.to_numpy()
    preds["boosting"] = d.boosting.to_numpy()
    return pd.DataFrame(block_bootstrap(d.group.to_numpy(), y_prev, y, preds, []))


def fig_change(d):
    d = d.assign(obs=d.y - d.y_prev, pred=d.hybrid - d.y_prev)
    d["cx"] = (d.x // CELL_KM).astype(int)
    d["cy"] = (d.yk // CELL_KM).astype(int)
    cells = d.groupby(["cx", "cy"]).agg(obs=("obs", "mean"), pred=("pred", "mean"),
                                        n=("obs", "size")).reset_index()
    cells = cells[cells.n >= 10]
    lim = float(np.nanpercentile(np.abs(cells[["obs", "pred"]].to_numpy()), 98))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    fig.subplots_adjust(wspace=0.32)
    for ax, col, title in ((axs[0], "obs", "a  Observed change"),
                           (axs[1], "pred", "b  Predicted change, hybrid")):
        ax.scatter(d.x, d.yk, s=0.3, color="#dcdbd6", linewidths=0, zorder=1)
        im = ax.scatter((cells.cx + 0.5) * CELL_KM, (cells.cy + 0.5) * CELL_KM, c=cells[col],
                        cmap=DIVERGING, norm=norm, s=9, marker="s", linewidths=0, zorder=2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=8, color=INK, fontweight="bold")
    cb = fig.colorbar(im, ax=axs[:2], orientation="horizontal", fraction=0.05, pad=0.04,
                      shrink=0.6)
    cb.set_label("Mean change in log SOC per 100 km cell (loss < 0 < gain)", color=INK2)
    cb.outline.set_visible(False)

    ax = axs[2]
    hb = ax.hexbin(d.obs, d.pred, gridsize=45, extent=(-1.5, 1.5, -1.5, 1.5), mincnt=1,
                   cmap=LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#0d366b"]),
                   bins="log", linewidths=0)
    ax.plot([-1.5, 1.5], [-1.5, 1.5], color=INK2, linewidth=0.8)
    slope = np.polyfit(d.obs, d.pred, 1)[0]
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_xlabel("Observed change in log SOC")
    ax.set_ylabel("Predicted change")
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title(f"c  Point by point, slope {slope:.2f}", loc="left", fontsize=8, color=INK,
                 fontweight="bold")
    return fig, slope, len(cells)


def main() -> int:
    d = load()
    t = config.TABLES
    print(d.land_use.value_counts().to_string())
    lu = by_class(d, "land_use", ["cropland", "grassland"])
    lu.to_csv(t / "breakdown_land_use.csv", index=False)
    d["soc_level"] = pd.qcut(d.soc_prev, 3, labels=["low", "middle", "high"])
    edges = d.groupby("soc_level", observed=True).soc_prev.agg(["min", "max"])
    lv = by_class(d, "soc_level", ["low", "middle", "high"])
    lv.to_csv(t / "breakdown_soc_level.csv", index=False)
    sb = simple_baselines(d)
    sb.to_csv(t / "breakdown_simple_baselines.csv", index=False)
    ch = change_summary(d)
    ch.to_csv(t / "change_by_land_use.csv", index=False)
    fig, slope, n_cells = fig_change(d)
    out = t.parent / "figures" / "fig_change_map"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    fmt = lambda v: f"{v:.3f}"
    print("\n", lu.to_string(index=False, float_format=fmt))
    print("\nSOC tertile edges (g/kg):\n", edges.round(1).to_string())
    print("\n", lv.to_string(index=False, float_format=fmt))
    print("\n", ch.to_string(index=False, float_format=fmt))
    print("\n", sb.to_string(index=False, float_format=fmt))
    print(f"\nfitted slope of predicted on observed change: {slope:.3f}; map cells: {n_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
