"""Result figures for the manuscript, drawn from the result tables and saved predictions.

    Fig. 3  skill of every model against the noise ceiling, and paired differences with
            95 % block-bootstrap intervals
    Fig. 4  what each input adds: static inputs, then Landsat, then ERA5-Land climate,
            for the hybrid and for gradient boosting, with 95 % block-bootstrap intervals
            and the mean-reversion floor
    Fig. 5  the 8,100 mineral points and the four spatially blocked folds

Colours: categorical slots 1 and 2 of the reference palette (blue for the hybrid, orange
for gradient boosting), validated for colour-vision deficiency; text stays in ink tones.

    python scripts/make_figures.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

HYB, GBM = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CEILING = 0.48
OUT = config.TABLES.parent / "figures"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_skill():
    t = config.TABLES
    abl = pd.read_csv(t / "ablations_mineral.csv").set_index("model")["skill_vs_persistence"]
    boot = pd.read_csv(t / "bootstrap_comparison.csv")
    sp = boot[boot.scheme == "spatial"].set_index("quantity")
    lin = pd.read_csv(t / "breakdown_simple_baselines.csv").set_index("quantity").loc[
        "linear on previous value and campaign"]

    rows = [  # label, skill, family, bootstrap key
        ("Full model, LSTM", abl["full model (LSTM)"], "h", "hybrid, full"),
        ("No recurrent decoder", abl["1. no recurrent decoder"], "h", "hybrid, no recurrent decoder"),
        ("No learned encoder", abl["2. no learned encoder"], "h", None),
        ("GRU in place of LSTM", abl["3. GRU in place of LSTM"], "h", None),
        ("No static embedding", abl["4. no static embedding"], "h", None),
        ("No previous observation", abl["5. no previous observation"], "h", None),
        ("Series replaced by zeros", sp.loc["hybrid, series replaced by zeros", "estimate"], "h",
         "hybrid, series replaced by zeros"),
        ("Gradient boosting, with previous value",
         sp.loc["gradient boosting, with previous value", "estimate"], "g",
         "gradient boosting, with previous value"),
        ("Gradient boosting, no previous value",
         sp.loc["gradient boosting, no previous value", "estimate"], "g",
         "gradient boosting, no previous value"),
        ("Mean-reversion floor", lin.loc["estimate"], "l", "linear"),
    ]
    rows.sort(key=lambda r: r[1])

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.2, 6.6), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for i, (lab, s, fam, key) in enumerate(rows):
        c = {"h": HYB, "g": GBM, "l": INK2}[fam]
        if key == "linear":
            lo, hi = lin.loc["ci_low"], lin.loc["ci_high"]
            a.plot([lo, hi], [i, i], color=c, linewidth=1.4, solid_capstyle="round", zorder=2)
        elif key is not None:
            lo, hi = sp.loc[key, ["ci_low", "ci_high"]]
            a.plot([lo, hi], [i, i], color=c, linewidth=1.4, solid_capstyle="round", zorder=2)
        a.scatter([s], [i], s=30, color=c, edgecolor="white", linewidth=0.8, zorder=3)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    a.set_xlim(0.1, 0.3)
    a.set_xticks([0.10, 0.15, 0.20, 0.25, 0.30])
    a.set_xlabel("Skill against persistence")
    _grid(a)
    a.scatter([], [], color=HYB, s=30, label="Hybrid architecture")
    a.scatter([], [], color=GBM, s=30, label="Gradient boosting")
    a.scatter([], [], color=INK2, s=30, label="Mean reversion")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=7,
             handletextpad=0.2, columnspacing=1.0, borderaxespad=0)
    a.set_title("a", loc="left", fontweight="bold", color=INK, pad=16)

    rob = pd.read_csv(t / "robustness.csv").set_index("quantity")
    clim = pd.read_csv(t / "climate_comparison.csv").set_index("quantity")
    items = [  # label, (estimate, low, high)
        ("Hybrid − boosting", rob.loc["hybrid minus boosting", ["estimate", "ci95_low", "ci95_high"]]),
        ("Hybrid − mean-reversion floor", rob.loc["hybrid minus linear", ["estimate", "ci95_low", "ci95_high"]]),
        ("No decoder − hybrid", rob.loc["no decoder minus hybrid", ["estimate", "ci95_low", "ci95_high"]]),
        ("Hybrid − zero series", sp.loc["hybrid, full minus hybrid, series replaced by zeros",
                                        ["estimate", "ci_low", "ci_high"]]),
        ("Hybrid − boosting, both with climate",
         clim.loc["hybrid, with climate minus gradient boosting with previous value, with climate",
                  ["estimate", "ci_low", "ci_high"]]),
        ("Later date, new places: hybrid − boosting",
         rob.loc["spatiotemporal hybrid minus boosting", ["estimate", "ci95_low", "ci95_high"]]),
        ("Later date, new places: hybrid − floor",
         rob.loc["spatiotemporal hybrid minus linear", ["estimate", "ci95_low", "ci95_high"]]),
    ]
    ys = list(range(len(items)))[::-1]
    for y, (lab, v) in zip(ys, items):
        est, lo, hi = [float(x) for x in v]
        b.plot([lo, hi], [y, y], color=INK, linewidth=1.4, solid_capstyle="round")
        b.scatter([est], [y], s=30, color=INK, edgecolor="white", linewidth=0.8, zorder=3)
    b.set_yticks(ys, [it[0] for it in items])
    b.axvline(0, color=INK2, linewidth=0.8)
    b.set_xlabel("Skill difference")
    b.axvspan(-0.02, 0.02, color=GRID, alpha=0.5, zorder=0, linewidth=0)
    b.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
    _grid(b)
    b.set_title("b", loc="left", fontweight="bold", color=INK, pad=16)
    fig.tight_layout()
    return fig


def _static_only_intervals():
    """Skill of both methods on static inputs alone, with block-bootstrap intervals.

    The hybrid with its series replaced by zeros is saved by bootstrap_comparison.py; boosting
    on the static properties and previous value is refitted on the same folds (seconds).
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import baselines
    from run_ablations import load_npz
    from bootstrap_comparison import block_bootstrap
    d = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    pr = np.load(config.RESULTS / "predictions" / "spatial_mineral.npz")
    X = np.hstack([d.x_static, d.y_prev[:, None]])
    gb = np.zeros(len(d.y))
    for f in np.unique(pr["fold"]):
        te = pr["fold"] == f
        gb[te] = baselines.fit_gradient_boosting(X[~te], d.y[~te]).predict(X[te])
    rows = block_bootstrap(pr["groups"], d.y_prev, d.y,
                           {"hybrid": pr["hybrid__series_replaced_by_zeros"], "boosting": gb}, [])
    return {r["quantity"]: (r["estimate"], r["ci_low"], r["ci_high"]) for r in rows}


def fig_inputs():
    t = config.TABLES
    clim = pd.read_csv(t / "climate_comparison.csv").set_index("quantity")
    floor = pd.read_csv(t / "breakdown_simple_baselines.csv").set_index("quantity").loc[
        "linear on previous value and campaign", "estimate"]
    static = _static_only_intervals()

    def row(q):
        r = clim.loc[q]
        return r["estimate"], r["ci_low"], r["ci_high"]
    series = {
        "Hybrid": (HYB, "o", [static["hybrid"], row("hybrid, Landsat only"),
                              row("hybrid, with climate")]),
        "Gradient boosting": (GBM, "s", [static["boosting"],
                                         row("gradient boosting with previous value, Landsat only"),
                                         row("gradient boosting with previous value, with climate")]),
    }
    steps = ["Static inputs\nonly", "+ Landsat\nseries", "+ ERA5-Land\nclimate"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(steps), dtype=float)
    ax.axhline(floor, color=INK2, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    ax.text(x[-1] + 0.32, floor - 0.004, "Mean-reversion\nfloor", ha="right", va="top",
            fontsize=6.5, color=INK2)
    for k, (name, (c, m, pts)) in enumerate(series.items()):
        off = -0.07 if k == 0 else 0.07
        est = np.array([p[0] for p in pts])
        lo = est - np.array([p[1] for p in pts])
        hi = np.array([p[2] for p in pts]) - est
        ax.plot(x + off, est, color=c, linewidth=1.2, zorder=2)
        ax.errorbar(x + off, est, yerr=[lo, hi], fmt=m, color=c, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, elinewidth=0.9, capsize=0,
                    zorder=3, label=name)
        for xi, v in zip(x + off, est):
            ax.text(xi + (-0.09 if k == 0 else 0.09), v, f"{v:.3f}", fontsize=6.5, color=c,
                    ha="right" if k == 0 else "left", va="center")
    ax.set_xticks(x, steps)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0.14, 0.31)
    ax.set_ylabel("Skill against persistence")
    _grid(ax, "y")
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.6)
    fig.tight_layout()
    return fig


def fig_map():
    z = np.load(config.DATA_PROCESSED / "panel_mineral.npz", allow_pickle=True)
    fold = np.load(config.TABLES.parent / "predictions" / "spatial_mineral.npz")["fold"]
    xy = z["coords"] / 1000.0
    first = z["times"] == z["times"].min()          # one dot per point, not per observation
    xy, fold = xy[first], fold[first]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), sharex=True, sharey=True)
    for f, ax in enumerate(axs):
        held = fold == f
        ax.scatter(xy[~held, 0], xy[~held, 1], s=0.6, color="#f2a582", linewidths=0)
        ax.scatter(xy[held, 0], xy[held, 1], s=0.9, color=HYB, linewidths=0)
        ax.set_title(f"Fold {f + 1} held out ({held.sum():,} points)", fontsize=7, color=INK)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("fig_skill", fig_skill), ("fig_inputs", fig_inputs), ("fig_map", fig_map)):
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT / name}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
