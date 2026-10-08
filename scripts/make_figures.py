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

    def ci(key):
        return tuple(sp.loc[key, ["estimate", "ci_low", "ci_high"]])
    groups = [  # (group, colour, [(label, estimate, low, high) ...]); low None = single run
        ("Hybrid and its ablations", HYB, [
            ("Full model (LSTM)", *ci("hybrid, full")),
            ("GRU in place of LSTM", abl["3. GRU in place of LSTM"], None, None),
            ("No recurrent decoder", *ci("hybrid, no recurrent decoder")),
            ("No learned encoder", abl["2. no learned encoder"], None, None),
            ("No previous observation", abl["5. no previous observation"], None, None),
            ("Series replaced by zeros", *ci("hybrid, series replaced by zeros")),
            ("No static embedding", abl["4. no static embedding"], None, None)]),
        ("Gradient boosting", GBM, [
            ("With previous value", *ci("gradient boosting, with previous value")),
            ("No previous value", *ci("gradient boosting, no previous value"))]),
        ("Reference", INK2, [
            ("Mean-reversion floor", lin["estimate"], lin["ci_low"], lin["ci_high"])]),
    ]

    rob = pd.read_csv(t / "robustness.csv").set_index("quantity")
    clim = pd.read_csv(t / "climate_comparison.csv").set_index("quantity")
    bf = pd.read_csv(t / "boosting_features.csv").set_index(["design", "quantity"])

    def r(q):
        return tuple(rob.loc[q, ["estimate", "ci95_low", "ci95_high"]])

    def f(design, q):
        return tuple(bf.loc[(design, q), ["estimate", "ci95_low", "ci95_high"]])
    diffs = [
        ("Across space", [
            ("Hybrid \u2212 boosting (protocol)", *r("hybrid minus boosting")),
            ("Hybrid \u2212 boosting (timing features)",
             *f("spatial blocking", "hybrid minus timing, fixed")),
            ("Hybrid \u2212 boosting, both with climate", *tuple(clim.loc[
                "hybrid, with climate minus gradient boosting with previous value, with climate",
                ["estimate", "ci_low", "ci_high"]])),
            ("Hybrid \u2212 mean-reversion floor", *r("hybrid minus linear")),
            ("No decoder \u2212 full hybrid", *r("no decoder minus hybrid")),
            ("Hybrid \u2212 hybrid with zero series",
             *ci("hybrid, full minus hybrid, series replaced by zeros"))]),
        ("Later survey, new places", [
            ("Hybrid \u2212 boosting (protocol)", *r("spatiotemporal hybrid minus boosting")),
            ("Hybrid \u2212 boosting (tuned)",
             *f("new places, later survey", "hybrid minus summary, tuned")),
            ("Hybrid \u2212 mean-reversion floor", *r("spatiotemporal hybrid minus linear"))]),
    ]

    def layout(blocks):
        """y positions top-down, with a gap and a header row before each block."""
        ys, labels, heads, y = [], [], [], 0.0
        for name, items in blocks:
            heads.append((y, name)); y -= 1.0
            for it in items:
                ys.append(y); labels.append(it); y -= 1.0
            y -= 0.4
        return ys, labels, heads

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.2, 7.6),
                               gridspec_kw=dict(height_ratios=[1.25, 1]))

    # (a) skill of every model
    flat = [(g, c, it) for g, c, items in groups for it in items]
    ys, _, heads = layout([(g, items) for g, c, items in groups])
    for y, (g, c, (lab, est, lo, hi)) in zip(ys, flat):
        if lo is not None:
            a.plot([lo, hi], [y, y], color=c, linewidth=1.4, solid_capstyle="round", zorder=2)
            a.scatter([est], [y], s=34, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        else:
            a.scatter([est], [y], s=30, facecolor="white", edgecolor=c, linewidth=1.2, zorder=3)
        a.text((hi if hi is not None else est) + 0.004, y, f"{est:.3f}", va="center",
               fontsize=6.5, color=c)
    a.set_yticks(ys, [it[0] for _, _, it in flat])
    for y, name in heads:
        a.text(-0.01, y, name, transform=a.get_yaxis_transform(), ha="right", va="center",
               fontsize=7.5, fontweight="bold", color=INK)
    a.axvline(lin["estimate"], color=INK2, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    a.set_xlim(0.10, 0.31)
    a.set_xticks([0.10, 0.15, 0.20, 0.25, 0.30])
    a.set_ylim(min(ys) - 0.8, 0.6)
    a.set_xlabel("Skill against persistence")
    a.tick_params(axis="y", length=0)
    _grid(a)
    a.scatter([], [], s=30, color=INK2, label="95 % block-bootstrap interval")
    a.scatter([], [], s=30, facecolor="white", edgecolor=INK2, linewidth=1.2,
              label="Single run, no interval")
    a.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=6.5,
             handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)
    a.set_title("a", loc="left", fontweight="bold", color=INK, pad=6)

    # (b) paired differences
    flat = [(blk, it) for blk, items in diffs for it in items]
    ys, _, heads = layout(diffs)
    b.axvspan(-0.02, 0.02, color=GRID, alpha=0.55, zorder=0, linewidth=0)
    b.axvline(0, color=INK2, linewidth=0.8, zorder=1)
    for y, (blk, (lab, est, lo, hi)) in zip(ys, flat):
        c = GBM if "boosting" in lab else (INK2 if "floor" in lab else HYB)
        clear = lo > 0 or hi < 0
        b.plot([lo, hi], [y, y], color=c, linewidth=1.4, solid_capstyle="round", zorder=2)
        b.scatter([est], [y], s=34, facecolor=c if clear else "white", edgecolor=c,
                  linewidth=1.2, zorder=3)
        b.text(hi + 0.003, y, f"{est:+.3f}".replace("-", "\u2212"), va="center", fontsize=6.5,
               color=c)
    b.set_yticks(ys, [it[0] for _, it in flat])
    for y, name in heads:
        b.text(-0.01, y, name, transform=b.get_yaxis_transform(), ha="right", va="center",
               fontsize=7.5, fontweight="bold", color=INK)
    b.text(0.0, max(ys) + 1.15, "equivalence margin", ha="center", va="center", fontsize=6,
           color=INK2)
    b.set_xlim(-0.045, 0.105)
    b.set_xticks([-0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
    b.set_ylim(min(ys) - 0.8, max(ys) + 1.6)
    b.set_xlabel("Difference in skill")
    b.tick_params(axis="y", length=0)
    _grid(b)
    b.scatter([], [], s=30, color=INK2, label="Interval excludes zero")
    b.scatter([], [], s=30, facecolor="white", edgecolor=INK2, linewidth=1.2,
              label="Interval includes zero")
    b.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=6.5,
             handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)
    b.set_title("b", loc="left", fontweight="bold", color=INK, pad=6)
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
        # hybrid labels below its intervals, boosting labels above, clear of every line
        for xi, v, l, h in zip(x + off, est, est - lo, est + hi):
            if k == 0:
                ax.text(xi + 0.03, l - 0.003, f"{v:.3f}", fontsize=6.5, color=c, ha="right",
                        va="top")
            else:
                ax.text(xi - 0.03, h + 0.003, f"{v:.3f}", fontsize=6.5, color=c, ha="left",
                        va="bottom")
    ax.set_xticks(x, steps)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0.14, 0.31)
    ax.set_ylabel("Skill against persistence")
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.6)
    fig.tight_layout()
    return fig


def fig_map():
    z = np.load(config.DATA_PROCESSED / "panel_mineral.npz", allow_pickle=True)
    fold = np.load(config.TABLES.parent / "predictions" / "spatial_mineral.npz")["fold"]
    xy = z["coords"] / 1000.0
    first = z["times"] == z["times"].min()          # one dot per point, not per observation
    xy, fold = xy[first], fold[first]
    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.6), sharex=True, sharey=True)
    for f, ax in enumerate(axs.flat):
        held = fold == f
        ax.scatter(xy[~held, 0], xy[~held, 1], s=1.2, color="#f2a582", linewidths=0)
        ax.scatter(xy[held, 0], xy[held, 1], s=1.6, color=HYB, linewidths=0)
        ax.set_title(f"({'abcd'[f]}) Fold {f + 1} held out ({held.sum():,} points)",
                     fontsize=8, color=INK, loc="left")
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
