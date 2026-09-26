"""Result figures for the manuscript, drawn from the result tables and saved predictions.

    Fig. 3  skill of every model against the noise ceiling, and paired differences with
            95 % block-bootstrap intervals
    Fig. 4  what each input adds: static inputs, then Landsat, then ERA5-Land climate,
            for the hybrid and for gradient boosting
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

    rows = [  # label, skill, family, bootstrap key
        ("Full model, LSTM", abl["full model (LSTM)"], "h", "hybrid, full"),
        ("No recurrent decoder", abl["1. no recurrent decoder"], "h", "hybrid, no recurrent decoder"),
        ("No learned encoder", abl["2. no learned encoder"], "h", None),
        ("GRU in place of LSTM", abl["3. GRU in place of LSTM"], "h", None),
        ("No static embedding", abl["4. no static embedding"], "h", None),
        ("λ = 0", abl["5. lambda = 0 (no penalty)"], "h", None),
        ("No previous observation", abl["6. no previous observation"], "h", None),
        ("Series replaced by zeros", sp.loc["hybrid, series replaced by zeros", "estimate"], "h",
         "hybrid, series replaced by zeros"),
        ("Gradient boosting, with previous value",
         sp.loc["gradient boosting, with previous value", "estimate"], "g",
         "gradient boosting, with previous value"),
        ("Gradient boosting, no previous value",
         sp.loc["gradient boosting, no previous value", "estimate"], "g",
         "gradient boosting, no previous value"),
    ]
    rows.sort(key=lambda r: r[1])

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1]))
    for i, (lab, s, fam, key) in enumerate(rows):
        c = HYB if fam == "h" else GBM
        if key is not None:
            lo, hi = sp.loc[key, ["ci_low", "ci_high"]]
            a.plot([lo, hi], [i, i], color=c, linewidth=1.4, solid_capstyle="round", zorder=2)
        a.scatter([s], [i], s=30, color=c, edgecolor="white", linewidth=0.8, zorder=3)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    a.axvline(CEILING, color=INK2, linewidth=0.8)
    a.text(CEILING - 0.006, -0.6, "noise ceiling", ha="right", va="center",
           color=INK2, fontsize=7)
    a.set_xlim(0.1, 0.5)
    a.set_ylim(-1.1, len(rows) - 0.5)
    a.set_xlabel("Skill against persistence")
    _grid(a)
    a.scatter([], [], color=HYB, s=30, label="Hybrid architecture")
    a.scatter([], [], color=GBM, s=30, label="Gradient boosting")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=7,
             handletextpad=0.2, columnspacing=1.0, borderaxespad=0)
    a.set_title("a", loc="left", fontweight="bold", color=INK, pad=16)

    diffs = [
        ("Full − boosting (fair)", "spatial",
         "hybrid, full minus gradient boosting, with previous value"),
        ("Full − boosting, no previous value", "spatial",
         "hybrid, full minus gradient boosting, no previous value"),
        ("No decoder − full", "spatial", "hybrid, no recurrent decoder minus hybrid, full"),
        ("Full − zero series", "spatial", "hybrid, full minus hybrid, series replaced by zeros"),
        ("Full − boosting, forward in time", "temporal",
         "hybrid, full minus gradient boosting, with previous value"),
    ]
    clim = pd.read_csv(t / "climate_comparison.csv").set_index("quantity")
    ys = list(range(len(diffs) + 1))[::-1]
    for y, (lab, scheme, key) in zip(ys, diffs):
        r = boot[(boot.scheme == scheme) & (boot.quantity == key)].iloc[0]
        b.plot([r.ci_low, r.ci_high], [y, y], color=INK, linewidth=1.4, solid_capstyle="round")
        b.scatter([r.estimate], [y], s=30, color=INK, edgecolor="white", linewidth=0.8, zorder=3)
    r = clim.loc["hybrid, with climate minus gradient boosting with previous value, with climate"]
    b.plot([r.ci_low, r.ci_high], [ys[-1]] * 2, color=INK, linewidth=1.4, solid_capstyle="round")
    b.scatter([r.estimate], [ys[-1]], s=30, color=INK, edgecolor="white", linewidth=0.8, zorder=3)
    b.set_yticks(ys, [d[0] for d in diffs] + ["Full − boosting, both with climate"])
    b.axvline(0, color=INK2, linewidth=0.8)
    b.set_xlabel("Skill difference")
    b.set_xticks([-0.02, 0, 0.02, 0.04])
    b.set_xlim(-0.022, 0.045)
    _grid(b)
    b.set_title("b", loc="left", fontweight="bold", color=INK, pad=16)
    fig.tight_layout()
    return fig


def fig_inputs():
    t = config.TABLES
    dyn = pd.read_csv(t / "dynamic_diagnostic.csv").set_index("model")["skill"]
    boot = pd.read_csv(t / "bootstrap_comparison.csv")
    sp = boot[boot.scheme == "spatial"].set_index("quantity")["estimate"]
    clim = pd.read_csv(t / "climate_comparison.csv").set_index("quantity")["estimate"]
    steps = ["Static inputs\nonly", "+ Landsat\nseries", "+ ERA5-Land\nclimate"]
    hyb = [dyn["hybrid, series replaced by zeros"], sp["hybrid, full"], clim["hybrid, with climate"]]
    gbm = [dyn["gradient boosting, static + previous only"],
           sp["gradient boosting, with previous value"],
           clim["gradient boosting with previous value, with climate"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(steps))
    w = 0.36
    for off, vals, c, lab in ((-w / 2 - 0.01, hyb, HYB, "Hybrid architecture"),
                              (w / 2 + 0.01, gbm, GBM, "Gradient boosting")):
        ax.bar(x + off, vals, width=w, color=c, label=lab, edgecolor="white", linewidth=1)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, steps)
    ax.set_ylim(0, 0.37)
    ax.set_ylabel("Skill against persistence")
    _grid(ax, "y")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
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
        ax.scatter(xy[~held, 0], xy[~held, 1], s=0.6, color="#c9c8c2", linewidths=0)
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
