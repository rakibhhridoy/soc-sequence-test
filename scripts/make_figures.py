"""Result figures for the manuscript, drawn from the result tables and saved predictions.

    fig_skill       skill of every model against the noise ceiling, and paired differences
                    with 95 % block-bootstrap intervals (double column)
    fig_inputs      what each input adds: static inputs, then Landsat, then ERA5-Land climate,
                    for the hybrid and for gradient boosting (single column)
    fig_change_map  observed and predicted change per 100 km cell, and point by point
                    (double column); also called by soil_breakdown.py
    fig_map         the mineral points and the four spatially blocked folds (double column)

Artwork follows the Nature Portfolio specification: 89 mm single and 183 mm double column,
Arial at 5 to 7 pt, bold lowercase panel letters at 8 pt, thin strokes, text in ink tones and
never in a series colour. Colours are categorical slots 1 and 2 of the reference palette (blue
for the hybrid, orange for gradient boosting), validated for colour-vision deficiency, with a
blue-grey-red diverging scale for change. Country outlines are Natural Earth 1:50m (public
domain), in data/external/naturalearth, projected to the panel's ETRS89-LAEA (EPSG:3035).

    python scripts/make_figures.py
"""
from __future__ import annotations
import sys, json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

HYB, GBM = "#2a78d6", "#eb6834"
TRAIN = "#f2a582"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LAND, BORDER = "#f3f2ef", "#c9c8c3"
OUT = config.TABLES.parent / "figures"
NE = config.DATA_PROCESSED.parent / "external" / "naturalearth" / "ne_50m_admin_0_countries.geojson"
MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM
DIVERGING = LinearSegmentedColormap.from_list("loss_gain", ["#c93a39", "#f0efec", "#1c5cab"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5",
                                                        "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "lines.solid_capstyle": "round",
})


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def _panel(ax, letter, text=None, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK,
            ha="left", va="bottom")
    if text:
        from matplotlib.transforms import offset_copy
        ax.text(x, y, text, transform=offset_copy(ax.transAxes, fig=ax.figure, x=8, units="points"),
                fontsize=7, color=INK, ha="left", va="bottom")


def _fmt(v, signed=False):
    return (f"{v:+.3f}" if signed else f"{v:.3f}").replace("-", "−")


# ----------------------------------------------------------------------------- basemap
def _countries(extent):
    """Country polygons from Natural Earth, projected to EPSG:3035 (km) and clipped."""
    from pyproj import Transformer
    from shapely.geometry import shape, box
    from shapely.ops import transform
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    x0, x1, y0, y1 = extent
    clip = box(x0 * 1000, y0 * 1000, x1 * 1000, y1 * 1000)
    feats = json.loads(NE.read_text())["features"]
    polys = []
    for f in feats:
        lon, lat = f["properties"].get("LABEL_X", 0), f["properties"].get("LABEL_Y", 0)
        if not (-35 < lon < 60 and 25 < lat < 75):        # Europe and its neighbours only
            continue
        g = transform(tr.transform, shape(f["geometry"])).intersection(clip)
        if g.is_empty:
            continue
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon":
                polys.append(np.asarray(p.exterior.coords) / 1000.0)
    return polys


_BASE = {}


def _basemap(ax, extent):
    key = tuple(extent)
    if key not in _BASE:
        _BASE[key] = _countries(extent)
    ax.add_collection(PolyCollection(_BASE[key], facecolor=LAND, edgecolor=BORDER,
                                     linewidth=0.25, zorder=0))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _extent(xy, pad=120):
    return [xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad,
            xy[:, 1].max() + pad]


def _scalebar(ax, km=500, fx=0.03, fy=0.82):
    """A scale bar starting at the axes fraction (fx, fy), placed over open sea."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
    ax.plot([x, x + km], [y, y], color=INK2, linewidth=0.8, solid_capstyle="butt", zorder=5)
    ax.text(x + km / 2, y + 0.015 * (y1 - y0), f"{km} km", ha="center", va="bottom",
            fontsize=5.5, color=INK2, zorder=5)


# ----------------------------------------------------------------------------- Fig. skill
def fig_skill():
    t = config.TABLES
    abl = pd.read_csv(t / "ablations_mineral.csv").set_index("model")["skill_vs_persistence"]
    boot = pd.read_csv(t / "bootstrap_comparison.csv")
    sp = boot[boot.scheme == "spatial"].set_index("quantity")
    lin = pd.read_csv(t / "breakdown_simple_baselines.csv").set_index("quantity").loc[
        "linear on previous value and campaign"]
    nc = pd.read_csv(t / "noise_ceiling.csv").set_index(["design", "quantity"]).loc[
        ("all targets", "ceiling")]

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
            ("Mean-reversion floor", lin["estimate"], lin["ci_low"], lin["ci_high"]),
            ("Noise ceiling", nc["estimate"], nc["ci95_low"], nc["ci95_high"])]),
    ]

    rob = pd.read_csv(t / "robustness.csv").set_index("quantity")
    clim = pd.read_csv(t / "climate_replicates.csv").set_index("quantity")
    bf = pd.read_csv(t / "boosting_features.csv").set_index(["design", "quantity"])

    def r(q):
        return tuple(rob.loc[q, ["estimate", "ci95_low", "ci95_high"]])

    def f(design, q):
        return tuple(bf.loc[(design, q), ["estimate", "ci95_low", "ci95_high"]])
    diffs = [
        ("Across space", [
            ("Hybrid − boosting", *r("hybrid minus boosting")),
            ("Hybrid − boosting, timing features",
             *f("spatial blocking", "hybrid minus timing, fixed")),
            ("Hybrid − tuned boosting, with climate", *tuple(clim.loc[
                "hybrid climate minus boosting tuned climate",
                ["estimate", "ci95_low", "ci95_high"]])),
            ("Hybrid − mean-reversion floor", *r("hybrid minus linear")),
            ("No decoder − full hybrid", *r("no decoder minus hybrid")),
            ("Hybrid − hybrid with zero series",
             *ci("hybrid, full minus hybrid, series replaced by zeros"))]),
        ("Later survey, new places", [
            ("Hybrid − boosting", *r("spatiotemporal hybrid minus boosting")),
            ("Hybrid − tuned boosting",
             *f("new places, later survey", "hybrid minus summary, tuned")),
            ("Hybrid − mean-reversion floor", *r("spatiotemporal hybrid minus linear"))]),
    ]

    def layout(blocks):
        """y positions top-down, with a header row and a small gap before each block."""
        ys, heads, y = [], [], 0.0
        for name, items in blocks:
            heads.append((y, name)); y -= 1.0
            for _ in items:
                ys.append(y); y -= 1.0
            y -= 0.35
        return ys, heads

    fig = plt.figure(figsize=(DOUBLE, 3.55))
    a = fig.add_axes([0.205, 0.12, 0.27, 0.76])
    b = fig.add_axes([0.715, 0.12, 0.27, 0.76])
    ms, lw = 16, 1.0

    def header(ax, ys_heads):
        for y, name in ys_heads:
            ax.text(-0.03, y, name, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=6.5, fontweight="bold", color=INK)

    # a: skill of every model
    flat = [(c, it) for g, c, items in groups for it in items]
    ys, heads = layout([(g, items) for g, c, items in groups])
    XMIN = 0.0
    a.axvline(lin["estimate"], color=INK3, linestyle=(0, (3, 2)), linewidth=0.5, zorder=1)
    a.axvline(nc["estimate"], color=INK3, linestyle=(0, (1, 1.5)), linewidth=0.6, zorder=1)
    for y, (c, (lab, est, lo, hi)) in zip(ys, flat):
        if est < XMIN:
            # no static embedding falls far below persistence: mark the edge of the scale
            a.scatter([XMIN + 0.003], [y], s=14, marker="<", color=c, zorder=3, clip_on=False,
                      linewidths=0)
            a.text(XMIN + 0.011, y, f"{_fmt(est)}, off scale", va="center", fontsize=5.5,
                   color=INK2, zorder=4,
                   bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
            continue
        if lo is not None:
            a.plot([lo, hi], [y, y], color=c, linewidth=lw, zorder=2)
            a.scatter([est], [y], s=ms, color=c, edgecolor="white", linewidth=0.5, zorder=3)
            xt = hi + 0.006
        else:
            a.scatter([est], [y], s=ms - 2, facecolor="white", edgecolor=c, linewidth=0.9,
                      zorder=3)
            xt = est + 0.009
        a.text(xt, y, _fmt(est), va="center", fontsize=5.5, color=INK2)
    a.set_yticks(ys, [it[0] for _, it in flat])
    header(a, heads)
    a.set_xlim(XMIN, 0.40)
    a.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4], ["0", "0.1", "0.2", "0.3", "0.4"])
    a.set_ylim(min(ys) - 0.7, 0.6)
    a.set_xlabel("Skill against persistence")
    a.tick_params(axis="y", length=0, pad=3)
    _grid(a)
    a.legend(handles=[
        Line2D([], [], marker="o", linestyle="-", color=INK2, markerfacecolor=INK2,
               markeredgecolor="white", markersize=4, linewidth=lw, label="95 % interval"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=INK2, markersize=3.6, markeredgewidth=0.9, label="Single run"),
        Line2D([], [], color=INK3, linestyle=(0, (3, 2)), linewidth=0.5, label="Floor"),
        Line2D([], [], color=INK3, linestyle=(0, (1, 1.5)), linewidth=0.6, label="Ceiling")],
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=5.5,
        handlelength=1.4, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
    fig.text(0.005, 0.985, "a", fontsize=8, fontweight="bold", color=INK, va="top")

    # b: paired differences
    flat = [it for _, items in diffs for it in items]
    ys, heads = layout(diffs)
    b.axvspan(-0.02, 0.02, color=GRID, alpha=0.6, zorder=0, linewidth=0)
    b.axvline(0, color=INK2, linewidth=0.5, zorder=1)
    for y, (lab, est, lo, hi) in zip(ys, flat):
        c = GBM if "boosting" in lab else (INK2 if "floor" in lab else HYB)
        clear = lo > 0 or hi < 0
        b.plot([lo, hi], [y, y], color=c, linewidth=lw, zorder=2)
        b.scatter([est], [y], s=ms, facecolor=c if clear else "white",
                  edgecolor="white" if clear else c, linewidth=0.5 if clear else 0.9, zorder=3)
        b.text(hi + 0.004, y, _fmt(est, signed=True), va="center", fontsize=5.5, color=INK2)
    b.set_yticks(ys, [it[0] for it in flat])
    header(b, heads)
    b.text(0.0, max(ys) + 1.0, "±0.02 margin", ha="center", va="center", fontsize=5,
           color=INK2, zorder=4,
           bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    b.set_xlim(-0.045, 0.235)
    b.set_xticks([-0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.20],
                 ["−0.04", "0", "0.04", "0.08", "0.12", "0.16", "0.20"])
    b.set_ylim(min(ys) - 0.7, max(ys) + 1.6)
    b.set_xlabel("Difference in skill")
    b.tick_params(axis="y", length=0, pad=3)
    _grid(b)
    b.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=INK2,
               markeredgecolor="white", markersize=4, label="Interval excludes zero"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=INK2, markersize=3.6, markeredgewidth=0.9,
               label="Interval includes zero")],
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False, fontsize=5.5,
        handletextpad=0.3, borderaxespad=0.1)
    fig.text(0.5, 0.985, "b", fontsize=8, fontweight="bold", color=INK, va="top")
    return fig


# ----------------------------------------------------------------------------- Fig. inputs
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
    # Landsat and climate steps pooled over the five replicates of climate_replicates.py;
    # the static-only step is the single assignment, the only design it was run on
    clim = pd.read_csv(t / "climate_replicates.csv").set_index("quantity")
    floor = pd.read_csv(t / "robustness.csv").set_index("quantity").loc["linear", "estimate"]
    static = _static_only_intervals()

    def row(q):
        r = clim.loc[q]
        return r["estimate"], r["ci95_low"], r["ci95_high"]
    series = {
        "Hybrid": (HYB, "o", [static["hybrid"], row("hybrid"), row("hybrid climate")]),
        "Gradient boosting": (GBM, "s", [static["boosting"], row("boosting"),
                                         row("boosting climate")]),
    }
    steps = ["Static inputs\nonly", "+ Landsat\nseries", "+ ERA5-Land\nclimate"]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.45))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.97)
    x = np.arange(len(steps), dtype=float)
    ax.axhline(floor, color=INK3, linestyle=(0, (3, 2)), linewidth=0.5, zorder=1)
    ax.text(x[-1] + 0.45, floor - 0.004, "Mean-reversion floor", ha="right", va="top",
            fontsize=5.5, color=INK2)
    for k, (name, (c, m, pts)) in enumerate(series.items()):
        off = -0.09 if k == 0 else 0.09
        est = np.array([p[0] for p in pts])
        lo = est - np.array([p[1] for p in pts])
        hi = np.array([p[2] for p in pts]) - est
        ax.plot(x + off, est, color=c, linewidth=0.8, zorder=2)
        ax.errorbar(x + off, est, yerr=[lo, hi], fmt=m, color=c, markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.5, elinewidth=0.8, capsize=0,
                    zorder=3, label=name)
        # hybrid values above-left of the markers and boosting below-right, clear of the
        # connecting lines since the hybrid sits above boosting at every step
        for xi, v in zip(x + off, est):
            if k == 0:
                ax.text(xi - 0.05, v + 0.003, _fmt(v), fontsize=5.5, color=INK2, ha="right",
                        va="bottom")
            else:
                ax.text(xi + 0.05, v - 0.003, _fmt(v), fontsize=5.5, color=INK2, ha="left",
                        va="top")
    ax.set_xticks(x, steps)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, len(steps) - 0.45)
    ax.set_ylim(0.0, 0.16)
    ax.set_yticks([0, 0.04, 0.08, 0.12, 0.16], ["0", "0.04", "0.08", "0.12", "0.16"])
    ax.set_ylabel("Skill against persistence")
    _grid(ax, "y")
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, handletextpad=0.4,
              borderaxespad=0.2)
    return fig


# ----------------------------------------------------------------------------- Fig. change
CELL_KM = 100


def fig_change_map(d=None):
    """Observed and hybrid-predicted change per 100 km cell, and point by point."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    if d is None:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from soil_breakdown import load
        d = load()
    d = d.assign(obs=d.y - d.y_prev, pred=d.hybrid - d.y_prev)
    d["cx"] = (d.x // CELL_KM).astype(int)
    d["cy"] = (d.yk // CELL_KM).astype(int)
    cells = d.groupby(["cx", "cy"]).agg(obs=("obs", "mean"), pred=("pred", "mean"),
                                        n=("obs", "size")).reset_index()
    cells = cells[cells.n >= 10]
    lim = float(np.nanpercentile(np.abs(cells[["obs", "pred"]].to_numpy()), 98))
    lim = np.ceil(lim * 20) / 20
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    ext = _extent(d[["x", "yk"]].to_numpy(), pad=80)

    fig = plt.figure(figsize=(DOUBLE, 2.75))
    axs = [fig.add_axes([0.0, 0.17, 0.33, 0.78]), fig.add_axes([0.325, 0.17, 0.33, 0.78])]
    c_ax = fig.add_axes([0.72, 0.2, 0.2, 0.70])
    for ax, col, letter, text in ((axs[0], "obs", "a", "Observed"),
                                  (axs[1], "pred", "b", "Predicted, hybrid")):
        _basemap(ax, ext)
        gap = 8
        rects = [Rectangle((cx * CELL_KM + gap / 2, cy * CELL_KM + gap / 2),
                           CELL_KM - gap, CELL_KM - gap) for cx, cy in zip(cells.cx, cells.cy)]
        pc = PatchCollection(rects, cmap=DIVERGING, norm=norm, linewidths=0, zorder=2)
        pc.set_array(cells[col].to_numpy())
        ax.add_collection(pc)
        _panel(ax, letter, text, x=0.02, y=0.96)
    _scalebar(axs[0])
    cax = fig.add_axes([0.13, 0.11, 0.4, 0.028])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    ticks = [v for v in (-0.2, -0.1, 0.0, 0.1, 0.2) if abs(v) <= lim]
    cb.set_ticks(ticks, labels=["0" if v == 0 else f"{v:+.1f}".replace("-", "\u2212")
                                for v in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.4, labelsize=5.5)
    cb.set_label("Mean change in log SOC per 100 km cell", fontsize=6, color=INK, labelpad=2)
    cb.ax.text(-0.01, 0.5, "Loss", transform=cb.ax.transAxes, ha="right", va="center",
               fontsize=5.5, color=INK2)
    cb.ax.text(1.01, 0.5, "Gain", transform=cb.ax.transAxes, ha="left", va="center",
               fontsize=5.5, color=INK2)

    ax = c_ax
    hb = ax.hexbin(d.obs, d.pred, gridsize=40, extent=(-1.5, 1.5, -1.5, 1.5), mincnt=1,
                   cmap=SEQUENTIAL, norm=LogNorm(), linewidths=0)
    ax.plot([-1.5, 1.5], [-1.5, 1.5], color=INK2, linewidth=0.5, linestyle=(0, (3, 2)))
    slope, icpt = np.polyfit(d.obs, d.pred, 1)
    xx = np.array([-1.5, 1.5])
    ax.plot(xx, icpt + slope * xx, color=INK, linewidth=0.8)
    ax.text(1.45, -1.42, f"Fitted slope {slope:.2f}\n1:1 dashed", ha="right", va="bottom",
            fontsize=5.5, color=INK2)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    lab = ["−1.5", "−1", "−0.5", "0", "0.5", "1", "1.5"]
    ax.set_xticks(ticks, lab); ax.set_yticks(ticks, lab)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Observed change in log SOC")
    ax.set_ylabel("Predicted change in log SOC")
    _panel(ax, "c", "Point by point", x=-0.32, y=1.03)
    hcax = fig.add_axes([0.93, 0.2, 0.009, 0.70])
    hcb = fig.colorbar(hb, cax=hcax)
    hcb.outline.set_visible(False)
    hcb.ax.tick_params(length=2, width=0.4, labelsize=5.5)
    hcb.set_ticks([1, 10, 100], labels=["1", "10", "100"])
    hcb.ax.minorticks_off()
    hcb.set_label("Observations per hexagon", fontsize=6, color=INK, labelpad=2)
    return fig, slope, len(cells)


# ----------------------------------------------------------------------------- Fig. folds
def fig_map():
    z = np.load(config.DATA_PROCESSED / "panel_mineral.npz", allow_pickle=True)
    fold = np.load(config.TABLES.parent / "predictions" / "spatial_mineral.npz")["fold"]
    xy = z["coords"] / 1000.0
    first = z["times"] == z["times"].min()          # one dot per point, not per observation
    xy, fold = xy[first], fold[first]
    ext = _extent(xy, pad=80)
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 2.05))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.12, wspace=0.03)
    for f, ax in enumerate(axs):
        held = fold == f
        _basemap(ax, ext)
        ax.scatter(xy[~held, 0], xy[~held, 1], s=0.35, color=TRAIN, linewidths=0, zorder=2,
                   rasterized=True)
        ax.scatter(xy[held, 0], xy[held, 1], s=0.45, color=HYB, linewidths=0, zorder=3,
                   rasterized=True)
        _panel(ax, "abcd"[f], f"Fold {f + 1}, {held.sum():,} points held out", x=0.0, y=1.0)
    _scalebar(axs[0])
    fig.legend(handles=[Patch(color=TRAIN, label="Training"), Patch(color=HYB, label="Held out")],
               loc="lower center", ncol=2, frameon=False, fontsize=6, handlelength=1.0,
               handleheight=0.8, bbox_to_anchor=(0.5, 0.0))
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    makers = (("fig_skill", fig_skill), ("fig_inputs", fig_inputs),
              ("fig_change_map", lambda: fig_change_map()[0]), ("fig_map", fig_map))
    for name, make in makers:
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", facecolor="white", dpi=600)
        plt.close(fig)
        print(f"wrote {OUT / name}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
