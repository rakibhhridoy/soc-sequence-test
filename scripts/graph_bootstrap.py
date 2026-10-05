"""Paired block-bootstrap intervals for the graph extension, mineral soils.

run_graph.py saves held-out predictions per neighbourhood. They share the single
assignment of blocks to folds behind results/predictions/spatial_mineral.npz, so every
model can be scored on the same resampled blocks, as in bootstrap_comparison.py.

    python scripts/graph_bootstrap.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, evaluate  # noqa: E402
from run_ablations import load_npz  # noqa: E402

N_BOOT = 2000


def main() -> int:
    d = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    pred_dir = config.RESULTS / "predictions"
    main_run = np.load(pred_dir / "spatial_mineral.npz")
    pred = {
        "hybrid": main_run["hybrid__full"],
        "gradient boosting": main_run["gradient_boosting__with_previous_value"],
    }
    for name, label in [("none", "graph, no edges (control)"),
                        ("geographic", "graph, geographic neighbours")]:
        g = np.load(pred_dir / f"graph_mineral_{name}.npz")
        if not (g["fold"] == main_run["fold"]).all():
            raise SystemExit(f"{name}: folds differ from the main run, so pairing is invalid")
        pred[label] = g["mu"]
    block = main_run["groups"]
    y, y0 = d.y, d.y_prev

    rng = np.random.default_rng(config.SEED)
    blocks = np.unique(block)
    members = {b: np.where(block == b)[0] for b in blocks}
    draws = [np.concatenate([members[b] for b in rng.choice(blocks, len(blocks))])
             for _ in range(N_BOOT)]

    def skill(p, s=slice(None)):
        return evaluate.skill_score(y0[s], y[s], p[s])

    rows = [dict(comparison=f"skill: {k}", difference=skill(v)) for k, v in pred.items()]
    for a, b in [("graph, geographic neighbours", "graph, no edges (control)"),
                 ("graph, geographic neighbours", "hybrid"),
                 ("graph, no edges (control)", "hybrid"),
                 ("graph, geographic neighbours", "gradient boosting")]:
        diff = np.array([skill(pred[a], s) - skill(pred[b], s) for s in draws])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(comparison=f"{a} minus {b}", difference=skill(pred[a]) - skill(pred[b]),
                         lo95=lo, hi95=hi, share_positive=(diff > 0).mean()))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.TABLES / "graph_mineral_bootstrap.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
