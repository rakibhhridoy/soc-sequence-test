"""Graph extension, an addition to the main comparison.

Adds message passing between neighbouring points on top of the hybrid architecture, under
the identical blocked folds used for the main comparison. Two neighbourhoods are
compared: geographic, which spatial blocking disrupts by design, and covariate-space,
which it does not.

    python scripts/run_graph.py --data data/processed/panel_mineral.npz
"""
from __future__ import annotations
import argparse, sys, pathlib, time
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, evaluate, graphs, graph_train, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--graphs", default="geographic,covariate-space",
                    help="comma list of neighbourhoods; 'none' is the edge-free control, the same "
                         "model and full-batch training with every edge removed")
    ap.add_argument("--block-size", type=float, default=None,
                    help="metres; default fits the SOC variogram, which is unstable on small panels")
    args = ap.parse_args()

    data = load_npz(args.data)
    data = train.with_previous(data)
    if data.point_id is None:
        raise SystemExit("panel lacks point_id, so same-point edges cannot be removed")
    config.seed_everything()
    device = config.device()
    folds, block, groups = train.spatial_folds(data, n_folds=args.folds,
                                               block_size=args.block_size)
    stem = pathlib.Path(args.data).stem.replace("panel_", "")
    # the hybrid's per-fold selection, so the graph layers are the only difference
    fc = tuning.fold_configs(data, folds, groups, device, tag=f"{stem} spatial")
    print(f"{len(data.y):,} observations | block {block/1000:.0f} km | k={args.k} | {device.type}")

    feats = np.hstack([baselines.summary_features(data.x_dyn), data.x_static])

    rows = []
    for name in args.graphs.split(","):
        mu = np.full(len(data.y), np.nan)
        sig = np.full(len(data.y), np.nan)
        t0 = time.time()
        for f, (tr_idx, te_idx) in enumerate(folds):
            fit, val = train.split_fit_val(tr_idx, groups, 0.2, seed=0)
            if name == "none":
                ei = torch.zeros((2, 0), dtype=torch.long)
            elif name == "geographic":
                ei = graphs.geographic_graph(data.coords, args.k)
            else:
                ei = graphs.covariate_graph(feats, fit, args.k)
            ei = graphs.causal_edges(ei, data.point_id, data.times)
            src, dst = ei.numpy()
            assert not (data.point_id[src] == data.point_id[dst]).any()
            assert (data.times[src] <= data.times[dst]).all()
            ms, ss = [], []
            for e in range(args.ensemble):
                cfg = train.TrainConfig(epochs=args.epochs, patience=30, seed=e,
                                        **{k: fc[f][k] for k in train.TUNED})
                m, sd = graph_train.fit_predict(data, ei, fit, val, te_idx, cfg, device)
                ms.append(m); ss.append(sd)
            m = np.mean(ms, axis=0)
            var = np.mean(np.array(ss) ** 2 + np.array(ms) ** 2, axis=0) - m ** 2
            mu[te_idx], sig[te_idx] = m, np.sqrt(np.clip(var, 1e-12, None))
        ok = ~np.isnan(mu)
        rows.append(dict(
            model=("graph model, no edges (control)" if name == "none"
                   else f"graph on hybrid ({name} k-NN)"),
            rmse=evaluate.rmse(data.y[ok], mu[ok]),
            ccc=evaluate.ccc(data.y[ok], mu[ok]),
            skill_vs_persistence=evaluate.skill_score(data.y_prev[ok], data.y[ok], mu[ok]),
            coverage_90=evaluate.coverage(data.y[ok], mu[ok], sig[ok], 0.90),
            n=int(ok.sum())))
        print(f"  {name:16s} skill {rows[-1]['skill_vs_persistence']:+.3f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    out = config.TABLES / f"graph_{stem}.csv"
    if out.exists():                    # keep rows from neighbourhoods not rerun here
        old = pd.read_csv(out)
        df = pd.concat([old[~old.model.isin(df.model)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
