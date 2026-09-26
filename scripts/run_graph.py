"""Graph extension, beyond the protocol fixed in advance.

Adds message passing between neighbouring points on top of the hybrid architecture, under
the identical blocked folds used for the fixed-protocol comparison. Two neighbourhoods are
compared: geographic, which spatial blocking disrupts by design, and covariate-space,
which it does not.

    python scripts/run_graph.py --data data/processed/panel_mineral.npz
"""
from __future__ import annotations
import argparse, sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, baselines, evaluate, graphs, graph_train, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    data = load_npz(args.data)
    data = train.with_previous(data)
    if data.point_id is None:
        raise SystemExit("panel lacks point_id, so same-point edges cannot be removed")
    config.seed_everything()
    device = config.device()
    folds, block = train.spatial_folds(data, n_folds=args.folds)
    print(f"{len(data.y):,} observations | block {block/1000:.0f} km | k={args.k} | {device.type}")

    feats = np.hstack([baselines.summary_features(data.x_dyn), data.x_static])
    cfg = train.TrainConfig(epochs=args.epochs, patience=30, hidden=32, rnn_hidden=32,
                            lr=1e-3, dropout=0.2)

    rows = []
    for name in ("geographic", "covariate-space"):
        mu = np.full(len(data.y), np.nan)
        sig = np.full(len(data.y), np.nan)
        t0 = time.time()
        for tr_idx, te_idx in folds:
            rng = np.random.default_rng(cfg.seed)
            perm = rng.permutation(tr_idx)
            cut = max(1, int(0.2 * len(perm)))
            val, fit = perm[:cut], perm[cut:]
            ei = (graphs.geographic_graph(data.coords, args.k) if name == "geographic"
                  else graphs.covariate_graph(feats, fit, args.k))
            ei = graphs.causal_edges(ei, data.point_id, data.times)
            src, dst = ei.numpy()
            assert not (data.point_id[src] == data.point_id[dst]).any()
            assert (data.times[src] <= data.times[dst]).all()
            m, s = graph_train.fit_predict(data, ei, fit, val, te_idx, cfg, device)
            mu[te_idx], sig[te_idx] = m, s
        ok = ~np.isnan(mu)
        rows.append(dict(
            model=f"graph on hybrid ({name} k-NN)",
            rmse=evaluate.rmse(data.y[ok], mu[ok]),
            ccc=evaluate.ccc(data.y[ok], mu[ok]),
            skill_vs_persistence=evaluate.skill_score(data.y_prev[ok], data.y[ok], mu[ok]),
            coverage_90=evaluate.coverage(data.y[ok], mu[ok], sig[ok], 0.90),
            n=int(ok.sum())))
        print(f"  {name:16s} skill {rows[-1]['skill_vs_persistence']:+.3f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    stem = pathlib.Path(args.data).stem.replace("panel_", "")
    out = config.TABLES / f"graph_{stem}.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
