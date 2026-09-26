"""A modest, documented hyperparameter search for the hybrid model.

The pre-registered protocol forbids tuning each ablation separately, because that would
confound the component removed with its hyperparameters. It says nothing about whether
the full model was given a fair chance against the gradient-boosting baseline, which ran
with sensible library defaults while the network ran with one arbitrary configuration.

This closes that gap. The search uses its own spatially blocked split, carved out of the
data before the evaluation folds are built, so nothing learned here leaks into the
reported skill. The winning configuration is then evaluated once, unchanged, under the
same blocked cross-validation as everything else.

    python scripts/tune_model.py
"""
from __future__ import annotations
import sys, pathlib, json, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, blocking, evaluate, train  # noqa: E402

GRID = [dict(hidden=h, rnn_hidden=r, lr=lr, dropout=dr)
        for h, r in ((32, 32), (64, 64), (64, 128))
        for lr in (1e-3, 3e-3)
        for dr in (0.1, 0.2)]


def main() -> int:
    from run_ablations import load_npz            # noqa: E402
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    device = config.device()
    config.seed_everything()

    # A blocked split reserved for the search alone.
    block_size = blocking.variogram_range(data.coords, data.y)
    bid = blocking.spatial_blocks(data.coords, block_size)
    rng = np.random.default_rng(0)
    blocks = np.unique(bid)
    rng.shuffle(blocks)
    val_blocks = set(blocks[: max(1, len(blocks) // 4)])
    val = np.where(np.isin(bid, list(val_blocks)))[0]
    fit = np.where(~np.isin(bid, list(val_blocks)))[0]
    print(f"search split: {len(fit):,} fit / {len(val):,} val | block {block_size:,.0f} m")

    data_prev = train.with_previous(data)
    rows = []
    for i, g in enumerate(GRID, 1):
        cfg = train.TrainConfig(epochs=200, patience=25, lam=0.1, **g)
        t0 = time.time()
        model, sc, _ = train.train_model(data_prev, fit, val, cfg, device)
        mu, _ = train.predict(model, sc, data_prev, val, device)
        sk = evaluate.skill_score(data.y_prev[val], data.y[val], mu)
        rows.append(dict(**g, rmse=evaluate.rmse(data.y[val], mu), skill=sk))
        print(f"  {i:2d}/{len(GRID)} {g} -> skill {sk:+.3f}  ({time.time()-t0:.0f}s)")

    best = max(rows, key=lambda r: r["skill"])
    print(f"\nbest: {best}")
    config.TABLES.mkdir(parents=True, exist_ok=True)
    (config.TABLES / "tuning.json").write_text(json.dumps(
        {"grid": rows, "best": best, "block_size_m": block_size}, indent=1))
    print(f"wrote {config.TABLES/'tuning.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
