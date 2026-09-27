"""Nested hyperparameter selection: one search inside every training fold.

A search scored on data that later serve as a test fold lets the test labels choose the
configuration. Here each outer training fold is split again by whole spatial blocks, every
configuration of the grid is fitted on one side and scored on the other, and the winner is
used for that fold alone. No test observation of the outer fold takes part in the choice.

The selection depends only on the data and the training indices, so it is cached by a hash
of both and shared by every script that uses the same folds.
"""
from __future__ import annotations
import hashlib, json, time
import numpy as np

import config, evaluate, train

GRID = [dict(hidden=h, rnn_hidden=r, lr=lr, dropout=dr)
        for h, r in ((32, 32), (64, 64), (64, 128))
        for lr in (1e-3, 3e-3)
        for dr in (0.1, 0.2)]
INNER_FRACTION = 0.25          # share of the training fold's blocks scored in the search
CACHE = config.TABLES / "nested_tuning.json"


def _key(data: train.Dataset, tr_idx) -> str:
    h = hashlib.md5()
    for a in (data.x_dyn.shape, data.x_static.shape):
        h.update(str(a).encode())
    h.update(np.ascontiguousarray(data.y).tobytes())
    h.update(np.ascontiguousarray(np.sort(tr_idx)).tobytes())
    return h.hexdigest()


def _load() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def select(data: train.Dataset, tr_idx, groups, device, base: train.TrainConfig,
           tag: str = "", seed: int = 0) -> dict:
    """Best configuration for one training fold, from an inner split of its blocks."""
    key = _key(data, tr_idx)
    cache = _load()
    if key in cache:
        return cache[key]["best"]
    t0 = time.time()
    # inner split: whole blocks when groups are given, random otherwise
    inner_fit, inner_val = train.split_fit_val(tr_idx, groups, INNER_FRACTION, seed=1000 + seed)
    fit_idx, stop_idx = train.split_fit_val(inner_fit, groups, 0.2, seed=2000 + seed)
    rows = []
    for g in GRID:
        cfg = train.TrainConfig(**{**base.__dict__, **g, "seed": seed})
        model, sc, _ = train.train_model(data, fit_idx, stop_idx, cfg, device)
        mu, _ = train.predict(model, sc, data, inner_val, device, cfg.use_static)
        rows.append({**g, "skill": float(evaluate.skill_score(
            data.y_prev[inner_val], data.y[inner_val], mu))})
    best = max(rows, key=lambda r: r["skill"])
    cache = _load()                     # reread, so concurrent entries are kept
    cache[key] = dict(tag=tag, n_train=int(len(tr_idx)), n_inner_val=int(len(inner_val)),
                      grid=rows, best=best)
    CACHE.write_text(json.dumps(cache, indent=1))
    print(f"    search {tag}: best {best} ({time.time() - t0:.0f}s)", flush=True)
    return best


def fold_configs(data: train.Dataset, folds, groups, device, tag: str = "",
                 base: train.TrainConfig | None = None) -> list[dict]:
    """Selected configuration for each fold, searched on the full model."""
    base = base or train.TrainConfig()
    return [select(data, tr, groups, device, base, tag=f"{tag} fold {f}", seed=f)
            for f, (tr, _) in enumerate(folds)]
