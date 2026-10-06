"""Baselines. Each answers a different objection to the architecture."""
from __future__ import annotations
import numpy as np


def persistence(y_prev):
    """Predict no change. A model that fails to beat this has learned nothing
    about change, whatever its R^2 against absolute concentration."""
    return np.asarray(y_prev, float).copy()


def summary_features(x_dyn: np.ndarray) -> np.ndarray:
    """Hand-engineered temporal summaries: the tabular alternative to a learned encoder.

    x_dyn (n, C, T) -> (n, C*5) with mean, sd, min, max and linear slope per channel.
    """
    x = np.asarray(x_dyn, float)
    n, c, t = x.shape
    tt = np.arange(t, dtype=float)
    tt_c = tt - tt.mean()
    slope = (x * tt_c).sum(-1) / (tt_c ** 2).sum()
    feats = [x.mean(-1), x.std(-1), x.min(-1), x.max(-1), slope]
    return np.concatenate([f.reshape(n, c) for f in feats], axis=1)


def temporal_features(x_dyn: np.ndarray) -> np.ndarray:
    """Summaries that keep the timing the five statistics above discard.

    The window runs January to December over the five years before a survey, so index
    t mod 12 is the calendar month. Per channel: the five statistics of summary_features,
    the mean of each calendar month (12), each year's mean (5), the amplitude of the mean
    seasonal cycle and its peak month as sine and cosine (3), the last twelve months minus
    the first forty-eight (1) and the mean absolute month-to-month change (1).
    x_dyn (n, C, T) with T a multiple of 12 -> (n, C*27).
    """
    x = np.asarray(x_dyn, float)
    n, c, t = x.shape
    if t % 12:
        raise ValueError("temporal_features needs whole years")
    years = x.reshape(n, c, t // 12, 12)
    clim = years.mean(2)                                   # (n, c, 12)
    peak = clim.argmax(-1) * (2 * np.pi / 12)
    feats = [summary_features(x).reshape(n, 5, c).transpose(0, 2, 1),
             clim, years.mean(-1),
             (clim.max(-1) - clim.min(-1))[..., None], np.sin(peak)[..., None],
             np.cos(peak)[..., None],
             (x[..., -12:].mean(-1) - x[..., :-12].mean(-1))[..., None],
             np.abs(np.diff(x, axis=-1)).mean(-1)[..., None]]
    return np.concatenate(feats, axis=-1).reshape(n, -1)


BOOSTING_GRID = [dict(max_depth=d, min_child_weight=w, colsample_bytree=cs)
                 for d in (3, 5, 8) for w in (1, 20) for cs in (0.5, 0.9)]


def fit_tuned_boosting(X, y, tr_idx, groups, seed: int = 0):
    """Boosting with its own nested selection, mirroring tuning.select for the hybrid.

    The training fold is split by whole blocks; every configuration of BOOSTING_GRID is
    fitted with early stopping on one side and scored on the other, and the winner is
    refitted on the training fold with early stopping on a block-held-out fifth, as the
    hybrid is trained. Returns the fitted model and the selection record.
    """
    from xgboost import XGBRegressor
    import evaluate, train
    X, y = np.asarray(X, float), np.asarray(y, float)

    def fit(params, fit_idx, stop_idx):
        m = XGBRegressor(n_estimators=3000, learning_rate=0.03, subsample=0.8,
                         early_stopping_rounds=100, tree_method="hist", random_state=seed,
                         n_jobs=8, **params)
        m.fit(X[fit_idx], y[fit_idx], eval_set=[(X[stop_idx], y[stop_idx])], verbose=False)
        return m

    inner_fit, inner_val = train.split_fit_val(tr_idx, groups, 0.25, seed=1000 + seed)
    fit_idx, stop_idx = train.split_fit_val(inner_fit, groups, 0.2, seed=2000 + seed)
    y_prev = X[:, -1]                    # the previous value is the last column
    rows = []
    for g in BOOSTING_GRID:
        mu = fit(g, fit_idx, stop_idx).predict(X[inner_val])
        rows.append({**g, "skill": float(evaluate.skill_score(y_prev[inner_val], y[inner_val], mu))})
    best = max(rows, key=lambda r: r["skill"])
    fit_idx, stop_idx = train.split_fit_val(tr_idx, groups, 0.2, seed=seed)
    params = {k: best[k] for k in BOOSTING_GRID[0]}
    model = fit(params, fit_idx, stop_idx)
    return model, dict(best=best, n_trees=int(model.best_iteration) + 1)


def fit_gradient_boosting(X, y, **kw):
    """Gradient boosting on summary statistics: the working standard of the field, and
    the comparison that decides whether the architecture earns its complexity."""
    from xgboost import XGBRegressor
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=0)
    params.update(kw)
    model = XGBRegressor(**params)
    model.fit(np.asarray(X, float), np.asarray(y, float))
    return model
