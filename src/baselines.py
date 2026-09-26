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
