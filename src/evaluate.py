"""Metrics. Skill is reported on the change in SOC as well as on the level.

A model that reproduces the last observed value scores well on the level while
carrying no information about change, so the level alone is never a result.
"""
from __future__ import annotations
import numpy as np


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y, yhat) -> float:
    """Mean error. Positive means the prediction runs high."""
    return float(np.mean(np.asarray(yhat, float) - np.asarray(y, float)))


def ccc(y, yhat) -> float:
    """Lin's concordance correlation coefficient.

    Preferred to R^2, which rewards correlation without penalising a systematic offset.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    vy, vp = y.var(), yhat.var()
    cov = ((y - y.mean()) * (yhat - yhat.mean())).mean()
    denom = vy + vp + (y.mean() - yhat.mean()) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def metrics(y, yhat, n: int | None = None) -> dict:
    out = {"rmse": rmse(y, yhat), "bias": bias(y, yhat), "ccc": ccc(y, yhat),
           "n": int(len(np.asarray(y)) if n is None else n)}
    return out


def skill_score(y_prev, y_true, y_pred) -> float:
    """Skill against persistence: 1 - RMSE(model) / RMSE(no-change).

    Positive means the model beats assuming no change; zero means it matches it;
    negative means it is worse than doing nothing.
    """
    ref = rmse(np.asarray(y_true, float), np.asarray(y_prev, float))
    return float("nan") if ref == 0 else float(1.0 - rmse(y_true, y_pred) / ref)


def evaluate_level_and_change(y_prev, y_true, y_pred) -> dict:
    """Metrics on the SOC level and on the change since the previous observation.

    Note that RMSE on change equals RMSE on level by construction, since the error
    ``(y_pred - y_prev) - (y_true - y_prev)`` reduces to ``y_pred - y_true``. The change
    block is still worth reporting because its correlation terms differ and because the
    comparison that matters is against persistence, which is what ``skill`` gives.
    """
    y_prev = np.asarray(y_prev, float)
    lvl = metrics(y_true, y_pred)
    chg = metrics(np.asarray(y_true, float) - y_prev, np.asarray(y_pred, float) - y_prev)
    return {"level": lvl, "change": chg,
            "persistence_change_rmse": rmse(np.asarray(y_true, float) - y_prev,
                                            np.zeros_like(y_prev)),
            "skill_vs_persistence": skill_score(y_prev, y_true, y_pred)}


def coverage(y, mu, sigma, level: float = 0.90) -> float:
    """Share of observations inside the central prediction interval.

    A calibrated 90 % interval contains close to 90 % of held-out values. An
    uncalibrated interval is reported as uncalibrated, never quietly widened.
    """
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    y, mu, sigma = (np.asarray(a, float) for a in (y, mu, sigma))
    return float(np.mean(np.abs(y - mu) <= z * sigma))
