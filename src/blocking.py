"""Validation design: spatial blocks and forward temporal splits.

This module exists before any model, because the evaluation design makes blocked
validation the condition under which a skill claim means anything. ``random_kfold`` is
provided only as the labelled optimism reference.
"""
from __future__ import annotations
from typing import Iterator, Sequence
import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- variogram
def empirical_variogram(coords: np.ndarray, values: np.ndarray, n_lags: int = 20,
                        max_dist: float | None = None):
    """Semivariance against separation distance.

    coords: (n, 2) projected coordinates in metres. Returns (lag_centres, gamma, counts).
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu = np.triu_indices(len(coords), k=1)
    d = d[iu]
    sq = 0.5 * (values[iu[0]] - values[iu[1]]) ** 2
    if max_dist is None:
        max_dist = np.percentile(d, 50)
    edges = np.linspace(0, max_dist, n_lags + 1)
    idx = np.digitize(d, edges) - 1
    keep = (idx >= 0) & (idx < n_lags)
    lags, gamma, counts = [], [], []
    for b in range(n_lags):
        m = keep & (idx == b)
        if m.sum() >= 30:
            lags.append(0.5 * (edges[b] + edges[b + 1]))
            gamma.append(sq[m].mean())
            counts.append(int(m.sum()))
    return np.array(lags), np.array(gamma), np.array(counts)


def _spherical(h, nugget, sill, rng):
    h = np.asarray(h, float)
    out = np.where(h < rng, nugget + sill * (1.5 * h / rng - 0.5 * (h / rng) ** 3),
                   nugget + sill)
    return out


def variogram_range(coords: np.ndarray, values: np.ndarray, **kw) -> float:
    """Fitted spherical range: the distance beyond which points stop being correlated.

    This sets the spatial block size, so that a held-out site has no training neighbour
    inside its own autocorrelation distance.
    """
    lags, gamma, _ = empirical_variogram(coords, values, **kw)
    if len(lags) < 4:
        raise ValueError("too few populated lag bins to fit a variogram")
    p0 = [gamma.min(), gamma.max() - gamma.min(), lags[len(lags) // 2]]
    bounds = ([0, 0, lags[1]], [gamma.max(), 10 * gamma.max(), 5 * lags[-1]])
    popt, _ = curve_fit(_spherical, lags, gamma, p0=p0, bounds=bounds, maxfev=20000)
    return float(popt[2])


# ------------------------------------------------------------------------ spatial blocks
def spatial_blocks(coords: np.ndarray, block_size: float) -> np.ndarray:
    """Assign each point to a square block of side ``block_size`` (same units as coords)."""
    coords = np.asarray(coords, float)
    ij = np.floor((coords - coords.min(axis=0)) / float(block_size)).astype(int)
    _, block_id = np.unique(ij, axis=0, return_inverse=True)
    return block_id


def spatial_block_folds(block_id: np.ndarray, n_folds: int = 5, seed: int | None = None
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-blocks-out folds. Whole blocks move together, so no test point has a
    training neighbour inside the block."""
    rng = np.random.default_rng(seed)
    blocks = np.unique(block_id)
    rng.shuffle(blocks)
    for part in np.array_split(blocks, n_folds):
        test = np.isin(block_id, part)
        yield np.where(~test)[0], np.where(test)[0]


# ----------------------------------------------------------------------- temporal splits
def forward_temporal_splits(times: Sequence, n_splits: int = 3, buffer: float = 0.0
                            ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Train only on periods before the test window, with a buffer between them.

    ``buffer`` is in the same units as ``times`` and should be at least the
    autocorrelation length of the covariate series, so a lagged copy of a test-period
    observation cannot reach the training set.
    """
    t = np.asarray(times, float)
    cuts = np.quantile(np.unique(t), np.linspace(0, 1, n_splits + 2)[1:-1])
    for cut in cuts:
        train = np.where(t <= cut)[0]
        test = np.where(t > cut + buffer)[0]
        if len(train) and len(test):
            yield train, test


def random_kfold(n: int, k: int = 5, seed: int | None = None
                 ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Random k-fold. OPTIMISM REFERENCE ONLY.

    Reported solely as the gap against blocked validation. A skill number from this
    function is never a headline result.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for part in np.array_split(idx, k):
        test = np.zeros(n, bool)
        test[part] = True
        yield np.where(~test)[0], np.where(test)[0]
