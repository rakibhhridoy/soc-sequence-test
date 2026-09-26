"""Area of applicability, after Meyer and Pebesma (2021).

A model fitted in one place may be asked to predict somewhere its training data never
went. The dissimilarity index measures how far a new point sits from the training data in
weighted predictor space; beyond a threshold set by the training data itself, the model
is outside the conditions it learned and its prediction is withheld rather than shown.
"""
from __future__ import annotations
import numpy as np


class AOA:
    def fit(self, X: np.ndarray, weights: np.ndarray | None = None):
        X = np.asarray(X, float)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-9
        w = np.ones(X.shape[1]) if weights is None else np.asarray(weights, float)
        self.w_ = w / (w.sum() + 1e-12) * len(w)
        Z = (X - self.mean_) / self.std_ * self.w_
        d = self._nearest(Z, Z, self_exclude=True)
        self.scale_ = d.mean() + 1e-12
        di_train = d / self.scale_
        q1, q3 = np.percentile(di_train, [25, 75])
        self.threshold_ = q3 + 1.5 * (q3 - q1)
        self.Z_ = Z
        return self

    @staticmethod
    def _nearest(A, B, self_exclude=False):
        """Distance from each row of A to its nearest row in B.

        A blocked full distance matrix needs tens of gigabytes at this sample size, so a
        neighbour index is used instead.
        """
        from sklearn.neighbors import NearestNeighbors
        k = 2 if self_exclude else 1
        nn = NearestNeighbors(n_neighbors=k).fit(B)
        d, _ = nn.kneighbors(A)
        return d[:, -1] if self_exclude else d[:, 0]

    def di(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean_) / self.std_ * self.w_
        return self._nearest(Z, self.Z_) / self.scale_

    def inside(self, X: np.ndarray) -> np.ndarray:
        return self.di(X) <= self.threshold_
