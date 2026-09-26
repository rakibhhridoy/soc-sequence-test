"""Graphs over soil sampling points, for message passing between neighbours.

Two neighbourhoods are built and compared. A geographic graph joins points that are close
on the ground, which is the conventional choice and the one spatial blocking is designed
to disrupt, since a held-out block sits far from anything in training. A covariate-space
graph joins points whose soil and climate conditions resemble each other, which survives
blocking intact and acts as a learned instance-based smoother.

Only node features travel along edges, but the previous carbon value is a node feature,
and in a repeat panel one point's previous value is another observation's target. Every
graph therefore passes through `causal_edges`, which removes edges between observations of
the same point and any edge carrying information from a later survey round.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def knn_edges(X: np.ndarray, k: int = 8) -> torch.Tensor:
    """Symmetric k-nearest-neighbour edge index, self-loops excluded."""
    X = np.asarray(X, float)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    src = np.repeat(np.arange(len(X)), k)
    dst = idx[:, 1:].reshape(-1)
    edges = np.unique(np.sort(np.c_[src, dst], axis=1), axis=0)          # undirected
    both = np.vstack([edges, edges[:, ::-1]])
    return torch.as_tensor(both.T.copy(), dtype=torch.long)


def geographic_graph(coords: np.ndarray, k: int = 8) -> torch.Tensor:
    return knn_edges(coords, k)


def covariate_graph(features: np.ndarray, train_idx: np.ndarray, k: int = 8) -> torch.Tensor:
    """Neighbours by feature similarity, standardised on the training fold alone."""
    X = np.asarray(features, float)
    m, s = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
    return knn_edges((X - m) / s, k)


def causal_edges(edge_index: torch.Tensor, point_id: np.ndarray,
                 times: np.ndarray) -> torch.Tensor:
    """Keep a directed edge j -> i only if j is another point and was surveyed no later.

    Without this, the 2018 observation of a point, whose previous value is the 2015
    measurement, sits at distance zero from the 2015 observation and passes it its own
    target. Both share a spatial block, so blocking does not prevent the leak.
    """
    src, dst = edge_index.numpy()
    pid, t = np.asarray(point_id), np.asarray(times)
    keep = (pid[src] != pid[dst]) & (t[src] <= t[dst])
    return edge_index[:, torch.as_tensor(keep)]
