"""Training and cross-validated evaluation under the protocol fixed in advance.

Two rules are enforced here rather than left to the caller:

* Standardisation is fitted on the training fold alone. Fitting it on everything would
  leak the test distribution into training and inflate the result.
* Every fold is a blocked fold. Random splitting is available only through
  ``blocking.random_kfold`` and is reported as the labelled optimism reference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import numpy as np
import torch

import blocking, evaluate, losses, models


@dataclass
class Dataset:
    """One row per observation: a point measured at a given time."""
    x_dyn: np.ndarray            # (n, C, T) dynamic covariate series
    x_static: np.ndarray         # (n, P) static soil and terrain covariates
    y: np.ndarray                # (n,) SOC concentration at the observation
    y_prev: np.ndarray           # (n,) previous observed SOC at the same point
    coords: np.ndarray           # (n, 2) projected coordinates, for spatial blocking
    times: np.ndarray            # (n,) survey year, for forward temporal splits
    point_id: np.ndarray | None = None
    delta_max: np.ndarray | None = None   # (n,) bound on annual change, for the penalty
    x_dyn_next: np.ndarray | None = None  # (n, C, T) one step ahead, for the penalty

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx):
        def take(a):
            return None if a is None else a[idx]
        return Dataset(self.x_dyn[idx], self.x_static[idx], self.y[idx], self.y_prev[idx],
                       self.coords[idx], self.times[idx], take(self.point_id),
                       take(self.delta_max), take(self.x_dyn_next))


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 128
    patience: int = 20           # early stopping on the blocked validation fold
    lam: float = 0.0             # weight on the mechanistic penalty
    dropout: float = 0.1
    hidden: int = 32
    rnn_hidden: int = 32
    static_dim: int = 16
    cell: str = "lstm"
    use_recurrent: bool = True
    use_static: bool = True
    use_encoder: bool = True
    seed: int = 0
    model_kwargs: dict = field(default_factory=dict)


class Standardiser:
    """Fitted on the training fold only."""

    def fit(self, x_dyn, x_static, y):
        self.m_dyn = x_dyn.mean(axis=(0, 2), keepdims=True)
        self.s_dyn = x_dyn.std(axis=(0, 2), keepdims=True) + 1e-8
        self.m_st = x_static.mean(0, keepdims=True)
        self.s_st = x_static.std(0, keepdims=True) + 1e-8
        self.m_y, self.s_y = float(y.mean()), float(y.std() + 1e-8)
        return self

    def dyn(self, a):
        return (a - self.m_dyn) / self.s_dyn

    def static(self, a):
        return (a - self.m_st) / self.s_st

    def y_fwd(self, a):
        return (a - self.m_y) / self.s_y

    def y_inv(self, a):
        return a * self.s_y + self.m_y

    def sigma_inv(self, a):
        return a * self.s_y


def _to(t, device):
    return torch.as_tensor(np.ascontiguousarray(t), dtype=torch.float32, device=device)


def build_model(cfg: TrainConfig, n_dynamic: int, n_static: int) -> models.HybridSOC:
    torch.manual_seed(cfg.seed)
    return models.HybridSOC(
        n_dynamic=n_dynamic, n_static=n_static, hidden=cfg.hidden,
        static_dim=cfg.static_dim, rnn_hidden=cfg.rnn_hidden, cell=cfg.cell,
        dropout=cfg.dropout, use_recurrent=cfg.use_recurrent,
        use_static=cfg.use_static, use_encoder=cfg.use_encoder, **cfg.model_kwargs)


def train_model(data: Dataset, train_idx, val_idx, cfg: TrainConfig, device=None):
    """Fit one model, stopping when the blocked validation loss stops improving."""
    device = device or torch.device("cpu")
    tr, va = data.subset(train_idx), data.subset(val_idx)
    sc = Standardiser().fit(tr.x_dyn, tr.x_static, tr.y)

    xt, st, yt = (_to(sc.dyn(tr.x_dyn), device), _to(sc.static(tr.x_static), device),
                  _to(sc.y_fwd(tr.y), device))
    xv, sv, yv = (_to(sc.dyn(va.x_dyn), device), _to(sc.static(va.x_static), device),
                  _to(sc.y_fwd(va.y), device))
    xt_next = _to(sc.dyn(tr.x_dyn_next), device) if (cfg.lam > 0 and tr.x_dyn_next is not None) else None
    dmax = _to(tr.delta_max / sc.s_y, device) if (cfg.lam > 0 and tr.delta_max is not None) else None

    model = build_model(cfg, data.x_dyn.shape[1], data.x_static.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best, best_state, bad = np.inf, None, 0
    n = len(yt)
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.epochs):
        model.train()
        for b in np.array_split(rng.permutation(n), max(1, n // cfg.batch_size)):
            if len(b) < 2:
                continue
            opt.zero_grad()
            mu, log_sigma = model(xt[b], st[b] if cfg.use_static else None)
            mu_next = None
            if xt_next is not None:
                mu_next, _ = model(xt_next[b], st[b] if cfg.use_static else None)
            loss = losses.total_loss(mu, log_sigma, yt[b], mu_next,
                                     None if dmax is None else dmax[b], cfg.lam)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mu, log_sigma = model(xv, sv if cfg.use_static else None)
            vloss = losses.gaussian_nll(mu, log_sigma, yv).item()
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, sc, best


@torch.no_grad()
def predict(model, sc: Standardiser, data: Dataset, idx, device=None, use_static=True):
    device = device or torch.device("cpu")
    d = data.subset(idx)
    model.eval()
    mu, log_sigma = model(_to(sc.dyn(d.x_dyn), device),
                          _to(sc.static(d.x_static), device) if use_static else None)
    mu = sc.y_inv(mu.cpu().numpy())
    sigma = sc.sigma_inv(np.exp(np.clip(log_sigma.cpu().numpy(), -7, 7)))
    return mu, sigma


def cross_validate(data: Dataset, cfg: TrainConfig, folds: Iterable, device=None,
                   n_ensemble: int = 1, val_fraction: float = 0.2):
    """Run one configuration over pre-built folds and return out-of-fold predictions."""
    device = device or torch.device("cpu")
    n = len(data)
    mu_oof = np.full(n, np.nan)
    sigma_oof = np.full(n, np.nan)
    for tr_idx, te_idx in folds:
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(tr_idx)
        cut = max(1, int(len(perm) * val_fraction))
        va_idx, fit_idx = perm[:cut], perm[cut:]
        mus, sigmas = [], []
        for k in range(n_ensemble):
            c = TrainConfig(**{**cfg.__dict__, "seed": cfg.seed + k})
            model, sc, _ = train_model(data, fit_idx, va_idx, c, device)
            m, s = predict(model, sc, data, te_idx, device, cfg.use_static)
            mus.append(m)
            sigmas.append(s)
        mu = np.mean(mus, axis=0)
        var = np.mean(np.array(sigmas) ** 2 + np.array(mus) ** 2, axis=0) - mu ** 2
        mu_oof[te_idx] = mu
        sigma_oof[te_idx] = np.sqrt(np.clip(var, 1e-12, None))
    done = ~np.isnan(mu_oof)
    res = evaluate.evaluate_level_and_change(data.y_prev[done], data.y[done], mu_oof[done])
    res["coverage_90"] = evaluate.coverage(data.y[done], mu_oof[done], sigma_oof[done], 0.90)
    return {"metrics": res, "mu": mu_oof, "sigma": sigma_oof, "evaluated": done}


def with_previous(data: Dataset) -> Dataset:
    """Append the previous observation to the static vector.

    The last measured value at a location is known when a forecast is made, and it
    dominates the signal, so a model denied it is being asked to beat persistence with
    one hand tied. Withholding it is ablation 6.
    """
    x_static = np.hstack([data.x_static, data.y_prev[:, None]]).astype(np.float32)
    d = Dataset(**{**data.__dict__, "x_static": x_static})
    return d


def spatial_folds(data: Dataset, block_size: float | None = None, n_folds: int = 5,
                  seed: int = 0):
    """Blocked folds sized from the SOC variogram range unless overridden."""
    if block_size is None:
        block_size = blocking.variogram_range(data.coords, data.y)
    bid = blocking.spatial_blocks(data.coords, block_size)
    return list(blocking.spatial_block_folds(bid, n_folds=n_folds, seed=seed)), block_size
