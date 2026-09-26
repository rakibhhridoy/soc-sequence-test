"""Full-batch training for the graph variant, under the same blocked folds.

Message passing needs the whole graph in one forward pass, so training proceeds on all
nodes at once with the loss taken over training nodes alone. Standardisation is still
fitted on the training fold, and labels never cross an edge, so a test node aggregates
only its neighbours' covariates.
"""
from __future__ import annotations
import numpy as np
import torch

import graph_models, losses, train as T


def _to(a, device):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


def fit_predict(data: T.Dataset, edge_index, fit_idx, val_idx, test_idx,
                cfg: T.TrainConfig, device):
    """Train on fit_idx, early stop on val_idx, return predictions for test_idx."""
    sc = T.Standardiser().fit(data.x_dyn[fit_idx], data.x_static[fit_idx], data.y[fit_idx])
    x = _to(sc.dyn(data.x_dyn), device)
    s = _to(sc.static(data.x_static), device)
    y = _to(sc.y_fwd(data.y), device)
    ei = edge_index.to(device)

    torch.manual_seed(cfg.seed)
    model = graph_models.GraphSOC(
        n_dynamic=data.x_dyn.shape[1], n_static=data.x_static.shape[1],
        hidden=cfg.hidden, rnn_hidden=cfg.rnn_hidden, cell=cfg.cell,
        dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fit_t = torch.as_tensor(fit_idx, device=device)
    val_t = torch.as_tensor(val_idx, device=device)
    best, best_state, bad = np.inf, None, 0
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        mu, log_sigma = model(x, s, ei)
        loss = losses.gaussian_nll(mu[fit_t], log_sigma[fit_t], y[fit_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mu, log_sigma = model(x, s, ei)
            v = losses.gaussian_nll(mu[val_t], log_sigma[val_t], y[val_t]).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mu, log_sigma = model(x, s, ei)
    mu = sc.y_inv(mu.cpu().numpy())
    sigma = sc.sigma_inv(np.exp(np.clip(log_sigma.cpu().numpy(), -7, 7)))
    return mu[test_idx], sigma[test_idx]
