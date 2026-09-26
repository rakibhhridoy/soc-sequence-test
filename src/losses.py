"""Training objective: Gaussian likelihood plus a one-sided mechanistic penalty."""
from __future__ import annotations
import torch


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood, so the model learns its own uncertainty."""
    log_sigma = log_sigma.clamp(-7.0, 7.0)
    inv_var = torch.exp(-2.0 * log_sigma)
    return (0.5 * inv_var * (y - mu) ** 2 + log_sigma).mean()


def physics_penalty(mu_t: torch.Tensor, mu_next: torch.Tensor,
                    delta_max: torch.Tensor) -> torch.Tensor:
    """Squared hinge on year-on-year change beyond what carbon input can support.

    Inactive inside the plausible envelope, so it constrains the model only where a
    prediction would violate a mass balance. ``delta_max`` comes from a RothC or DayCent
    run under bounding management assumptions.
    """
    excess = (mu_next - mu_t).abs() - delta_max
    return torch.clamp(excess, min=0.0).pow(2).mean()


def total_loss(mu, log_sigma, y, mu_next=None, delta_max=None, lam: float = 0.0):
    loss = gaussian_nll(mu, log_sigma, y)
    if lam > 0 and mu_next is not None and delta_max is not None:
        loss = loss + lam * physics_penalty(mu, mu_next, delta_max)
    return loss
