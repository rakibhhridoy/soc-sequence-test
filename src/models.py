"""The convolutional-recurrent hybrid (Fig. 1a of the article).

Dynamic covariate series pass through a dilated 1-D convolutional encoder; static soil
and terrain covariates enter through a dense embedding broadcast across every time step;
a gated recurrent decoder carries state; a head emits a mean and a predictive standard
deviation.

The recurrent cell is a constructor argument so that LSTM and GRU are compared with
identical upstream weights, which is the design decision that makes that comparison
controlled rather than a comparison between two separately tuned models.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class DilatedCNNEncoder(nn.Module):
    """Valid (unpadded) dilated convolutions, dilation 2**(l-1) at layer l.

    Receptive field after L layers is 1 + (k-1)(2**L - 1) steps, so a season of context
    costs a parameter count that grows linearly while the context grows geometrically.
    """

    def __init__(self, in_channels: int, hidden: int = 32, kernel: int = 3, layers: int = 3):
        super().__init__()
        self.kernel, self.layers = kernel, layers
        blocks, c_in = [], in_channels
        for l in range(layers):
            blocks += [nn.Conv1d(c_in, hidden, kernel, dilation=2 ** l), nn.ReLU()]
            c_in = hidden
        self.net = nn.Sequential(*blocks)
        self.out_channels = hidden

    def receptive_field(self) -> int:
        return 1 + (self.kernel - 1) * (2 ** self.layers - 1)

    def output_length(self, t: int) -> int:
        return t - (self.receptive_field() - 1)

    def forward(self, x):                       # (B, C, T) -> (B, hidden, T')
        return self.net(x)


class SummaryEncoder(nn.Module):
    """Hand-engineered alternative to the learned encoder.

    Splits the series into fixed windows (a year by default) and takes five statistics
    per channel per window: mean, sd, min, max and linear slope. The decoder then sees a
    sequence of yearly summaries instead of learned features, which is what isolates the
    value of the convolutional representation rather than of having a sequence at all.
    """

    def __init__(self, in_channels: int, hidden: int = 32, window: int = 12):
        super().__init__()
        self.window = window
        self.proj = nn.Linear(in_channels * 5, hidden)
        self.out_channels = hidden

    def output_length(self, t: int) -> int:
        return t // self.window

    def forward(self, x):                        # (B, C, T) -> (B, hidden, T')
        b, c, t = x.shape
        n_win = t // self.window
        w = x[:, :, : n_win * self.window].reshape(b, c, n_win, self.window)
        tt = torch.arange(self.window, device=x.device, dtype=x.dtype)
        tt = tt - tt.mean()
        slope = (w * tt).sum(-1) / (tt ** 2).sum()
        feats = torch.cat([w.mean(-1), w.std(-1), w.amin(-1), w.amax(-1), slope], dim=1)
        return self.proj(feats.transpose(1, 2)).transpose(1, 2)


class HybridSOC(nn.Module):
    """Full model. ``cell`` selects the recurrent unit; ``use_recurrent=False`` is the
    ablation that removes temporal state while leaving everything else in place."""

    def __init__(self, n_dynamic: int, n_static: int, hidden: int = 32,
                 static_dim: int = 16, rnn_hidden: int = 32, cell: str = "lstm",
                 kernel: int = 3, layers: int = 3, dropout: float = 0.1,
                 use_recurrent: bool = True, use_static: bool = True,
                 use_encoder: bool = True, window: int = 12):
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError("cell must be 'lstm' or 'gru'")
        self.cell, self.use_recurrent, self.use_static = cell, use_recurrent, use_static
        self.use_encoder = use_encoder
        self.encoder = (DilatedCNNEncoder(n_dynamic, hidden, kernel, layers) if use_encoder
                        else SummaryEncoder(n_dynamic, hidden, window))
        self.static_emb = nn.Sequential(nn.Linear(n_static, static_dim), nn.ReLU()) \
            if use_static else None
        fuse_in = hidden + (static_dim if use_static else 0)
        self.fusion = nn.Sequential(nn.Linear(fuse_in, rnn_hidden), nn.ReLU())
        if use_recurrent:
            rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn_cls(rnn_hidden, rnn_hidden, batch_first=True)
        else:
            self.rnn = None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_hidden, 2)     # mean and log sigma

    def embed(self, x_dyn, x_static=None):
        """The representation the head reads, exposed so a graph layer can mix it
        with the representations of neighbouring points."""
        z = self.encoder(x_dyn).transpose(1, 2)
        if self.use_static:
            if x_static is None:
                raise ValueError("model was built with use_static=True")
            s = self.static_emb(x_static).unsqueeze(1).expand(-1, z.size(1), -1)
            z = torch.cat([z, s], dim=-1)
        u = self.fusion(z)
        if self.rnn is not None:
            out, _ = self.rnn(u)
            return out[:, -1, :]
        return u.mean(dim=1)

    def forward(self, x_dyn, x_static=None):
        """x_dyn (B, C, T), x_static (B, P) -> mu (B,), log_sigma (B,)."""
        z = self.encoder(x_dyn).transpose(1, 2)              # (B, T', hidden)
        if self.use_static:
            if x_static is None:
                raise ValueError("model was built with use_static=True")
            s = self.static_emb(x_static).unsqueeze(1).expand(-1, z.size(1), -1)
            z = torch.cat([z, s], dim=-1)
        u = self.fusion(z)                                    # (B, T', rnn_hidden)
        if self.rnn is not None:
            out, _ = self.rnn(u)
            feat = out[:, -1, :]                              # final state
        else:
            feat = u.mean(dim=1)                              # ablation: pool over time
        y = self.head(self.drop(feat))
        return y[:, 0], y[:, 1]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_recurrent_parameters(self) -> int:
        return 0 if self.rnn is None else sum(p.numel() for p in self.rnn.parameters())


def deep_ensemble(n_models: int = 5, seed: int = 0, **kwargs):
    """Independently initialised models; their spread is the predictive distribution."""
    models = []
    for i in range(n_models):
        torch.manual_seed(seed + i)
        models.append(HybridSOC(**kwargs))
    return models


@torch.no_grad()
def ensemble_predict(models, x_dyn, x_static=None):
    """Mixture mean and standard deviation over an ensemble."""
    mus, vars_ = [], []
    for m in models:
        m.eval()
        mu, log_sigma = m(x_dyn, x_static)
        mus.append(mu)
        vars_.append(torch.exp(2 * log_sigma.clamp(-7, 7)))
    mu = torch.stack(mus)
    var = torch.stack(vars_)
    mean = mu.mean(0)
    total_var = (var + mu ** 2).mean(0) - mean ** 2      # law of total variance
    return mean, total_var.clamp_min(1e-12).sqrt()
