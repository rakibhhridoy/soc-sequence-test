"""Checks the protocol machinery on synthetic data, before any real data exists.

Run: python scripts/selftest.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, blocking, evaluate, losses, models, baselines  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")


config.seed_everything()
rng = np.random.default_rng(0)

print("\n[1] variogram and spatial blocking")
# field with a known correlation length: nearby points share values
n = 400
coords = rng.uniform(0, 10_000, size=(n, 2))          # metres
centres = rng.uniform(0, 10_000, size=(12, 2))
vals = np.array([np.exp(-np.min(np.linalg.norm(c - centres, axis=1)) / 1500) for c in coords])
vals += rng.normal(0, 0.02, n)
rng_fit = blocking.variogram_range(coords, vals, n_lags=15)
check("variogram range is positive and within the domain", 0 < rng_fit < 10_000,
      f"range={rng_fit:.0f} m")

bid = blocking.spatial_blocks(coords, block_size=2000)
check("blocks partition every point", len(bid) == n and bid.min() == 0)
folds = list(blocking.spatial_block_folds(bid, n_folds=5, seed=1))
check("five spatial folds, all points tested once",
      len(folds) == 5 and sorted(np.concatenate([te for _, te in folds])) == list(range(n)))
no_leak = all(set(bid[tr]).isdisjoint(set(bid[te])) for tr, te in folds)
check("no block appears in both train and test", no_leak)

print("\n[2] forward temporal splits")
times = rng.integers(2009, 2019, size=n).astype(float)
tsplits = list(blocking.forward_temporal_splits(times, n_splits=3, buffer=1.0))
ok_order = all(times[tr].max() < times[te].min() for tr, te in tsplits)
ok_buffer = all(times[te].min() - times[tr].max() > 1.0 - 1e-9 for tr, te in tsplits)
check("training period always precedes the test period", ok_order)
check("buffer separates train from test", ok_buffer)
check("random k-fold exists but is labelled optimism-only",
      "OPTIMISM REFERENCE ONLY" in blocking.random_kfold.__doc__)

print("\n[3] metrics")
y = rng.normal(20, 5, 200)
check("perfect prediction gives rmse 0 and ccc 1",
      evaluate.rmse(y, y) == 0 and abs(evaluate.ccc(y, y) - 1) < 1e-9)
check("bias detects a systematic offset", abs(evaluate.bias(y, y + 2.0) - 2.0) < 1e-9)
# a persistence-like predictor: good on level, useless on change
y_prev = y + rng.normal(0, 0.5, 200)
res = evaluate.evaluate_level_and_change(y_prev, y, y_prev)
check("persistence scores well on level but has zero skill on change",
      res["level"]["rmse"] < 1.0 and abs(res["change"]["rmse"] - res["persistence_change_rmse"]) < 1e-9,
      f"level rmse={res['level']['rmse']:.2f}, change rmse={res['change']['rmse']:.2f}")
mu = rng.normal(0, 1, 20000); obs = mu + rng.normal(0, 1, 20000)
cov = evaluate.coverage(obs, mu, np.ones_like(mu), 0.90)
check("a calibrated 90% interval covers ~90%", abs(cov - 0.90) < 0.02, f"coverage={cov:.3f}")

print("\n[4] architecture")
B, C, T, P = 8, 6, config.N_TIMESTEPS, 5
enc = models.DilatedCNNEncoder(C, hidden=16, kernel=3, layers=3)
x = torch.randn(B, C, T)
check("receptive field matches 1+(k-1)(2^L-1)", enc.receptive_field() == 1 + 2 * (2 ** 3 - 1),
      f"RF={enc.receptive_field()} steps")
check("encoder output length matches the formula", enc(x).shape[-1] == enc.output_length(T),
      f"T={T} -> T'={enc(x).shape[-1]}")
check("twelve months of context fits in the receptive field", enc.receptive_field() >= 12)

m_lstm = models.HybridSOC(C, P, cell="lstm")
m_gru = models.HybridSOC(C, P, cell="gru")
mu, ls = m_lstm(x, torch.randn(B, P))
check("model returns a mean and a log sigma per sample", mu.shape == (B,) and ls.shape == (B,))
ratio = m_gru.n_recurrent_parameters() / m_lstm.n_recurrent_parameters()
check("GRU holds ~25% fewer recurrent parameters than LSTM", abs(ratio - 0.75) < 0.01,
      f"ratio={ratio:.3f}")
up_lstm = m_lstm.n_parameters() - m_lstm.n_recurrent_parameters()
up_gru = m_gru.n_parameters() - m_gru.n_recurrent_parameters()
check("everything upstream of the cell is identical in size", up_lstm == up_gru,
      f"{up_lstm} params either way")
m_abl = models.HybridSOC(C, P, use_recurrent=False)
check("ablation removes the recurrent block only", m_abl.n_recurrent_parameters() == 0
      and m_abl(x, torch.randn(B, P))[0].shape == (B,))

print("\n[5] losses")
yt = torch.randn(64)
good = losses.gaussian_nll(yt, torch.full((64,), -1.0), yt)
bad = losses.gaussian_nll(torch.randn(64), torch.full((64,), -1.0), yt)
check("likelihood prefers the better fit", good.item() < bad.item(),
      f"{good.item():.2f} < {bad.item():.2f}")
inside = losses.physics_penalty(torch.zeros(10), torch.full((10,), 0.2), torch.full((10,), 0.5))
outside = losses.physics_penalty(torch.zeros(10), torch.full((10,), 2.0), torch.full((10,), 0.5))
check("penalty is silent inside the plausible envelope", inside.item() == 0.0)
check("penalty activates outside it", outside.item() > 0)

print("\n[6] baselines")
xd = rng.normal(size=(50, C, T))
feats = baselines.summary_features(xd)
check("summary features give five statistics per channel", feats.shape == (50, C * 5))
ramp = np.tile(np.arange(T, dtype=float), (3, 1)).reshape(3, 1, T)
check("slope feature recovers a known trend",
      np.allclose(baselines.summary_features(ramp)[:, -1], 1.0))
tf = baselines.temporal_features(xd)
check("timing features give 27 per channel", tf.shape == (50, C * 27))
season = np.tile(np.sin(np.arange(T) * 2 * np.pi / 12), (2, 1)).reshape(2, 1, T)
tf = baselines.temporal_features(season)
check("timing features find the seasonal peak and amplitude",
      np.isclose(np.arctan2(tf[0, 23], tf[0, 24]) / (2 * np.pi / 12), 3.0) and np.isclose(tf[0, 22], 2.0))

print("\n[7] training reduces the loss (Apple GPU if available)")
dev = config.device()
torch.manual_seed(0)
xd_t = torch.randn(256, C, T)
signal = xd_t[:, 0, -12:].mean(1) * 3.0 + xd_t[:, 1, :].mean(1)
y_t = signal + torch.randn(256) * 0.1
model = models.HybridSOC(C, P, cell="gru").to(dev)
xs_t = torch.randn(256, P)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
xd_d, xs_d, y_d = xd_t.to(dev), xs_t.to(dev), y_t.to(dev)
first = None
for step in range(60):
    opt.zero_grad()
    mu, log_sigma = model(xd_d, xs_d)
    loss = losses.gaussian_nll(mu, log_sigma, y_d)
    loss.backward(); opt.step()
    if step == 0:
        first = loss.item()
check(f"loss decreases on a learnable signal (device={dev.type})", loss.item() < first,
      f"{first:.3f} -> {loss.item():.3f}")

print("\n[8] noise ceiling")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from noise_ceiling import ceiling  # noqa: E402
m = 50_000
s_true = np.cumsum(np.c_[rng.normal(3, 0.5, m), rng.normal(0, 0.1, (m, 2))], axis=1)
x = s_true + rng.normal(0, 0.3, (m, 3))
d1, d2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 1]
c = ceiling(d1, d2, np.r_[d1, d2])
check("error variance recovered from successive changes", abs(c["var_e"] - 0.09) < 0.005,
      f"{c['var_e']:.4f} vs 0.09")
oracle = 1 - np.sqrt(np.mean(np.r_[x[:, 1] - s_true[:, 1], x[:, 2] - s_true[:, 2]] ** 2)
                     / np.mean(np.r_[d1, d2] ** 2))
check("ceiling matches the skill of a forecast that knows the true value",
      abs(c["ceiling"] - oracle) < 0.01, f"{c['ceiling']:.3f} vs {oracle:.3f}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("failed:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
