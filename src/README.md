# Modules

Planned split. Each file is importable and holds no side effects at import time;
anything that runs belongs in `../scripts/`.

| Module | Responsibility |
|---|---|
| `config.py` | paths, random seeds, device selection (MPS / CPU) |
| `data_lucas.py` | load LUCAS survey rounds, match points across rounds, build the SOC target |
| `covariates.py` | assemble the dynamic series (reflectance, climate) and the static vector |
| `blocking.py` | spatial blocks from the SOC variogram range; forward temporal splits with a buffer |
| `models.py` | the CNN encoder, static embedding, fusion, and the swappable LSTM/GRU decoder |
| `losses.py` | Gaussian negative log-likelihood, and the one-sided process-model penalty |
| `baselines.py` | persistence, gradient boosting on summary statistics, process-model run |
| `evaluate.py` | RMSE, mean error, concordance; reported on both SOC level and change |
| `train.py` | blocked cross-validation, early stopping, ensembles, leakage-safe scaling |
| `uncertainty.py` | Monte Carlo dropout (deep ensembles live in `train.cross_validate`) |
| `aoa.py` | area-of-applicability mask |

Two rules carried over from the preprint's protocol:

- `blocking.py` exists before any model is trained. Random k-fold is available only as
  the labelled optimism reference, never as a headline result.
- `evaluate.py` reports skill on change alongside skill on level. A number quoted for
  the level alone is not a result.
