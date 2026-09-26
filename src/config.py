"""Paths, seeding and device selection."""
from __future__ import annotations
from pathlib import Path
import os, random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = RESULTS / "models"

SEED = 20260923

# Dynamic covariate window fixed at five years of monthly steps (decision 2026-09-23).
N_TIMESTEPS = 60
TIMESTEP_MONTHS = 1


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def device():
    """Apple GPU when available, otherwise CPU."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
