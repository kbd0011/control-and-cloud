"""Deterministic seeding across numpy, random, and torch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Trade speed for determinism in tests and small runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch always present in deps
        pass
