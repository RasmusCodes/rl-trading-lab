"""Reproducible seeding.

Every run derives all of its random streams from one integer seed. `derive_seed` turns any tuple of labels
into a stable 31-bit seed (hash-based, independent of PYTHONHASHSEED), so e.g. the environment's start-date
sampler, the network initialisation and the exploration noise of run (algo, fold, seed) never collide with
those of another run.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def make_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))
