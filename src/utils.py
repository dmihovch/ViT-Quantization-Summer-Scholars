"""General-purpose utilities shared across all experiment scripts."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set all relevant random seeds to ensure reproducibility.

    Covers Python's ``random`` module, NumPy, PyTorch (CPU and CUDA), and
    cuDNN determinism flags.  After calling this function, any non-determinism
    should be limited to operations that have no deterministic kernel
    implementation in the installed PyTorch version.

    Parameters
    ----------
    seed:
        Integer seed value.  The same seed value on the same hardware and
        software stack should reproduce identical results.
    """
    raise NotImplementedError


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns
    -------
    torch.device
        ``torch.device("cuda")`` if a CUDA-capable GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    raise NotImplementedError


def ensure_dir(path: Path) -> None:
    """Create a directory and all missing parent directories.

    Equivalent to ``mkdir -p``.  Has no effect if the directory already
    exists.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    raise NotImplementedError
