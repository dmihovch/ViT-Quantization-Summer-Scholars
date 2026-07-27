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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Seeds set to %d", seed)


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns
    -------
    torch.device
        ``torch.device("cuda")`` if a CUDA-capable GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug("Using device: %s", device)
    return device


def ensure_dir(path: Path) -> None:
    """Create a directory and all missing parent directories.

    Equivalent to ``mkdir -p``.  Has no effect if the directory already
    exists.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)


def log_system_info() -> None:
    """Log hardware and software versions for reproducibility.

    Records PyTorch version, CUDA availability, GPU name, and nnsight
    version at INFO level.  Call once at the start of any experiment script.
    """
    import sys

    logger.info("Python %s", sys.version.split()[0])
    logger.info("PyTorch %s", torch.__version__)
    if torch.cuda.is_available():
        logger.info("CUDA available: %s (%.1f GB)",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.info("CUDA not available; using CPU")
    try:
        import nnsight
        ver = getattr(nnsight, "__version__", "unknown")
        logger.info("nnsight %s", ver)
    except ImportError:
        logger.info("nnsight not installed")
