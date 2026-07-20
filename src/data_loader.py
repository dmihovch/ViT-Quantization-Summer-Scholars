"""DataLoader construction for the ImageNet-1K validation split.

Uses ``torchvision.datasets.ImageFolder`` which expects the standard
ImageNet directory layout: ``<data_dir>/<class_name>/<image>.JPEG``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

from src.exceptions import DataDirectoryError

logger = logging.getLogger(__name__)

_NUM_WORKERS: int = 4


def build_val_loader(
    data_dir: Path,
    transform: Callable,
    batch_size: int,
    num_images: int | None,
    device: torch.device,  # noqa: ARG001 — kept for API symmetry; pin_memory depends on CUDA
) -> DataLoader:
    """Build a DataLoader over the ImageNet validation split.

    Parameters
    ----------
    data_dir:
        Root directory of the ImageFolder dataset.  Must exist and contain
        at least one image sub-directory.
    transform:
        Preprocessing callable applied to each PIL image before batching.
        Typically obtained from ``model.load_vit``.
    batch_size:
        Number of samples per batch.
    num_images:
        If provided, wraps the full dataset in a ``torch.utils.data.Subset``
        containing the first ``num_images`` samples (deterministic, no
        shuffle).  Pass ``None`` to use the entire split.
    device:
        Compute device; used only to decide whether ``pin_memory`` should be
        enabled (i.e. when CUDA is the target device).

    Returns
    -------
    DataLoader
        Configured with ``shuffle=False``, ``pin_memory=True`` (for CUDA),
        and ``num_workers=4``.

    Raises
    ------
    DataDirectoryError
        If ``data_dir`` does not exist on disk, or if the constructed dataset
        contains zero samples (empty directory).
    """
    if not data_dir.exists():
        raise DataDirectoryError(
            f"Data directory does not exist: {data_dir}"
        )

    try:
        dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    except FileNotFoundError as exc:
        # ImageFolder raises FileNotFoundError when no class subdirectories exist.
        raise DataDirectoryError(
            f"Data directory exists but contains no class subdirectories: {data_dir}"
        ) from exc

    if len(dataset) == 0:
        raise DataDirectoryError(
            f"Data directory exists but contains no images: {data_dir}"
        )

    if num_images is not None:
        if num_images > len(dataset):
            logger.warning(
                "num_images=%d exceeds dataset size=%d; using full dataset.",
                num_images,
                len(dataset),
            )
        else:
            dataset = Subset(dataset, list(range(num_images)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(
        "Built DataLoader: %d images, batch_size=%d",
        len(dataset),
        batch_size,
    )
    return loader
