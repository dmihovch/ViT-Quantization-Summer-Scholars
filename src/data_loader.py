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
    shuffle: bool | None = None,
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
    shuffle:
        Whether to shuffle the dataset at each epoch.  When ``None``
        (default), auto-selects to ``True`` (class-diverse batches for
        representative per-batch statistics).  Pass an explicit ``bool``
        to override.

    Returns
    -------
    DataLoader
        Configured with ``pin_memory=True`` (for CUDA) and ``num_workers=4``.

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

    full_size: int = len(dataset)
    is_subset: bool = num_images is not None and num_images < full_size

    # Resolve effective shuffle before subsetting so the auto-selected
    # value controls both random sampling and DataLoader shuffling.
    # Always shuffle by default — class-diverse batches produce
    # representative per-batch σ, reducing the outlier-fraction
    # overestimate documented in open-issues.md §10.1.
    if shuffle is None:
        shuffle = True

    if num_images is not None:
        if num_images > full_size:
            logger.warning(
                "num_images=%d exceeds dataset size=%d; using full dataset.",
                num_images,
                full_size,
            )
        else:
            # When shuffling, randomly sample indices so that different
            # seeds select different images (enables cross-seed variance).
            # When not shuffling, use a seeded random permutation to avoid
            # class imbalance: ImageFolder returns images grouped by class
            # (alphabetical order), so taking the first N would only sample
            # from the first few classes.
            if shuffle:
                indices = torch.randperm(full_size)[:num_images].tolist()
            else:
                # Seeded permutation for deterministic class-balanced subsets.
                g = torch.Generator()
                g.manual_seed(42)
                indices = torch.randperm(full_size, generator=g)[:num_images].tolist()
            dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(
        "Built DataLoader: %d images, batch_size=%d, shuffle=%s",
        len(dataset),
        batch_size,
        shuffle,
    )
    return loader
