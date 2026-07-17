"""Tests for the DataLoader construction utilities in :mod:`src.data_loader`.

These tests do NOT load real ImageNet images or the actual ViT transform;
they only exercise the validation guards in build_val_loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.transforms as T

from src.data_loader import build_val_loader
from src.exceptions import DataDirectoryError

# A minimal transform sufficient to pass through build_val_loader without
# touching model weights.
_DUMMY_TRANSFORM = T.Compose([T.Resize((8, 8)), T.ToTensor()])
_DEVICE = torch.device("cpu")


def test_build_val_loader_raises_on_missing_dir() -> None:
    """build_val_loader must raise DataDirectoryError when data_dir does not exist."""
    nonexistent = Path("/tmp/this_path_should_never_exist_vit_quant_test")
    with pytest.raises(DataDirectoryError):
        build_val_loader(
            data_dir=nonexistent,
            transform=_DUMMY_TRANSFORM,
            batch_size=2,
            num_images=4,
            device=_DEVICE,
        )


def test_build_val_loader_raises_on_empty_dir(tmp_path: Path) -> None:
    """build_val_loader must raise DataDirectoryError when data_dir has no images.

    tmp_path exists on disk but contains no ImageFolder-compatible subdirectories,
    so the constructed dataset will have zero samples.
    """
    with pytest.raises(DataDirectoryError):
        build_val_loader(
            data_dir=tmp_path,
            transform=_DUMMY_TRANSFORM,
            batch_size=2,
            num_images=4,
            device=_DEVICE,
        )
