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


# ---------------------------------------------------------------------------
# F1 — Shuffle auto-select tests (always shuffle by default)
# ---------------------------------------------------------------------------


def test_build_val_loader_auto_shuffle_default_is_true(tmp_path: Path) -> None:
    """build_val_loader with shuffle=None must default to shuffle=True.

    Creates a minimal ImageFolder-compatible directory and verifies that the
    returned DataLoader uses a RandomSampler (not SequentialSampler), which
    confirms shuffle=True was auto-selected.
    """
    from torch.utils.data import SequentialSampler

    # Create a minimal ImageFolder tree.
    class_dir = tmp_path / "class_0"
    class_dir.mkdir()
    from PIL import Image
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    img.save(class_dir / "img_0.png")
    img.save(class_dir / "img_1.png")

    loader = build_val_loader(
        data_dir=tmp_path,
        transform=_DUMMY_TRANSFORM,
        batch_size=2,
        num_images=None,
        device=_DEVICE,
        shuffle=None,
    )
    # When shuffle=True, DataLoader uses RandomSampler (not SequentialSampler).
    assert not isinstance(loader.sampler, SequentialSampler), (
        "Expected RandomSampler (shuffle=True), got SequentialSampler (shuffle=False)"
    )


def test_build_val_loader_auto_shuffle_full_dataset_is_shuffled(tmp_path: Path) -> None:
    """build_val_loader with num_images=None must auto-select shuffle=True.

    This replaces the old behaviour where full-dataset runs defaulted to
    shuffle=False.  The new behaviour ensures class-diverse batches for
    representative per-batch statistics.
    """
    from torch.utils.data import SequentialSampler

    class_dir = tmp_path / "class_0"
    class_dir.mkdir()
    from PIL import Image
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    img.save(class_dir / "img_0.png")
    img.save(class_dir / "img_1.png")

    loader = build_val_loader(
        data_dir=tmp_path,
        transform=_DUMMY_TRANSFORM,
        batch_size=2,
        num_images=None,
        device=_DEVICE,
    )
    assert not isinstance(loader.sampler, SequentialSampler), (
        "Full dataset (num_images=None) must auto-select shuffle=True"
    )


def test_build_val_loader_explicit_false_overrides_auto(tmp_path: Path) -> None:
    """build_val_loader with shuffle=False must use sequential sampling.

    Explicit overrides must still work — the auto-select logic only applies
    when shuffle=None.
    """
    from torch.utils.data import SequentialSampler

    class_dir = tmp_path / "class_0"
    class_dir.mkdir()
    from PIL import Image
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    img.save(class_dir / "img_0.png")
    img.save(class_dir / "img_1.png")

    loader = build_val_loader(
        data_dir=tmp_path,
        transform=_DUMMY_TRANSFORM,
        batch_size=2,
        num_images=None,
        device=_DEVICE,
        shuffle=False,
    )
    assert isinstance(loader.sampler, SequentialSampler), (
        "Explicit shuffle=False must produce SequentialSampler"
    )
