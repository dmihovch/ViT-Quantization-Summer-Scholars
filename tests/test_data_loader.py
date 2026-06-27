"""
test_data_loader.py
===================

Tests for image discovery, the dataset, and the DataLoader. They use the
`temp_image_dir` fixture (three tiny 8x8 PNGs plus a decoy text file), so no
real ImageNet data is needed.
"""

from pathlib import Path

import pytest
import torch
from torchvision.transforms import ToTensor

from src.data_loader import (
    UnlabeledImageDataset,
    build_image_dataloader,
    find_image_files,
)

# A simple, picklable preprocessing transform: PIL image -> [3, H, W] float
# tensor. (Picklable matters because the DataLoader uses worker processes.)
TRANSFORM = ToTensor()


def test_find_image_files_ignores_non_images(temp_image_dir: Path) -> None:
    found = find_image_files(temp_image_dir)
    assert len(found) == 3
    assert all(path.suffix == ".png" for path in found)


def test_find_image_files_is_sorted(temp_image_dir: Path) -> None:
    found = find_image_files(temp_image_dir)
    assert found == sorted(found)


def test_dataset_length_and_item_shape(temp_image_dir: Path) -> None:
    dataset = UnlabeledImageDataset(temp_image_dir, TRANSFORM, max_images=10)
    assert len(dataset) == 3

    first_item = dataset[0]
    assert isinstance(first_item, torch.Tensor)
    assert first_item.shape == (3, 8, 8)  # channels, height, width


def test_dataset_respects_max_images(temp_image_dir: Path) -> None:
    dataset = UnlabeledImageDataset(temp_image_dir, TRANSFORM, max_images=2)
    assert len(dataset) == 2


def test_dataset_raises_when_directory_has_no_images(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        UnlabeledImageDataset(empty_dir, TRANSFORM, max_images=10)


def test_dataloader_yields_correctly_sized_batches(temp_image_dir: Path) -> None:
    loader = build_image_dataloader(
        temp_image_dir, TRANSFORM, batch_size=2, max_images=10
    )
    batches = list(loader)

    # 3 images at batch size 2 -> one full batch of 2 and one partial batch of 1.
    assert len(batches) == 2
    assert batches[0].shape == (2, 3, 8, 8)
    assert batches[1].shape == (1, 3, 8, 8)
