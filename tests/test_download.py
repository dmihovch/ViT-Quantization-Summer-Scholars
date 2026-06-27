"""
test_download.py
================

Tests for the ImageNet downloader's network-free logic: argument parsing,
image saving, and label/image coercion. The actual streaming download is not
tested here because it requires network access and (for the official dataset)
Hugging Face authentication.
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

from download_imagenet_val import (
    coerce_to_label,
    coerce_to_pil_image,
    parse_config,
    save_image,
)
from src.data_loader import find_image_files


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["download_imagenet_val.py"])
    config = parse_config()
    assert config.num_images == 4096
    assert config.split == "validation"
    assert config.output_dir == Path("data")


def test_custom_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_imagenet_val.py", "--num-images", "128", "--output-dir", "data/tmp"],
    )
    config = parse_config()
    assert config.num_images == 128
    assert config.output_dir == Path("data/tmp")


def test_save_image_writes_jpeg_in_class_folder(tmp_path: Path) -> None:
    image = Image.new("RGB", (10, 10), color=(1, 2, 3))
    save_image(image, label=7, index=3, output_dir=tmp_path)

    expected_path = tmp_path / "class_007" / "val_00003.jpeg"
    assert expected_path.exists()
    assert Image.open(expected_path).size == (10, 10)


def test_save_image_converts_grayscale_to_rgb(tmp_path: Path) -> None:
    grayscale_image = Image.new("L", (8, 8), color=128)
    save_image(grayscale_image, label=0, index=0, output_dir=tmp_path)

    saved = Image.open(tmp_path / "class_000" / "val_00000.jpeg")
    assert saved.mode == "RGB"


def test_save_image_without_label_uses_unlabeled_folder(tmp_path: Path) -> None:
    save_image(Image.new("RGB", (8, 8)), label=None, index=0, output_dir=tmp_path)
    assert (tmp_path / "unlabeled" / "val_00000.jpeg").exists()


def test_saved_images_are_discoverable_by_the_data_loader(tmp_path: Path) -> None:
    save_image(Image.new("RGB", (8, 8)), label=1, index=0, output_dir=tmp_path)
    save_image(Image.new("RGB", (8, 8)), label=2, index=1, output_dir=tmp_path)

    found = find_image_files(tmp_path)
    assert len(found) == 2


def test_coerce_to_label_handles_ints_and_missing() -> None:
    assert coerce_to_label(217) == 217
    assert coerce_to_label(None) is None
    assert coerce_to_label("not a label") is None
    assert coerce_to_label(True) is None  # bool must not be treated as a label


def test_coerce_to_pil_image_passes_through_and_decodes_bytes() -> None:
    original = Image.new("RGB", (4, 4), color=(9, 9, 9))
    assert coerce_to_pil_image(original) is original

    # The undecoded form: a dict carrying raw PNG bytes.
    buffer = io.BytesIO()
    original.save(buffer, format="PNG")
    decoded = coerce_to_pil_image({"bytes": buffer.getvalue()})
    assert decoded.size == (4, 4)
