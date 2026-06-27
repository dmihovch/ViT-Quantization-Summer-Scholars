"""
conftest.py
===========

Shared pytest fixtures. Anything defined here is automatically available to
every test file in this directory without needing an import.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor


@pytest.fixture
def synthetic_activations() -> Tensor:
    """
    A small, fully deterministic activation tensor shaped like a real layer
    output: [Batch=2, Sequence=4, Features=5].

    We deliberately plant *known* outliers so tests can assert exact numbers:
      * feature channel 2 is a persistent outlier (value 10.0 in every token);
      * one extreme spike of 100.0 fixes the maximum magnitude.

    With 2 x 4 = 8 tokens, channel 2 contributes 8 values above 6.0, and the
    single 100.0 spike adds one more, for 9 outliers out of 2 x 4 x 5 = 40.
    """
    activations = torch.zeros(2, 4, 5)
    activations[..., 2] = 10.0  # channel 2: persistent outlier across all tokens
    activations[0, 0, 4] = 100.0  # a single extreme value
    return activations


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """
    Create a throwaway directory holding a few tiny RGB images plus one
    non-image file, so data-loading tests have real files to read.

    `tmp_path` is a built-in pytest fixture: a unique temporary directory that
    is cleaned up automatically after the test.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    for index in range(3):
        tiny_image = Image.new("RGB", (8, 8), color=(index * 10, 0, 0))
        tiny_image.save(image_dir / f"img_{index}.png")

    # A decoy file the loader must ignore.
    (image_dir / "notes.txt").write_text("not an image")

    return image_dir
