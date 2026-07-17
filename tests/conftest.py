"""
conftest.py
===========

Shared pytest fixtures. Anything defined here is automatically available to
every test file in this directory without needing an import.
"""

from pathlib import Path

import pytest
from PIL import Image


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
