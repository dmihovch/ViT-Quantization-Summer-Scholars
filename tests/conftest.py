"""Shared pytest fixtures for the ViT quantization test suite.

All fixtures are designed to be fast (no model weights loaded, no network
access) and deterministic (fixed seeds, known tensor values).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.hooks import LayerStats


@pytest.fixture()
def temp_image_dir(tmp_path: Path) -> Path:
    """Create a minimal ImageFolder-compatible directory tree.

    Layout::

        <tmp_path>/
            class_0/
                img_0.png
                img_1.png
                decoy_0.txt
            class_1/
                img_0.png
                img_1.png
                decoy_1.txt
            class_2/
                img_0.png
                img_1.png
                decoy_2.txt
            class_3/
                img_0.png
                img_1.png
                decoy_3.txt

    Each PNG is a tiny 8×8 RGB image filled with a constant colour so that
    construction is instantaneous.  The ``.txt`` decoys verify that
    ``ImageFolder`` correctly ignores non-image files.

    Parameters
    ----------
    tmp_path:
        pytest-supplied temporary directory (unique per test invocation).

    Returns
    -------
    Path
        Root of the constructed ImageFolder tree.
    """
    rng = np.random.default_rng(seed=0)
    for class_idx in range(4):
        class_dir = tmp_path / f"class_{class_idx}"
        class_dir.mkdir()
        for img_idx in range(2):
            colour = tuple(rng.integers(0, 256, size=3).tolist())
            img = Image.new("RGB", (8, 8), colour)
            img.save(class_dir / f"img_{img_idx}.png")
        (class_dir / f"decoy_{class_idx}.txt").write_text("not an image\n")
    return tmp_path


@pytest.fixture()
def tiny_layer_stats() -> dict[str, LayerStats]:
    """Return a dict of three fake :class:`~src.hooks.LayerStats` entries.

    Values are chosen to be easy to reason about in tests:

    - ``std=2.0``, ``mean=0.1`` — effectively centred Gaussian-ish
    - ``max=8.0``, ``min=-8.0`` — symmetric, four standard deviations out
    - ``kurtosis=1.5`` — leptokurtic (heavy-tailed)
    - ``outlier_frac`` — small but non-zero fractions for each threshold

    Keys use the ``"{layer_name}/{site}"`` format expected by the stats dict.

    Returns:
        Mapping from ``"{layer_name}/{site}"`` to :class:`~src.hooks.LayerStats`.
    """
    names = ["blocks.0.mlp.act", "blocks.6.mlp.act", "blocks.11.mlp.act"]
    return {
        f"{name}/pre_gelu": LayerStats(
            site="pre_gelu",
            layer_name=name,
            max=8.0,
            min=-8.0,
            mean=0.1,
            std=2.0,
            kurtosis=1.5,
            outlier_frac={"3": 0.0027, "4": 0.0001, "6": 0.0},
            per_channel_std=None,
            attn_entropy=None,
            n_samples=1_000_000,
        )
        for name in names
    }


@pytest.fixture()
def dummy_tensor() -> torch.Tensor:
    """Return a realistic ViT FFN pre-GELU-shaped tensor filled with randn.

    Shape is ``(4, 197, 3072)``: batch=4, tokens=197 (196 patches + CLS),
    hidden=3072 (4× the ViT-B/16 embed dim of 768).

    The seed is fixed to 0 so tests that inspect specific values are
    deterministic across runs.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(4, 197, 3072)``.
    """
    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.randn(4, 197, 3072, generator=generator)
