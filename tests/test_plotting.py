"""Tests for the plotting utilities in :mod:`src.plotting`.

All tests write to a temporary directory and assert that the expected PNG
file is created.  No model weights are loaded; inputs are synthetic arrays
and LUT instances.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.integer_gelu import GELULut, build_lut
from src.plotting import (
    plot_activation_histogram,
    plot_lut_vs_fp32,
    plot_per_channel_std_heatmap,
)

_SCALE_IN: float = 0.05
_SCALE_OUT: float = 0.05


def test_plot_activation_histogram_creates_file(tmp_path: Path) -> None:
    """plot_activation_histogram must write a PNG file to the given path."""
    rng = np.random.default_rng(seed=1)
    activations = rng.standard_normal(1000).astype(np.float32)
    output_path = tmp_path / "histogram.png"

    plot_activation_histogram(
        activations=activations,
        layer_name="blocks.0.mlp.act",
        output_path=output_path,
        log_scale=True,
    )

    assert output_path.exists(), "Expected histogram PNG was not created."
    assert output_path.stat().st_size > 0, "Histogram PNG is empty."


def test_plot_lut_vs_fp32_creates_file(tmp_path: Path) -> None:
    """plot_lut_vs_fp32 must write a PNG file to the given path."""
    lut_values = build_lut(scale_in=_SCALE_IN, scale_out=_SCALE_OUT)
    gelu_lut = GELULut(
        layer_name="blocks.3.mlp.act",
        scale_in=_SCALE_IN,
        scale_out=_SCALE_OUT,
        lut=lut_values,
    )
    output_path = tmp_path / "lut_vs_fp32.png"

    plot_lut_vs_fp32(lut=gelu_lut, output_path=output_path)

    assert output_path.exists(), "Expected LUT vs FP32 PNG was not created."
    assert output_path.stat().st_size > 0, "LUT vs FP32 PNG is empty."


def test_plot_per_channel_std_heatmap_creates_file(tmp_path: Path) -> None:
    """plot_per_channel_std_heatmap must write a PNG file to the given path."""
    rng = np.random.default_rng(seed=2)
    stds: dict[str, list[float]] = {
        f"blocks.{i}/pre_gelu": rng.random(16).tolist() for i in range(3)
    }
    output_path = tmp_path / "heatmap.png"
    plot_per_channel_std_heatmap(stds, output_path)
    assert output_path.exists(), "Expected heatmap PNG was not created."
    assert output_path.stat().st_size > 0, "Heatmap PNG is empty."
