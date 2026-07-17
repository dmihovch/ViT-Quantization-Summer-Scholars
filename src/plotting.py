"""Matplotlib figure generation for all three experimental phases.

All public functions save a figure to ``output_path`` and return ``None``.
The ``Agg`` backend is selected at import time so figures can be rendered in
headless/CI environments without a display server.  Every function calls
``plt.close(fig)`` after saving to avoid memory leaks from unclosed figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.ablation import AblationResult  # noqa: E402 — must follow matplotlib.use()
from src.integer_gelu import GELULut  # noqa: E402

logger = logging.getLogger(__name__)


def plot_activation_histogram(
    activations: np.ndarray,
    layer_name: str,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """Save a histogram of pre-GELU activation values for a single layer.

    Parameters
    ----------
    activations:
        Flat or multidimensional NumPy array of sampled activation values.
    layer_name:
        Human-readable layer identifier used as the plot title.
    output_path:
        File path where the PNG is written.  Parent directories must exist.
    log_scale:
        If ``True``, the y-axis is rendered on a log scale to reveal the
        tails of the distribution (default behaviour for Phase 1 analysis).
    """
    raise NotImplementedError


def plot_accuracy_vs_threshold(
    results: list[AblationResult],
    output_path: Path,
) -> None:
    """Save a line plot of top-1 accuracy against sigma threshold.

    Aggregates results across all layers for each unique
    ``sigma_threshold`` value and plots the mean top-1 accuracy.

    Parameters
    ----------
    results:
        Full list of :class:`~ablation.AblationResult` from Phase 2.
    output_path:
        Destination PNG path.
    """
    raise NotImplementedError


def plot_pct_zeroed_per_layer(
    results: list[AblationResult],
    sigma_k: float,
    output_path: Path,
) -> None:
    """Save a bar chart of percentage zeroed for each layer at a single threshold.

    Filters ``results`` to rows matching ``sigma_k`` and plots one bar per
    layer, sorted by layer name for reproducibility.

    Parameters
    ----------
    results:
        Full list of :class:`~ablation.AblationResult` from Phase 2.
    sigma_k:
        Threshold multiplier to filter on (e.g. ``3.0`` for 3σ).
    output_path:
        Destination PNG path.
    """
    raise NotImplementedError


def plot_lut_vs_fp32(lut: GELULut, output_path: Path) -> None:
    """Save an overlay of the FP32 GELU curve and the LUT step function.

    Both curves are plotted over the full INT8 input domain ``[-128, 127]``,
    dequantised to FP32 using ``lut.scale_in``.  The LUT approximation is
    rendered as a step function to make the quantisation error visible.

    Parameters
    ----------
    lut:
        The :class:`~integer_gelu.GELULut` to visualise.
    output_path:
        Destination PNG path.
    """
    raise NotImplementedError
