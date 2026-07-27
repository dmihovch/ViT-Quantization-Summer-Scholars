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
    fig, ax = plt.subplots(figsize=(7, 4))
    flat = activations.ravel()

    ax.hist(flat, bins=200, color="steelblue", alpha=0.8, density=False)

    if log_scale:
        ax.set_yscale("log")

    # Annotate mean ± 3σ and ± 6σ boundaries.
    mu = float(np.mean(flat))
    sigma = float(np.std(flat))
    ylim = ax.get_ylim()
    ax.axvline(mu, color="black", linestyle="-", linewidth=0.8, label=f"μ = {mu:.3f}")
    for k, style, lbl in [
        (3, "--", f"±3σ = [{mu - 3 * sigma:.3f}, {mu + 3 * sigma:.3f}]"),
        (6, ":", f"±6σ = [{mu - 6 * sigma:.3f}, {mu + 6 * sigma:.3f}]"),
    ]:
        ax.axvline(mu - k * sigma, color="red", linestyle=style, linewidth=0.8)
        ax.axvline(mu + k * sigma, color="red", linestyle=style, linewidth=0.8, label=lbl)

    ax.set_ylim(ylim)  # restore after vline additions
    ax.set_title(layer_name, fontsize=10)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count" + (" (log scale)" if log_scale else ""))
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved histogram to %s", output_path)


def plot_per_channel_std_heatmap(
    per_channel_stds: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save a heatmap of per-channel standard deviations across layers.

    Parameters
    ----------
    per_channel_stds:
        Mapping from layer name (e.g. ``"blocks.3/pre_gelu"``) to a list of
        per-channel population standard deviations.  All lists must have the
        same length (the channel dimension D or D_mlp).
    output_path:
        File path where the PNG is written.  Parent directories must exist.
    """
    if not per_channel_stds:
        logger.warning("Empty per_channel_stds dict; skipping heatmap.")
        return

    # Sort keys for deterministic layer ordering.
    sorted_keys = sorted(per_channel_stds.keys())
    data = np.array([per_channel_stds[k] for k in sorted_keys])  # (L, D)

    fig, ax = plt.subplots(figsize=(12, max(4, len(sorted_keys) * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels(sorted_keys, fontsize=7)
    ax.set_xlabel("Channel index")
    ax.set_title("Per-channel population σ (layers × channels)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("σ (population)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved per-channel σ heatmap to %s", output_path)


def plot_attention_entropy_heatmap(
    entropies: dict[str, list[float]],
    output_path: Path,
    title: str = "Attention entropy per head (nats)",
) -> None:
    """Save a heatmap of per-head mean attention entropy across blocks.

    Renders a (num_blocks × num_heads) matrix using ``imshow`` with the
    "viridis" colormap.  Each row is one block; each column is one head.

    Parameters
    ----------
    entropies:
        Mapping from block identifier (e.g. ``"blocks.3/post_softmax"``) to a
        list of per-head mean Shannon entropies in nats. All lists must have
        the same length (num_heads).
    output_path:
        File path where the PNG is written.  Parent directories must exist.
    title:
        Plot title (used to distinguish CLS vs patch heatmaps).
    """
    if not entropies:
        logger.warning("Empty entropies dict; skipping attention entropy heatmap.")
        return

    # Sort keys for deterministic block ordering.
    sorted_keys = sorted(entropies.keys())
    data = np.array([entropies[k] for k in sorted_keys])  # (num_blocks, num_heads)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_keys) * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels(sorted_keys, fontsize=7)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Block")
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean entropy (nats)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved attention entropy heatmap to %s", output_path)


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
