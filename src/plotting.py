"""Matplotlib figure generation for all experimental phases.

All public functions save a figure to ``output_path`` and return ``None``.
The ``Agg`` backend is selected at import time so figures can be rendered in
headless/CI environments without a display server.  Every function calls
``plt.close(fig)`` after saving to avoid memory leaks from unclosed figures.

Plot categories
---------------
Phase 1 — from ``profiling_result.json``
    * ``plot_activation_histogram`` — raw activation distribution per site
    * ``plot_per_channel_std_heatmap`` — per-channel σ (layers × channels)
    * ``plot_per_channel_mean_heatmap`` — per-channel μ (layers × channels)
    * ``plot_attention_entropy_heatmap`` — per-head Shannon entropy
    * ``plot_kurtosis_heatmap`` — per-site excess kurtosis
    * ``plot_outlier_fraction_heatmap`` — per-site outlier % at given σ threshold
    * ``plot_ln2_amplification_ratio`` — ‖LN2(x)‖₂ / ‖x_skip‖₂ per block

Phase 2 — from ``ablation_results.csv``
    * ``plot_accuracy_vs_threshold`` — top-1 accuracy vs k
    * ``plot_pct_zeroed_per_layer`` — % zeroed per block at fixed k
    * ``plot_accuracy_comparison`` — overlay two accuracy curves (e.g. global vs per-channel)
    * ``plot_ablation_mode_comparison`` — overlay outlier / mean_only / var_only
    * ``plot_entropy_delta_heatmap`` — per-head entropy change after ablation
    * ``plot_ci_delta`` — 95% CI on accuracy delta
    * ``plot_effective_channels`` — effective channels preserved per block
    * ``plot_degradation_efficiency`` — accuracy loss per unit sparsity
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.ablation import AblationResult  # noqa: E402 — must follow matplotlib.use()
from src.plotting_utils import (
    ANALYTICAL_COLORS,
    LABELS,
    block_sort_key,
    extract_block_index,
    format_site_label,
    site_sort_key,
)

# Backward-compatible aliases (used by tests).
_site_sort_key = site_sort_key
_block_sort_key = block_sort_key

logger = logging.getLogger(__name__)


# ===========================================================================
# Phase 1 plots — from profiling_result.json
# ===========================================================================


def plot_activation_histogram(
    activations: np.ndarray,
    layer_name: str,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """Save a histogram of activation values for a single layer.

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
        tails of the distribution.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    flat = activations.ravel()

    ax.hist(flat, bins=200, color="steelblue", alpha=0.8, density=False)

    if log_scale:
        ax.set_yscale("log")

    # Annotate mean ± 3σ, ± 4σ, and ± 6σ boundaries (matching OUTLIER_SIGMAS).
    mu = float(np.mean(flat))
    sigma = float(np.std(flat))
    ylim = ax.get_ylim()
    ax.axvline(mu, color="black", linestyle="-", linewidth=0.8, label=f"μ = {mu:.3f}")
    for k, style, lbl in [
        (3, "--", f"±3σ = [{mu - 3 * sigma:.3f}, {mu + 3 * sigma:.3f}]"),
        (4, "-.", f"±4σ = [{mu - 4 * sigma:.3f}, {mu + 4 * sigma:.3f}]"),
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


# ---------------------------------------------------------------------------
# Per-channel heatmaps (σ and μ)
# ---------------------------------------------------------------------------


def _plot_per_channel_heatmap(
    data: dict[str, list[float]],
    output_path: Path,
    cbar_label: str,
    title: str,
) -> None:
    """Generic per-channel heatmap: layers × channels.

    Uses a consistent aspect ratio regardless of the number of rows.
    For high-dimensional data (e.g. 768 or 3072 channels), the heatmap
    is replaced with a summary line plot showing per-layer statistics
    (mean ± std band) rather than a dense, unreadable barcode.
    """
    if not data:
        logger.warning("Empty data dict; skipping per-channel heatmap.")
        return

    sorted_keys = sorted(data.keys(), key=site_sort_key)
    matrix = np.array([data[k] for k in sorted_keys])  # (L, D)
    num_layers, num_channels = matrix.shape

    # For high-dimensional data, use a summary line plot instead of a
    # dense heatmap that looks like an unreadable barcode.
    if num_channels > 100:
        _plot_per_channel_summary_line(matrix, sorted_keys, output_path,
                                        cbar_label, title, num_channels)
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(sorted_keys) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels([format_site_label(k) for k in sorted_keys], fontsize=7)
    ax.set_xlabel(LABELS["channel_index"])
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved per-channel heatmap to %s", output_path)


def _plot_per_channel_summary_line(
    matrix: np.ndarray,
    sorted_keys: list[str],
    output_path: Path,
    cbar_label: str,
    title: str,
    num_channels: int,
) -> None:
    """Plot per-layer summary statistics as a line chart with ±1σ band.

    Replaces the dense barcode heatmap for high-dimensional channel data.
    Shows the mean per-channel value per layer with a shaded band for
    ±1 standard deviation, plus min/max envelope lines to highlight
    rogue outlier channels.
    """
    num_layers = matrix.shape[0]
    layer_means = np.mean(matrix, axis=1)
    layer_stds = np.std(matrix, axis=1)
    layer_mins = np.min(matrix, axis=1)
    layer_maxs = np.max(matrix, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(num_layers)

    # ±1σ band.
    ax.fill_between(x,
                    layer_means - layer_stds,
                    layer_means + layer_stds,
                    alpha=0.2, color="steelblue", linewidth=0,
                    label="±1σ of per-channel values")
    # Min/max envelope (dashed).
    ax.plot(x, layer_mins, "--", color="coral", linewidth=0.8, alpha=0.7,
            label="Min / Max per layer")
    ax.plot(x, layer_maxs, "--", color="coral", linewidth=0.8, alpha=0.7)
    # Mean line.
    ax.plot(x, layer_means, "o-", color="steelblue", linewidth=2, markersize=6,
            label="Mean per-channel value")

    ax.set_xticks(x)
    ax.set_xticklabels([format_site_label(k) for k in sorted_keys],
                        rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel(cbar_label)
    ax.set_title(f"{title}\n({num_channels} channels per layer — summary statistics)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved per-channel summary line plot to %s", output_path)


def plot_per_channel_std_heatmap(
    per_channel_stds: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save a heatmap of per-channel standard deviations across layers.

    Parameters
    ----------
    per_channel_stds:
        Mapping from site identifier (e.g. ``"blocks.3/pre_gelu"``) to a list of
        per-channel population standard deviations.  All lists must have the
        same length (the channel dimension D or D_mlp).
    output_path:
        File path where the PNG is written.  Parent directories must exist.
    """
    _plot_per_channel_heatmap(
        per_channel_stds, output_path,
        cbar_label="σ (population)",
        title="Per-channel population σ (layers × channels)",
    )


def plot_per_channel_mean_heatmap(
    per_channel_means: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save a heatmap of per-channel means across layers.

    Parameters
    ----------
    per_channel_means:
        Mapping from site identifier to a list of per-channel population means.
    output_path:
        File path where the PNG is written.
    """
    _plot_per_channel_heatmap(
        per_channel_means, output_path,
        cbar_label="μ (population)",
        title="Per-channel population μ (layers × channels)",
    )


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------


def plot_attention_entropy_heatmap(
    entropies: dict[str, list[float]],
    output_path: Path,
    title: str = "Attention entropy per head (nats)",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
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
    vmin, vmax:
        Fixed colour scale limits.  When provided, the colour mapping is
        locked so that CLS and patch entropy heatmaps can be compared
        visually with a shared scale.
    """
    if not entropies:
        logger.warning("Empty entropies dict; skipping attention entropy heatmap.")
        return

    sorted_keys = sorted(entropies.keys(), key=site_sort_key)
    data = np.array([entropies[k] for k in sorted_keys])  # (num_blocks, num_heads)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_keys) * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest",
                   vmin=vmin, vmax=vmax)

    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels([format_site_label(k) for k in sorted_keys], fontsize=7)
    ax.set_xlabel(LABELS["head_index"])
    ax.set_ylabel(LABELS["block"])
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean entropy (nats)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved attention entropy heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# Kurtosis heatmap
# ---------------------------------------------------------------------------


def plot_kurtosis_heatmap(
    kurtosis_by_site: dict[str, float],
    output_path: Path,
) -> None:
    """Save a heatmap of per-site excess kurtosis across blocks.

    Renders a (num_blocks × num_sites) matrix using ``imshow``.  Each row is
    one block; each column is one measurement site.  Kurtosis > 0 indicates
    heavy tails relative to a Gaussian (leptokurtic).

    Parameters
    ----------
    kurtosis_by_site:
        Mapping from site_identifier (e.g. ``"blocks.3/pre_gelu"``) to
        excess kurtosis value.  Gaussian = 0, heavier tails > 0.
    output_path:
        File path where the PNG is written.
    """
    if not kurtosis_by_site:
        logger.warning("Empty kurtosis dict; skipping heatmap.")
        return

    # Group by site type and block index to build a (blocks × sites) matrix.
    site_types = sorted({
        sid.split("/", 1)[1] for sid in kurtosis_by_site
        if "/" in sid
    })
    blocks = sorted({
        bi
        for sid in kurtosis_by_site
        if "/" in sid and (bi := extract_block_index(sid)) is not None
    })

    if not blocks or not site_types:
        logger.warning("Could not parse block/site structure; skipping kurtosis heatmap.")
        return

    matrix = np.full((len(blocks), len(site_types)), np.nan)
    for b_idx, blk in enumerate(blocks):
        for s_idx, st in enumerate(site_types):
            sid = f"blocks.{blk}/{st}"
            if sid in kurtosis_by_site:
                matrix[b_idx, s_idx] = kurtosis_by_site[sid]

    fig, ax = plt.subplots(figsize=(max(8, len(site_types) * 1.8),
                                     max(4, len(blocks) * 0.4)))
    # Use symlog scale to handle extreme kurtosis spikes (>4000) without
    # letting them hijack the entire colourbar.  Linear threshold = 1.0
    # means values below 1 are linear, above 1 are logarithmic.
    # Also clip vmax to 500 so the Block 6/7 anomaly saturates in deep
    # colour while the rest of the model becomes visible.
    vmin = float(np.nanmin(matrix))
    vmax_clipped = min(float(np.nanmax(matrix)), 500.0)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   norm=matplotlib.colors.SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax_clipped))

    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {b}" for b in blocks], fontsize=7)
    ax.set_xticks(range(len(site_types)))
    ax.set_xticklabels(site_types, fontsize=7, rotation=45, ha="right")
    ax.set_title("Per-site excess kurtosis\n(Gaussian = 0, positive = heavy tails; symlog scale, vmax=500)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Excess kurtosis (symlog)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved kurtosis heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# Outlier fraction heatmap
# ---------------------------------------------------------------------------


def plot_outlier_fraction_heatmap(
    outlier_fractions: dict[str, dict[str, float]],
    sigma_key: str,
    output_path: Path,
    *,
    vmax: float | None = None,
) -> None:
    """Save a heatmap of per-site outlier fractions at a given sigma threshold.

    Renders a (num_blocks × num_sites) matrix.  Uses a log-scale colormap
    because outlier fractions span several orders of magnitude.

    Parameters
    ----------
    outlier_fractions:
        Mapping from site_identifier to a dict of ``{sigma_key: fraction}``,
        e.g. ``"3.0_sigma": 0.0027``.
    sigma_key:
        Which outlier fraction key to plot, e.g. ``"3.0_sigma"``.
    output_path:
        File path where the PNG is written.
    vmax:
        Fixed upper bound for the LogNorm.  When provided, the colour scale
        is locked across calls so that multiple sigma thresholds can be
        compared visually.  When None, uses the data max.
    """
    # Extract scalar per site.
    scalar: dict[str, float] = {}
    for sid, fracs in outlier_fractions.items():
        if sigma_key in fracs:
            scalar[sid] = fracs[sigma_key]

    if not scalar:
        logger.warning("No data for sigma_key=%s; skipping outlier fraction heatmap.", sigma_key)
        return

    site_types = sorted({
        sid.split("/", 1)[1] for sid in scalar
        if "/" in sid
    })
    blocks = sorted({
        bi
        for sid in scalar
        if "/" in sid and (bi := extract_block_index(sid)) is not None
    })

    if not blocks or not site_types:
        logger.warning("Could not parse block/site structure; skipping outlier fraction heatmap.")
        return

    matrix = np.full((len(blocks), len(site_types)), np.nan)
    for b_idx, blk in enumerate(blocks):
        for s_idx, st in enumerate(site_types):
            sid = f"blocks.{blk}/{st}"
            if sid in scalar:
                matrix[b_idx, s_idx] = scalar[sid]

    fig, ax = plt.subplots(figsize=(max(8, len(site_types) * 1.8),
                                     max(4, len(blocks) * 0.4)))
    data_min = float(np.nanmin(matrix))
    data_max = float(np.nanmax(matrix))
    # Use provided vmax for consistent scaling, or data max.
    colour_vmax = vmax if vmax is not None else data_max
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   norm=matplotlib.colors.LogNorm(vmin=max(1e-5, data_min),
                                                  vmax=colour_vmax))

    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {b}" for b in blocks], fontsize=7)
    ax.set_xticks(range(len(site_types)))
    ax.set_xticklabels(site_types, fontsize=7, rotation=45, ha="right")
    ax.set_title(f"Outlier fraction at {sigma_key.replace('_', ' ')}", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved outlier fraction heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# LN2 amplification ratio
# ---------------------------------------------------------------------------


def plot_ln2_amplification_ratio(
    ratios: dict[str, float],
    output_path: Path,
) -> None:
    """Save a bar chart of LN2 amplification ratio per block.

    The ratio ‖LN2(x)‖₂ / ‖x_skip‖₂ measures how aggressively the second
    LayerNorm scales the signal before it enters the MLP — the primary
    driver of pre-GELU activation range expansion (Bondarenko et al. 2021,
    §4.2; Wei et al. 2022, §3.1).

    Parameters
    ----------
    ratios:
        Mapping from site_identifier (e.g. ``"blocks.3/residual_stream"``) to
        the mean LN2 amplification ratio.
    output_path:
        File path where the PNG is written.
    """
    if not ratios:
        logger.warning("Empty ratios dict; skipping LN2 amplification plot.")
        return

    sorted_ids = sorted(ratios.keys(), key=site_sort_key)
    # Format labels as "Block N" instead of raw "blocks.N".
    labels: list[str] = []
    for sid in sorted_ids:
        bi = extract_block_index(sid)
        if bi is not None:
            labels.append(f"Block {bi}")
        else:
            labels.append(sid.replace("/residual_stream", ""))
    values = [ratios[sid] for sid in sorted_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ANALYTICAL_COLORS["histogram"] for _ in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Ratio = 1 (no amplification)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("Encoder Block")
    ax.set_ylabel(r"$\|\mathrm{LN2}(x)\|_2 \,/\, \|x_{\mathrm{skip}}\|_2$")
    ax.set_title("LN2 Amplification Ratio per Block\n(>1 = LN2 amplifies signal before MLP)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved LN2 amplification ratio plot to %s", output_path)


# ===========================================================================
# Phase 2 plots — from ablation_results.csv
# ===========================================================================


def plot_accuracy_vs_threshold(
    results: list[AblationResult],
    output_path: Path,
    label: str | None = None,
    color: str = "steelblue",
) -> None:
    """Save a dot plot of top-1 accuracy against sigma threshold.

    Aggregates results across all layers for each unique
    ``sigma_threshold`` value and plots the mean top-1 accuracy with
    baseline shown as a horizontal reference line.  Uses unconnected
    markers because sigma thresholds are discrete hyperparameter
    evaluations, not a continuous variable.

    Parameters
    ----------
    results:
        Full list of :class:`~ablation.AblationResult` from Phase 2
        (all for a single site).
    output_path:
        Destination PNG path.
    label:
        Legend label for this curve.  Defaults to None (no legend entry).
    color:
        Marker colour.
    """
    if not results:
        logger.warning("Empty results; skipping accuracy-vs-threshold plot.")
        return

    # Group by sigma_threshold, average top-1 across layers.
    by_k: dict[float, list[float]] = {}
    baseline = results[0].baseline_top1
    for r in results:
        by_k.setdefault(r.sigma_threshold, []).append(r.top1_accuracy)

    ks = sorted(by_k.keys())
    means = [np.mean(by_k[k]) for k in ks]
    stds = [np.std(by_k[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(ks, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.15, linewidth=0)
    # Use unconnected markers — sigma thresholds are discrete, not continuous.
    ax.plot(ks, means, "o", color=color, linewidth=2, markersize=8,
            label=label)
    ax.axhline(baseline, color=ANALYTICAL_COLORS["baseline"], linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.2f}%)")
    ax.set_xlabel(LABELS["sigma_threshold"])
    ax.set_ylabel(LABELS["accuracy"])
    ax.set_title(f"Accuracy vs sigma threshold — {results[0].site}")
    if label:
        ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved accuracy-vs-threshold plot to %s", output_path)


def plot_pct_zeroed_per_layer(
    results: list[AblationResult],
    sigma_k: float,
    output_path: Path,
) -> None:
    """Save a bar chart of percentage zeroed for each layer at a single threshold.

    Filters ``results`` to rows matching ``sigma_k`` and plots one bar per
    layer, sorted by site_identifier using numeric block ordering.

    Parameters
    ----------
    results:
        Full list of :class:`~ablation.AblationResult` from Phase 2
        (all for a single site).
    sigma_k:
        Threshold multiplier to filter on (e.g. ``3.0`` for 3σ).
    output_path:
        Destination PNG path.
    """
    filtered = [r for r in results if r.sigma_threshold == sigma_k]
    if not filtered:
        logger.warning(
            "No results for sigma_k=%.1f; skipping pct-zeroed plot.", sigma_k
        )
        return

    filtered.sort(key=lambda r: site_sort_key(r.site_identifier))
    labels = [format_site_label(r.site_identifier) for r in filtered]
    values = [r.pct_zeroed for r in filtered]

    # Cap figure height to prevent absurdly tall plots with many layers.
    fig_height = min(12, max(4, len(labels) * 0.3))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=ANALYTICAL_COLORS["per_channel"], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(LABELS["pct_zeroed"])
    ax.set_title(f"Percentage zeroed at k={sigma_k} — {filtered[0].site}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved pct-zeroed plot to %s", output_path)


# ---------------------------------------------------------------------------
# Accuracy comparison (overlay two curves)
# ---------------------------------------------------------------------------


def plot_accuracy_comparison(
    results_a: list[AblationResult],
    results_b: list[AblationResult],
    output_path: Path,
    label_a: str = "A",
    label_b: str = "B",
    title: str | None = None,
) -> None:
    """Grouped bar chart comparing accuracy at discrete sigma thresholds.

    Uses grouped bars instead of a line plot because sigma thresholds are
    discrete categorical steps (3.0, 4.0, 6.0), not a continuous variable.
    A dashed baseline line anchors the comparison.

    Parameters
    ----------
    results_a:
        First set of ablation results.
    results_b:
        Second set of ablation results.
    output_path:
        Destination PNG path.
    label_a:
        Legend label for the first condition.
    label_b:
        Legend label for the second condition.
    title:
        Override plot title.  If None, derived from ``results_a[0].site``.
    """
    if not results_a or not results_b:
        logger.warning("Empty results; skipping accuracy comparison plot.")
        return

    baseline = results_a[0].baseline_top1

    def _group(results: list[AblationResult]) -> tuple[list[float], list[float], list[float]]:
        by_k: dict[float, list[float]] = {}
        for r in results:
            by_k.setdefault(r.sigma_threshold, []).append(r.top1_accuracy)
        ks = sorted(by_k.keys())
        return ks, [np.mean(by_k[k]) for k in ks], [np.std(by_k[k]) for k in ks]

    ks_a, means_a, stds_a = _group(results_a)
    ks_b, means_b, stds_b = _group(results_b)

    # Use common ks (intersection) for grouped bars.
    common_ks = sorted(set(ks_a) & set(ks_b))
    if not common_ks:
        common_ks = sorted(set(ks_a) | set(ks_b))

    # Build lookup for common ks.
    lookup_a = dict(zip(ks_a, zip(means_a, stds_a)))
    lookup_b = dict(zip(ks_b, zip(means_b, stds_b)))

    m_a = [lookup_a[k][0] for k in common_ks]
    s_a = [lookup_a[k][1] for k in common_ks]
    m_b = [lookup_b[k][0] for k in common_ks]
    s_b = [lookup_b[k][1] for k in common_ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(common_ks))
    width = 0.35

    ax.bar(x - width / 2, m_a, width, label=label_a,
           color=ANALYTICAL_COLORS["global"], edgecolor="black", linewidth=0.5,
           yerr=s_a, capsize=4)
    ax.bar(x + width / 2, m_b, width, label=label_b,
           color=ANALYTICAL_COLORS["per_channel"], edgecolor="black", linewidth=0.5,
           yerr=s_b, capsize=4)

    ax.axhline(baseline, color=ANALYTICAL_COLORS["baseline"], linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.2f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} σ" for k in common_ks], fontsize=10)
    ax.set_xlabel(LABELS["sigma_threshold"])
    ax.set_ylabel(LABELS["accuracy"])
    ax.set_title(title or f"Accuracy at discrete thresholds — {results_a[0].site}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Consistent y-axis: start from 0 to avoid exaggerating differences.
    ax.set_ylim(0, baseline + 5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy comparison plot to %s", output_path)


# ---------------------------------------------------------------------------
# Ablation mode comparison (outlier / mean_only / var_only)
# ---------------------------------------------------------------------------


def plot_ablation_mode_comparison(
    mode_results: dict[str, list[AblationResult]],
    output_path: Path,
    sigma_k: float | None = None,
) -> None:
    """Compare accuracy across ablation modes at a given sigma threshold.

    Each mode (``"outlier"``, ``"mean_only"``, ``"var_only"``) gets its own
    grouped bar showing top-1 accuracy vs baseline.

    Parameters
    ----------
    mode_results:
        Mapping from mode name to list of :class:`AblationResult`.
        Typical keys: ``"outlier"``, ``"mean_only"``, ``"var_only"``.
    output_path:
        Destination PNG path.
    sigma_k:
        If provided, filter results to this threshold only.  If None, uses
        the first unique sigma_threshold found.
    """
    if not mode_results:
        logger.warning("Empty mode_results; skipping ablation mode comparison.")
        return

    # Determine sigma_k if not provided.
    if sigma_k is None:
        for results in mode_results.values():
            if results:
                sigma_k = results[0].sigma_threshold
                break
    if sigma_k is None:
        return

    baseline: float | None = None
    mode_names: list[str] = []
    accuracies: list[float] = []
    acc_stds: list[float] = []

    for mode, results in mode_results.items():
        filtered = [r for r in results if r.sigma_threshold == sigma_k]
        if not filtered:
            continue
        if baseline is None:
            baseline = filtered[0].baseline_top1
        mode_names.append(mode)
        accs = [r.top1_accuracy for r in filtered]
        accuracies.append(np.mean(accs))
        acc_stds.append(np.std(accs))

    if not mode_names:
        logger.warning("No results at k=%.1f for any mode.", sigma_k)
        return

    fig, ax = plt.subplots(figsize=(max(6, len(mode_names) * 1.5), 5))
    x = np.arange(len(mode_names))
    mode_colors = [ANALYTICAL_COLORS["per_channel"], ANALYTICAL_COLORS["global"], "seagreen"][:len(mode_names)]
    ax.bar(x, accuracies, yerr=acc_stds, color=mode_colors, edgecolor="black", linewidth=0.5, width=0.5,
           capsize=5)

    if baseline is not None:
        ax.axhline(baseline, color=ANALYTICAL_COLORS["baseline"], linestyle="--", linewidth=1,
                   label=f"Baseline ({baseline:.2f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, fontsize=9)
    ax.set_ylabel(LABELS["accuracy"])
    ax.set_title(f"Ablation mode comparison at k={sigma_k}")
    if baseline is not None:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Set ylim to show the baseline clearly, but don't compress differences.
    min_acc = min(accuracies) if accuracies else 0
    max_acc = max(accuracies) if accuracies else 100
    y_lower = max(0, min(min_acc, (baseline or 100)) - 5)
    y_upper = max(max_acc, (baseline or 100)) + 5
    ax.set_ylim(y_lower, y_upper)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ablation mode comparison to %s", output_path)


# ---------------------------------------------------------------------------
# Entropy delta heatmap
# ---------------------------------------------------------------------------


def plot_entropy_delta_heatmap(
    per_head_deltas: dict[str, list[float]],
    output_path: Path,
    title: str | None = None,
) -> None:
    """Save a heatmap of per-head entropy deltas across blocks.

    Renders a (num_heads × num_blocks) matrix using ``imshow`` with a
    diverging colormap.  Each row is one attention head; each column is
    one block.  Positive deltas (red) mean entropy *increased* after
    ablation (more uniform attention); negative (blue) mean it decreased.

    Parameters
    ----------
    per_head_deltas:
        Mapping from site_identifier (e.g. ``"blocks.3/pre_softmax"``) to
        a list of per-head entropy deltas ``[H]``.
    output_path:
        Destination PNG path.
    title:
        Override plot title.
    """
    if not per_head_deltas:
        logger.warning("No per-head entropy delta data; skipping.")
        return

    # Build (num_heads, num_blocks) matrix.
    entries: list[tuple[int, list[float]]] = []
    for sid, deltas in per_head_deltas.items():
        bi = extract_block_index(sid)
        if bi is None:
            continue
        entries.append((bi, deltas))

    if not entries:
        logger.warning("No valid block indices in entropy delta data; skipping.")
        return

    entries.sort(key=lambda x: x[0])
    num_blocks = max(b for b, _ in entries) + 1
    num_heads = len(entries[0][1]) if entries else 1

    matrix = np.full((num_heads, num_blocks), np.nan)
    for blk, deltas in entries:
        for h in range(min(num_heads, len(deltas))):
            matrix[h, blk] = deltas[h]

    # Diverging colormap centred at 0.
    v_abs = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-10)

    fig, ax = plt.subplots(figsize=(max(8, num_blocks * 0.6),
                                     max(4, num_heads * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   vmin=-v_abs, vmax=v_abs)

    ax.set_yticks(range(num_heads))
    ax.set_yticklabels([f"Head {h}" for h in range(num_heads)], fontsize=8)
    ax.set_xticks(range(num_blocks))
    ax.set_xticklabels([f"{b}" for b in range(num_blocks)], fontsize=8)
    ax.set_xlabel(LABELS["block"], fontsize=10)
    ax.set_ylabel(LABELS["head_index"], fontsize=10)
    ax.set_title(title or "Per-head entropy delta", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(LABELS["delta_entropy"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved entropy delta heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# 95% CI on accuracy delta
# ---------------------------------------------------------------------------


def plot_ci_delta(
    ci_results: dict[float, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot 95% CI for per-channel minus global accuracy delta.

    Computed via closed-form two-proportion z-interval.

    Parameters
    ----------
    ci_results:
        Mapping from sigma_k to ``{"delta_point_estimate", "delta_ci_low_pct",
        "delta_ci_high_pct"}``, as produced by ``_compute_ci_delta`` in
        ``scripts/analyze_ablation_results.py``.
    output_path:
        Destination PNG path.
    """
    if not ci_results:
        logger.warning("Empty CI results; skipping CI delta plot.")
        return

    ks = sorted(ci_results.keys())
    deltas = [ci_results[k]["delta_point_estimate"] for k in ks]
    ci_lows = [ci_results[k]["delta_ci_low_pct"] for k in ks]
    ci_highs = [ci_results[k]["delta_ci_high_pct"] for k in ks]
    errors_low = [d - l for d, l in zip(deltas, ci_lows)]
    errors_high = [h - d for d, h in zip(deltas, ci_highs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    # Use unconnected dots (no line) — sigma thresholds are discrete
    # hyperparameter evaluations, not a continuous variable.
    ax.errorbar(
        ks, deltas,
        yerr=[errors_low, errors_high],
        fmt="o", capsize=5, color=ANALYTICAL_COLORS["per_channel"], markersize=10,
        elinewidth=1.5, label="Per-channel − Global (95% CI)",
    )
    ax.axhline(y=0, color=ANALYTICAL_COLORS["baseline"], linestyle="--", linewidth=0.8)
    ax.set_xlabel(LABELS["sigma_threshold"])
    ax.set_ylabel(LABELS["delta_accuracy"])
    ax.set_title("Per-Channel vs Global Ablation Delta\n(95% CI, two-proportion z-interval)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved CI delta plot to %s", output_path)


# ---------------------------------------------------------------------------
# Effective channels preserved (migrated from analyze_ablation_results.py)
# ---------------------------------------------------------------------------


def plot_effective_channels(
    channels: dict[str, dict[str, float]],
    sigma_k: float,
    output_path: Path,
    dim_label: str = "D_mlp",
    channel_count: int = 3072,
) -> None:
    """Plot ablated (dropped) channels: two conditions per block.

    Plots *dropped* channels rather than preserved channels to avoid the
    "dynamite plot" trap where all bars look identical because they hover
    near the top of a 0–3072 y-axis.  Comparing a bar of height 5 against
    a bar of height 20 instantly communicates a 4× difference.

    Parameters
    ----------
    channels:
        Mapping from site_identifier to ``{"global_channels", "pc_channels",
        "delta_channels", ...}``.
    sigma_k:
        Sigma threshold for the plot title.
    output_path:
        Destination PNG path.
    dim_label:
        Label for the channel dimension (e.g. ``"D_mlp"`` or ``"D"``).
    channel_count:
        Total number of channels (for the reference line and axis label).
    """
    if not channels:
        logger.warning("Empty channels dict; skipping effective channels plot.")
        return

    sids = sorted(channels.keys(), key=site_sort_key)
    labels = [format_site_label(sid) for sid in sids]
    x = np.arange(len(sids))
    width = 0.35

    # Plot *dropped* channels = total − preserved.
    g_preserved = [channels[sid].get("global_channels", channels[sid].get("channels_a", 0)) for sid in sids]
    p_preserved = [channels[sid].get("pc_channels", channels[sid].get("channels_b", 0)) for sid in sids]
    g_dropped = [channel_count - ch for ch in g_preserved]
    p_dropped = [channel_count - ch for ch in p_preserved]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, g_dropped, width, label="Global", color=ANALYTICAL_COLORS["global"],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, p_dropped, width, label="Per-channel", color=ANALYTICAL_COLORS["per_channel"],
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel(LABELS["block"])
    ax.set_ylabel(f"Channels Dropped (of {channel_count})")
    ax.set_title(f"{dim_label} Channels Dropped at k={sigma_k}\n(Lower = more channels preserved)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved effective channels plot to %s", output_path)


# ---------------------------------------------------------------------------
# Degradation efficiency (migrated from analyze_ablation_results.py)
# ---------------------------------------------------------------------------


def plot_degradation_efficiency(
    deg_results: dict[float, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot Pareto frontier: ΔAccuracy vs % Sparsity for two conditions.

    Replaces the unstable ratio metric (ΔAccuracy / %Sparsity, which blows
    up at near-zero sparsity denominators) with a Pareto frontier scatter
    plot.  Each condition traces a path through (sparsity, accuracy_drop)
    space, with points labelled by sigma threshold k.

    Parameters
    ----------
    deg_results:
        Mapping from sigma_k to ``{"mean_pct_zeroed_a",
        "mean_pct_zeroed_b", "degradation_a_per_pct",
        "degradation_b_per_pct", ...}``.
    output_path:
        Destination PNG path.
    """
    if not deg_results:
        logger.warning("Empty degradation results; skipping efficiency plot.")
        return

    ks = sorted(deg_results.keys())

    # Extract sparsity (x) and accuracy drop (y) for each condition.
    g_sparsity = [deg_results[k].get("mean_pct_zeroed_a", 0) for k in ks]
    p_sparsity = [deg_results[k].get("mean_pct_zeroed_b", 0) for k in ks]
    # Accuracy drop = degradation_per_pct × sparsity (recover from stored values).
    g_drop = [
        deg_results[k].get("degradation_a_per_pct", 0) * deg_results[k].get("mean_pct_zeroed_a", 0)
        for k in ks
    ]
    p_drop = [
        deg_results[k].get("degradation_b_per_pct", 0) * deg_results[k].get("mean_pct_zeroed_b", 0)
        for k in ks
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw connecting lines to show the Pareto frontier.
    ax.plot(g_sparsity, g_drop, "o-", color=ANALYTICAL_COLORS["global"],
            linewidth=2, markersize=10, label="Global", zorder=3)
    ax.plot(p_sparsity, p_drop, "s--", color=ANALYTICAL_COLORS["per_channel"],
            linewidth=2, markersize=10, label="Per-channel", zorder=3)

    # Annotate each point with its k value.
    for k, sx, sy in zip(ks, g_sparsity, g_drop):
        ax.annotate(f"k={k:.0f}", (sx, sy),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9, color=ANALYTICAL_COLORS["global"], fontweight="bold")
    for k, sx, sy in zip(ks, p_sparsity, p_drop):
        ax.annotate(f"k={k:.0f}", (sx, sy),
                    textcoords="offset points", xytext=(10, -12),
                    fontsize=9, color=ANALYTICAL_COLORS["per_channel"], fontweight="bold")

    ax.set_xlabel(LABELS["pct_zeroed"])
    ax.set_ylabel(LABELS["delta_accuracy"])
    ax.set_title("Pareto Frontier: Accuracy Drop vs Sparsity\n(Lower-left = better: less drop at lower sparsity)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved degradation efficiency plot to %s", output_path)
