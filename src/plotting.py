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
    * ``plot_bootstrap_ci_delta`` — bootstrap CI on accuracy delta
    * ``plot_effective_channels`` — effective channels preserved per block
    * ``plot_degradation_efficiency`` — accuracy loss per unit sparsity
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.ablation import AblationResult  # noqa: E402 — must follow matplotlib.use()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sort key helpers
#
# Naive ``sorted(site_ids)`` gives string ordering:
#   blocks.0, blocks.1, blocks.10, blocks.11, blocks.2, ...
# which is wrong for heatmaps.  We extract the numeric block index.
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"blocks\.(\d+)")


def _site_sort_key(site_id: str) -> tuple[int, int]:
    """Return a numeric sort key for a site identifier.

    Sorts ``patch_embed/...`` first ``(0, 0)``, then ``blocks.{N}/...`` by
    numeric N ``(1, N)``.  Unknown prefixes sort last ``(2, 0)``.

    Parameters
    ----------
    site_id:
        Site identifier string, e.g. ``"blocks.5/pre_gelu"``.

    Returns
    -------
    tuple[int, int]
        Sort key for use with ``sorted(key=...)``.
    """
    if site_id.startswith("patch_embed"):
        return (0, 0)
    m = _BLOCK_RE.search(site_id)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


def _block_sort_key(site_id: str) -> tuple[int, int | str]:
    """Return a sort key for the block portion of a site identifier."""
    if site_id.startswith("patch_embed"):
        return (0, 0)
    if site_id.startswith("blocks."):
        try:
            n = int(site_id.split(".", 1)[1].split("/")[0])
            return (1, n)
        except (ValueError, IndexError):
            pass
    return (2, site_id)


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
    """Generic per-channel heatmap: layers × channels."""
    if not data:
        logger.warning("Empty data dict; skipping per-channel heatmap.")
        return

    sorted_keys = sorted(data.keys(), key=_site_sort_key)
    matrix = np.array([data[k] for k in sorted_keys])  # (L, D)

    fig, ax = plt.subplots(figsize=(12, max(4, len(sorted_keys) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels(sorted_keys, fontsize=7)
    ax.set_xlabel("Channel index")
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved per-channel heatmap to %s", output_path)


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

    sorted_keys = sorted(entropies.keys(), key=_site_sort_key)
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
        int(_BLOCK_RE.search(sid).group(1))
        for sid in kurtosis_by_site
        if _BLOCK_RE.search(sid) and "/" in sid
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
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", interpolation="nearest",
                   vmin=-2, vmax=5)  # reasonable kurtosis range for activations

    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {b}" for b in blocks], fontsize=7)
    ax.set_xticks(range(len(site_types)))
    ax.set_xticklabels(site_types, fontsize=7, rotation=45, ha="right")
    ax.set_title("Per-site excess kurtosis\n(Gaussian = 0, positive = heavy tails)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Excess kurtosis")

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
        int(_BLOCK_RE.search(sid).group(1))
        for sid in scalar
        if _BLOCK_RE.search(sid) and "/" in sid
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
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   norm=matplotlib.colors.LogNorm(vmin=max(1e-5, np.nanmin(matrix)),
                                                  vmax=max(1e-1, np.nanmax(matrix))))

    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {b}" for b in blocks], fontsize=7)
    ax.set_xticks(range(len(site_types)))
    ax.set_xticklabels(site_types, fontsize=7, rotation=45, ha="right")
    ax.set_title(f"Outlier fraction at {sigma_key.replace('_', ' ')}\n(log scale)", fontsize=10)

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

    sorted_ids = sorted(ratios.keys(), key=_site_sort_key)
    labels = [sid.replace("/residual_stream", "") for sid in sorted_ids]
    values = [ratios[sid] for sid in sorted_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if v >= 0 else "coral" for v in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Ratio = 1 (no amplification)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Encoder Block")
    ax.set_ylabel("‖LN2(x)‖₂ / ‖x_skip‖₂")
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
    linestyle: str = "o-",
) -> None:
    """Save a line plot of top-1 accuracy against sigma threshold.

    Aggregates results across all layers for each unique
    ``sigma_threshold`` value and plots the mean top-1 accuracy with
    baseline shown as a horizontal reference line.

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
        Line colour.
    linestyle:
        Matplotlib linestyle + marker string.
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

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, means, linestyle, color=color, linewidth=2, markersize=6,
            label=label)
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.2f}%)")
    ax.set_xlabel("Sigma threshold (k)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Accuracy vs outlier threshold — {results[0].site}")
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

    filtered.sort(key=lambda r: _site_sort_key(r.site_identifier))
    labels = [r.site_identifier for r in filtered]
    values = [r.pct_zeroed for r in filtered]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("% Zeroed")
    ax.set_title(f"Percentage zeroed at k={sigma_k}σ — {filtered[0].site}")
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
    """Overlay two accuracy-vs-threshold curves on one plot.

    Parameters
    ----------
    results_a:
        First set of ablation results.
    results_b:
        Second set of ablation results.
    output_path:
        Destination PNG path.
    label_a:
        Legend label for the first curve.
    label_b:
        Legend label for the second curve.
    title:
        Override plot title.  If None, derived from ``results_a[0].site``.
    """
    if not results_a or not results_b:
        logger.warning("Empty results; skipping accuracy comparison plot.")
        return

    baseline = results_a[0].baseline_top1

    def _group(results: list[AblationResult]) -> tuple[list[float], list[float]]:
        by_k: dict[float, list[float]] = {}
        for r in results:
            by_k.setdefault(r.sigma_threshold, []).append(r.top1_accuracy)
        ks = sorted(by_k.keys())
        return ks, [np.mean(by_k[k]) for k in ks]

    ks_a, means_a = _group(results_a)
    ks_b, means_b = _group(results_b)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks_a, means_a, "o-", color="coral", linewidth=2, markersize=6,
            label=label_a)
    ax.plot(ks_b, means_b, "s--", color="steelblue", linewidth=2, markersize=6,
            label=label_b)
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.2f}%)")
    ax.set_xlabel("Sigma threshold (k)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(title or f"Accuracy vs outlier threshold — {results_a[0].site}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

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

    for mode, results in mode_results.items():
        filtered = [r for r in results if r.sigma_threshold == sigma_k]
        if not filtered:
            continue
        if baseline is None:
            baseline = filtered[0].baseline_top1
        mode_names.append(mode)
        accuracies.append(np.mean([r.top1_accuracy for r in filtered]))

    if not mode_names:
        logger.warning("No results at k=%.1f for any mode.", sigma_k)
        return

    fig, ax = plt.subplots(figsize=(max(6, len(mode_names) * 1.5), 5))
    x = np.arange(len(mode_names))
    colors = ["steelblue", "coral", "seagreen"][:len(mode_names)]
    ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                   label=f"Baseline ({baseline:.2f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, fontsize=9)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Ablation mode comparison at k={sigma_k}σ")
    if baseline is not None:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Set ylim to show the baseline clearly.
    min_acc = min(accuracies) if accuracies else 0
    ax.set_ylim(min_acc - 2, (baseline or 100) + 2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ablation mode comparison to %s", output_path)


# ---------------------------------------------------------------------------
# Entropy delta heatmap
# ---------------------------------------------------------------------------


def plot_entropy_delta_heatmap(
    entropy_deltas: dict[str, dict[str, float]],
    output_path: Path,
    delta_key: str = "mean_cls_delta",
    title: str | None = None,
) -> None:
    """Save a heatmap of per-head (or per-site) entropy deltas after ablation.

    Parameters
    ----------
    entropy_deltas:
        Mapping from site_identifier to a dict with keys like
        ``"mean_cls_delta"`` and ``"mean_patch_delta"``.
    output_path:
        Destination PNG path.
    delta_key:
        Which delta key to visualise.
    title:
        Override plot title.
    """
    scalar: dict[str, float] = {}
    for sid, deltas in entropy_deltas.items():
        if delta_key in deltas:
            scalar[sid] = deltas[delta_key]

    if not scalar:
        logger.warning("No entropy delta data for key=%s; skipping.", delta_key)
        return

    sorted_ids = sorted(scalar.keys(), key=_site_sort_key)
    values = np.array([scalar[sid] for sid in sorted_ids])

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_ids) * 0.3)))
    y_pos = range(len(sorted_ids))
    colors = ["steelblue" if v >= 0 else "coral" for v in values]
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sid.replace("/pre_softmax", "") for sid in sorted_ids], fontsize=7)
    ax.set_xlabel("Δ Entropy (nats)")
    ax.set_title(title or f"Attention entropy delta — {delta_key}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved entropy delta plot to %s", output_path)


# ---------------------------------------------------------------------------
# Bootstrap CI on accuracy delta (migrated from analyze_ablation_results.py)
# ---------------------------------------------------------------------------


def plot_bootstrap_ci_delta(
    ci_results: dict[float, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot bootstrap 95% CI for per-channel minus global accuracy delta.

    Parameters
    ----------
    ci_results:
        Mapping from sigma_k to ``{"delta_point_estimate", "delta_ci_low_pct",
        "delta_ci_high_pct"}``, as produced by the bootstrap computation in
        ``scripts/analyze_ablation_results.py``.
    output_path:
        Destination PNG path.
    """
    if not ci_results:
        logger.warning("Empty CI results; skipping bootstrap CI plot.")
        return

    ks = sorted(ci_results.keys())
    deltas = [ci_results[k]["delta_point_estimate"] for k in ks]
    ci_lows = [ci_results[k]["delta_ci_low_pct"] for k in ks]
    ci_highs = [ci_results[k]["delta_ci_high_pct"] for k in ks]
    errors_low = [d - l for d, l in zip(deltas, ci_lows)]
    errors_high = [h - d for d, h in zip(deltas, ci_highs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        ks, deltas,
        yerr=[errors_low, errors_high],
        fmt="o-", capsize=5, color="steelblue", markersize=8,
        label="Per-channel − Global (95% bootstrap CI)",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sigma threshold k")
    ax.set_ylabel("Δ Top-1 Accuracy (%)")
    ax.set_title("Per-Channel vs Global Ablation Delta\n(95% bootstrap CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bootstrap CI plot to %s", output_path)


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
    """Plot effective channels preserved: two conditions per block.

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

    sids = sorted(channels.keys(), key=_site_sort_key)
    labels = [sid.split("/")[0] for sid in sids]
    x = np.arange(len(sids))
    width = 0.35

    g_ch = [channels[sid].get("global_channels", channels[sid].get("channels_a", 0)) for sid in sids]
    p_ch = [channels[sid].get("pc_channels", channels[sid].get("channels_b", 0)) for sid in sids]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, g_ch, width, label="Global", color="coral",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, p_ch, width, label="Per-channel", color="steelblue",
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Encoder Block")
    ax.set_ylabel(f"Effective Channels (of {channel_count})")
    ax.set_title(f"Effective {dim_label} Channels Preserved at k={sigma_k}\n(Higher = more channels active)")
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
    """Plot accuracy degradation per 1% sparsity for two conditions.

    Parameters
    ----------
    deg_results:
        Mapping from sigma_k to ``{"global_degradation_per_pct",
        "pc_degradation_per_pct", "efficiency_ratio", ...}``.
    output_path:
        Destination PNG path.
    """
    if not deg_results:
        logger.warning("Empty degradation results; skipping efficiency plot.")
        return

    ks = sorted(deg_results.keys())
    g_deg = [deg_results[k].get("global_degradation_per_pct",
                                 deg_results[k].get("degradation_a_per_pct", 0))
             for k in ks]
    p_deg = [deg_results[k].get("pc_degradation_per_pct",
                                 deg_results[k].get("degradation_b_per_pct", 0))
             for k in ks]
    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, g_deg, width, label="Global", color="coral",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, p_deg, width, label="Per-channel", color="steelblue",
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Sigma threshold k")
    ax.set_ylabel("Accuracy loss per 1% sparsity (pp / %)")
    ax.set_title("Accuracy Degradation Efficiency\n(Lower = more accuracy preserved per unit sparsity)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved degradation efficiency plot to %s", output_path)
