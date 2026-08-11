"""Poster-quality figure generation for the ViT Quantization project.

Unlike :mod:`src.plotting` (the workhorse module designed for rapid research
iteration), this module produces visually polished figures suitable for
posters, presentations, and publication-ready outputs.  Every function
targets a specific narrative insight and uses deliberate visual encoding
rather than matplotlib defaults.

Conventions (all functions)
---------------------------
* White background, thin spines, no chartjunk.
* Font sizes ≥ 14 pt for readability at poster distance.
* Direct annotation over legends where possible.
* Custom colour palettes: Viridis for sequential, Paul Tol-inspired for
  qualitative, coolwarm for diverging.
* No gray grids unless structurally meaningful.
* Consistent aspect ratios for side-by-side poster layout.

Plot index
----------
1. ``plot_activation_distribution_overlay``  — hero figure: original dist +
   global thresholds + per-channel threshold band + zeroed regions.
2. ``plot_outlier_site_grid``               — small-multiples grid: outlier
   fraction for all 73 sites at a glance.
3. ``plot_accuracy_vs_sparsity_scatter``    — connected scatter: accuracy vs
   %-zeroed, colour-coded by condition.
4. ``plot_per_channel_sigma_ridgeline``     — ridgeline plot of per-channel σ
   distributions across blocks.
5. ``plot_attention_entropy_streamgraph``   — streamgraph of CLS attention
   entropy collapse across blocks.
6. ``plot_ablation_waterfall``              — waterfall chart decomposing the
   per-channel accuracy benefit.
7. ``plot_per_channel_mean_hinton``         — Hinton diagram of per-channel
   means for a single block.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.ablation import AblationResult  # noqa: E402
from src.plotting_utils import (
    LABELS,
    POSTER_PALETTE,
    extract_block_index,
    format_site_label,
    site_sort_key,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases.
_PALETTE = POSTER_PALETTE
_site_sort_key = site_sort_key

_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "coolwarm_poster", ["#3B6FB6", "#EEEEEE", "#CC3311"]
)


def _poster_style(ax: plt.Axes, fontsize: int = 14) -> None:
    """Apply poster styling to an Axes: white bg, thin spines, no grid."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#CCCCCC")
    ax.tick_params(labelsize=fontsize - 2, colors=_PALETTE["dark"])
    ax.grid(False)


def _site_sort_key(site_id: str) -> tuple[int, int]:
    if site_id.startswith("patch_embed"):
        return (0, 0)
    m = _BLOCK_RE.search(site_id)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


# ===========================================================================
# 1. Hero figure — activation distribution overlay
# ===========================================================================


def plot_activation_distribution_overlay(
    activations: np.ndarray,
    layer_name: str,
    output_path: Path,
    *,
    global_mean: float,
    global_std: float,
    per_channel_stds: list[float] | None = None,
    per_channel_means: list[float] | None = None,
    sigma_k: float = 3.0,
    bins: int = 300,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Plot activation histogram with outlier threshold overlays.

    This is the hero figure. It shows the raw activation distribution
    overlaid with the global 3σ threshold (dashed red) and specific examples
    of per-channel thresholds. A single glance communicates how global thresholds
    can incorrectly clip activations that fall within a channel's normal range.

    Parameters
    ----------
    activations:
        Flat numpy array of activation values (all channels pooled).
    layer_name:
        Layer identifier for the title (e.g. ``"Block 10 pre-GELU"``).
    output_path:
        Destination PNG path.
    global_mean:
        Global mean μ of the activation distribution.
    global_std:
        Global standard deviation σ.
    per_channel_stds:
        Per-channel σ_c values.  If provided, the range of per-channel
        thresholds is drawn as an orange band.
    per_channel_means:
        Per-channel μ_c values.  Required with ``per_channel_stds``.
    sigma_k:
        Threshold multiplier (default 3.0).
    bins:
        Number of histogram bins.
    xlim:
        Override x-axis limits.  Auto-computed if None.
    """
    flat = activations.ravel()

    fig, ax = plt.subplots(figsize=(10, 6))
    _poster_style(ax, fontsize=16)

    # --- Histogram ---
    counts, edges, _patches = ax.hist(
        flat, bins=bins,
        color=_PALETTE["teal"], alpha=0.6, density=True,
        edgecolor="white", linewidth=0.2, zorder=1,
    )

    # --- Global threshold lines ---
    lo_global = global_mean - sigma_k * global_std
    hi_global = global_mean + sigma_k * global_std
    ax.axvline(lo_global, color=_PALETTE["coral"], linestyle="--", linewidth=2.0,
               zorder=4)
    ax.axvline(hi_global, color=_PALETTE["coral"], linestyle="--", linewidth=2.0, zorder=4,
               label=f"Global ±{sigma_k}σ [{lo_global:.1f}, {hi_global:.1f}]")

    # --- Per-channel exemplar threshold lines ---
    if per_channel_stds is not None and per_channel_means is not None and len(per_channel_stds) > 0:
        pc_stds = np.asarray(per_channel_stds)
        pc_means = np.asarray(per_channel_means)

        # Find exemplar channels
        min_std_idx = np.argmin(pc_stds)
        max_std_idx = np.argmax(pc_stds)

        # Thresholds for the least volatile channel
        lo_min_ch = pc_means[min_std_idx] - sigma_k * pc_stds[min_std_idx]
        hi_min_ch = pc_means[min_std_idx] + sigma_k * pc_stds[min_std_idx]
        ax.axvline(lo_min_ch, color=_PALETTE["blue"], linestyle=":", linewidth=2.0, zorder=3)
        ax.axvline(hi_min_ch, color=_PALETTE["blue"], linestyle=":", linewidth=2.0, zorder=3,
                   label=f"Ch. {min_std_idx} (min σ) bounds [{lo_min_ch:.1f}, {hi_min_ch:.1f}]")

        # Thresholds for the most volatile channel
        lo_max_ch = pc_means[max_std_idx] - sigma_k * pc_stds[max_std_idx]
        hi_max_ch = pc_means[max_std_idx] + sigma_k * pc_stds[max_std_idx]
        ax.axvline(lo_max_ch, color="#006400", linestyle="-.", linewidth=2.0, zorder=3) # Dark Green
        ax.axvline(hi_max_ch, color="#006400", linestyle="-.", linewidth=2.0, zorder=3,
                   label=f"Ch. {max_std_idx} (max σ) bounds [{lo_max_ch:.1f}, {hi_max_ch:.1f}]")

    # --- Zeroed regions ---
    # Determine x-limits dynamically to include all lines and data percentiles.
    x_values_to_include = [
        float(np.percentile(flat, 0.1)),
        float(np.percentile(flat, 99.9)),
        lo_global, hi_global,
    ]
    if 'lo_min_ch' in locals(): # Check if exemplar vars were created
        x_values_to_include.extend([lo_min_ch, hi_min_ch, lo_max_ch, hi_max_ch])
    
    x_min = min(x_values_to_include)
    x_max = max(x_values_to_include)
    padding = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - padding, x_max + padding)
    final_xlim = ax.get_xlim()

    ymax = ax.get_ylim()[1]
    ax.fill_between([final_xlim[0], lo_global], 0, ymax, color=_PALETTE["red"], alpha=0.04, zorder=0)
    ax.fill_between([hi_global, final_xlim[1]], 0, ymax, color=_PALETTE["red"], alpha=0.04, zorder=0)

    # --- Mean line ---
    ax.axvline(global_mean, color=_PALETTE["dark"], linestyle="-", linewidth=1.0,
               zorder=5, label=f"μ = {global_mean:.2f}")

    # --- Labels ---
    ax.set_xlabel("Activation value", fontsize=16, color=_PALETTE["dark"])
    ax.set_ylabel("Probability Density", fontsize=16, color=_PALETTE["dark"])
    ax.set_title(layer_name, fontsize=18, color=_PALETTE["dark"], fontweight="bold")
    ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#DDDDDD")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved activation distribution overlay to %s", output_path)


# ===========================================================================
# 2. Small-multiples grid — outlier fraction at a glance
# ===========================================================================


def plot_outlier_site_grid(
    outlier_fractions: dict[str, dict[str, float]],
    output_path: Path,
    *,
    sigma_key: str = "3.0_sigma",
    title: str | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    add_colorbar: bool = True,
) -> None:
    """Plot a small-multiples grid of outlier fractions across all sites.

    Renders a 12x6 grid of coloured tiles where each tile's intensity
    encodes the outlier fraction at that (block, site) pair. This
    communicates "where are the outliers?" in a single glance. The color
    saturation is strong enough to be clear without cross-referencing a
    colorbar, even at a poster-viewing distance.

    Parameters
    ----------
    outlier_fractions:
        Mapping from site_identifier to ``{sigma_key: fraction}`` dict.
    output_path:
        Destination PNG path.
    sigma_key:
        Which outlier fraction key to visualise, e.g. ``"3.0_sigma"``.
    title:
        Override suptitle.
    """
    # Canonical site order (same as profiling convention).
    site_order: list[str] = [
        "residual_stream", "post_layernorm_1", "pre_softmax",
        "post_softmax", "post_layernorm_2", "pre_gelu",
    ]

    # Determine number of blocks from data.
    block_indices: set[int] = set()
    for sid in outlier_fractions:
        bi = extract_block_index(sid)
        if bi is not None:
            block_indices.add(bi)
    num_blocks = max(block_indices) + 1 if block_indices else 12

    # Build matrix: blocks × sites.
    matrix = np.full((num_blocks, len(site_order)), np.nan)
    for sid, fracs in outlier_fractions.items():
        if sigma_key not in fracs:
            continue
        parts = sid.split("/", 1)
        if len(parts) != 2:
            continue
        block_str, site = parts[0], parts[1]
        if not block_str.startswith("blocks."):
            continue
        try:
            blk = int(block_str.split(".", 1)[1])
        except ValueError:
            continue
        if site not in site_order:
            continue
        if 0 <= blk < num_blocks:
            matrix[blk, site_order.index(site)] = fracs[sigma_key]

    fig, axes = plt.subplots(num_blocks, len(site_order),
                              figsize=(len(site_order) * 1.6, num_blocks * 0.45),
                              facecolor="white")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color=_PALETTE["dark"], y=0.98)

    _vmax = vmax if vmax is not None else (float(np.nanmax(matrix)) if np.any(~np.isnan(matrix)) else 0.01)
    _vmin = vmin

    for row in range(num_blocks):
        for col in range(len(site_order)):
            ax = axes[row, col]
            val = matrix[row, col]

            if np.isnan(val):
                ax.set_facecolor("#F0F0F0")
            else:
                # Map fraction to colour intensity clamped to [_vmin, _vmax].
                # Interpolate: light pink (#FFD0D0) → deep red (#8B0000).
                clamped = np.clip(val, _vmin, _vmax)
                intensity = (clamped - _vmin) / max(_vmax - _vmin, 1e-8)
                r = int(255 * (1.0 - intensity) + 139 * intensity)
                g = int(208 * (1.0 - intensity))
                b = int(208 * (1.0 - intensity))
                ax.set_facecolor((r / 255, g / 255, b / 255))

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.3)
                spine.set_color("#DDDDDD")

            if not np.isnan(val):
                ax.text(0.5, 0.5, f"{val * 100:.1f}%",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color=_PALETTE["dark"] if val < 0.5 * _vmax else "white",
                        transform=ax.transAxes)

    # Row labels.
    for row in range(num_blocks):
        axes[row, 0].set_ylabel(f"Blk {row}", fontsize=8, color=_PALETTE["dark"],
                                rotation=0, ha="right", va="center",
                                labelpad=15)

    # Column labels.
    for col, site in enumerate(site_order):
        axes[0, col].set_title(site.replace("_", " "), fontsize=7,
                               color=_PALETTE["dark"], pad=3)

    # Optional colorbar - placed to the right of the grid using a
    # manually-positioned axis so tight_layout cannot override it.
    if add_colorbar:
        sm = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=_vmin * 100, vmax=_vmax * 100),
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "outlier_red", ["#FFD0D0", "#8B0000"]
            ),
        )
        sm.set_array([])
        # Place colorbar axis manually: right edge of the last column + gap.
        # tight_layout will be called first, then we add the colorbar after.

    # Let tight_layout handle the grid first.
    fig.tight_layout(pad=1.2)

    if add_colorbar:
        # Get the bounding box of the last column's axes.
        last_col_axes = [axes[r, -1] for r in range(num_blocks)]
        # Compute the rightmost extent of the grid.
        right_edge = max(ax.get_position().x1 for ax in last_col_axes)
        bottom = axes[-1, 0].get_position().y0
        top = axes[0, 0].get_position().y1
        cbar_width = 0.015
        cbar_left = right_edge + 0.04  # ~one column-width gap
        cax = fig.add_axes([cbar_left, bottom, cbar_width, top - bottom])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Outlier fraction (%)", fontsize=10, color=_PALETTE["dark"])
        cbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved outlier site grid to %s", output_path)


# ===========================================================================
# 3. Grouped bar chart — accuracy at discrete sigma thresholds
# ===========================================================================


def plot_accuracy_vs_sparsity_scatter(
    results_a: list[AblationResult],
    results_b: list[AblationResult],
    output_path: Path,
    *,
    label_a: str = "Global",
    label_b: str = "Per-channel",
    sigma_ks: tuple[float, ...] = (3.0, 4.0, 6.0),
    title: str | None = None,
) -> None:
    """Plot accuracy as a grouped bar chart at discrete sigma thresholds.

    The x-axis has three discrete clusters ("3.0 σ", "4.0 σ", "6.0 σ"),
    # each with two bars side-by-side: one for Global, one for Per-channel.

    Parameters
    ----------
    results_a:
        First condition results (e.g. global).
    results_b:
        Second condition results (e.g. per-channel).
    output_path:
        Destination PNG path.
    label_a:
        Legend label for first condition.
    label_b:
        Legend label for second condition.
    sigma_ks:
        Sigma thresholds to plot.  Must match the data.
    title:
        Override plot title.
    """
    def _acc_at_k(
        results: list[AblationResult], k: float,
    ) -> tuple[float, float]:
        """Return (mean_accuracy, std_accuracy) at threshold k."""
        accs = [r.top1_accuracy for r in results
                if r.sigma_threshold == k and not r.is_random]
        if not accs:
            return (0.0, 0.0)
        return (float(np.mean(accs)), float(np.std(accs)))

    baseline = results_a[0].baseline_top1 if results_a else 0.0

    # Build per-k accuracy lookups.
    a_vals = {k: _acc_at_k(results_a, k) for k in sigma_ks}
    b_vals = {k: _acc_at_k(results_b, k) for k in sigma_ks}

    means_a = [a_vals[k][0] for k in sigma_ks]
    stds_a = [a_vals[k][1] for k in sigma_ks]
    means_b = [b_vals[k][0] for k in sigma_ks]
    stds_b = [b_vals[k][1] for k in sigma_ks]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    _poster_style(ax, fontsize=16)

    x = np.arange(len(sigma_ks))
    width = 0.30

    bars_a = ax.bar(
        x - width / 2, means_a, width,
        color=_PALETTE["coral"], edgecolor="white", linewidth=1.2,
        yerr=stds_a, capsize=6, label=label_a, zorder=3,
    )
    bars_b = ax.bar(
        x + width / 2, means_b, width,
        color=_PALETTE["teal"], edgecolor="white", linewidth=1.2,
        yerr=stds_b, capsize=6, label=label_b, zorder=3,
    )

    # Baseline reference line.
    ax.axhline(baseline, color=_PALETTE["gray"], linestyle="--", linewidth=2.0,
               zorder=1, alpha=0.8, label=f"Baseline {baseline:.1f}%")

    # Value annotations on bars.
    for bar_group, color in [(bars_a, _PALETTE["coral"]), (bars_b, _PALETTE["teal"])]:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 1.5,
                    f"{height:.2f}",
                    ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color=_PALETTE["dark"],
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:.0f} σ" for k in sigma_ks], fontsize=14)
    ax.set_xlabel(LABELS["sigma_threshold"], fontsize=16, color=_PALETTE["dark"])
    ax.set_ylabel(LABELS["accuracy"], fontsize=16, color=_PALETTE["dark"])
    ax.set_title(title or f"Accuracy by Threshold - {results_a[0].site}",
                 fontsize=18, color=_PALETTE["dark"], fontweight="bold")
    ax.legend(fontsize=13, loc="lower left", frameon=True,
              facecolor="white", edgecolor="#DDDDDD")

    # Consistent y-axis: start from 0, leave room for the baseline line.
    ax.set_ylim(0, baseline + 10)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved accuracy grouped bar chart to %s", output_path)


# ===========================================================================
# 4. Ridgeline — per-channel σ distributions
# ===========================================================================


def plot_per_channel_sigma_line(
    per_channel_stds: dict[str, list[float]],
    output_path: Path,
    *,
    title: str | None = None,
    cmap_name: str = "viridis",
) -> None:
    """Plot mean per-channel σ with ±1σ band across blocks.

    Each block is a single point: the mean of its per-channel σ_c values
    with a shaded band showing ±1 standard deviation of the σ_c
    distribution.  Uses a continuous sequential colormap tied to block
    depth so that early (narrow) and late (wide) blocks are visually
    comparable without log-axes.

    Parameters
    ----------
    per_channel_stds:
        Mapping from site identifier to per-channel σ list.
    output_path:
        Destination PNG path.
    title:
        Override plot title.
    cmap_name:
        Matplotlib sequential colormap name.
    """
    gelu_stds: list[tuple[int, np.ndarray]] = []
    for sid, stds in per_channel_stds.items():
        if "/pre_gelu" not in sid:
            continue
        bi = extract_block_index(sid)
        if bi is None:
            continue
        gelu_stds.append((bi, np.asarray(stds, dtype=np.float32)))
    gelu_stds.sort(key=lambda x: x[0])

    if not gelu_stds:
        logger.warning("No pre_gelu per-channel σ data found; skipping line chart.")
        return

    blocks = [b for b, _ in gelu_stds]
    means = [float(np.mean(arr)) for _, arr in gelu_stds]
    stds_of_stds = [float(np.std(arr)) for _, arr in gelu_stds]
    n = len(gelu_stds)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _poster_style(ax, fontsize=14)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    ax.fill_between(
        blocks,
        [m - s for m, s in zip(means, stds_of_stds)],
        [m + s for m, s in zip(means, stds_of_stds)],
        color=colors[0] if colors else _PALETTE["teal"],
        alpha=0.15,
        label="±1 std of per-channel σ",
        linewidth=0,
    )

    ax.plot(
        blocks, means, marker="o", markersize=8, linewidth=2.5,
        color=_PALETTE["teal"], zorder=4, label="Mean per-channel σ",
    )

    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks], fontsize=11)
    ax.set_xlabel(LABELS["block"], fontsize=14, color=_PALETTE["dark"])
    ax.set_ylabel("Per-channel standard deviation", fontsize=14, color=_PALETTE["dark"])
    ax.set_title(title or "Per-Channel Standard Deviation by Block Depth — pre-GELU",
                 fontsize=16, color=_PALETTE["dark"], fontweight="bold")
    ax.legend(fontsize=11, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#DDDDDD")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved per-channel σ line chart to %s", output_path)


# ===========================================================================
# 5. Streamgraph — attention entropy collapse
# ===========================================================================


def plot_attention_entropy_heatmap(
    cls_entropies: dict[str, list[float]],
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot CLS attention entropy as a 2D heatmap (heads × blocks).

    Each cell's colour intensity encodes entropy in nats for a specific
    (head, block) pair.  Unlike a stacked area chart, every head is
    independently readable — its baseline is always zero.

    Parameters
    ----------
    cls_entropies:
        Mapping from site_identifier (e.g. ``"blocks.3/post_softmax"``) to
        per-head CLS entropy list ``[H]``.
    output_path:
        Destination PNG path.
    title:
        Override plot title.
    """
    entries: list[tuple[int, list[float]]] = []
    for sid, ent in cls_entropies.items():
        bi = extract_block_index(sid)
        if bi is None:
            continue
        if bi < 0 or bi > 11:
            continue
        entries.append((bi, ent))

    if not entries:
        logger.warning("No CLS entropy data; skipping heatmap.")
        return

    entries.sort(key=lambda x: x[0])
    num_blocks = max(b for b, _ in entries) + 1
    num_heads = len(entries[0][1]) if entries else 1

    # Build (heads, blocks) matrix.
    matrix = np.zeros((num_heads, num_blocks))
    for blk, ent in entries:
        for h in range(min(num_heads, len(ent))):
            matrix[h, blk] = ent[h]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    _poster_style(ax, fontsize=14)

    im = ax.imshow(
        matrix, aspect="auto", origin="lower",
        cmap="viridis", interpolation="nearest",
    )

    ax.set_xticks(np.arange(num_blocks))
    ax.set_xticklabels([str(i) for i in range(num_blocks)], fontsize=11)
    ax.set_yticks(np.arange(num_heads))
    ax.set_yticklabels([f"Head {h}" for h in range(num_heads)], fontsize=10)
    ax.set_xlabel(LABELS["block"], fontsize=14, color=_PALETTE["dark"])
    ax.set_ylabel("Head", fontsize=14, color=_PALETTE["dark"])
    ax.set_title(title or "CLS Attention Entropy — Heatmap",
                 fontsize=16, color=_PALETTE["dark"], fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Entropy (nats)", fontsize=13, color=_PALETTE["dark"])
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved attention entropy heatmap to %s", output_path)


# ===========================================================================
# 6. Waterfall chart — ablation mode decomposition
# ===========================================================================


def plot_ablation_comparison(
    baseline: float,
    global_acc: float,
    mean_only_acc: float,
    var_only_acc: float,
    outlier_acc: float,
    output_path: Path,
    *,
    title: str | None = None,
    sigma_k: float = 3.0,
) -> None:
    """Plot a grouped bar chart comparing ablation conditions at threshold k.

    Shows baseline accuracy alongside four independent ablation conditions
    (global, mean_only, var_only, outlier) as grouped bars.  Unlike the
    previous waterfall design, this does not imply additivity between
    conditions — they are independent experiments.

    Parameters
    ----------
    baseline:
        Unablated top-1 accuracy (%).
    global_acc:
        Global-sigma ablation accuracy at k.
    mean_only_acc:
        ``mean_only`` ablation accuracy at k (per-channel mu_c, global sigma).
    var_only_acc:
        ``var_only`` ablation accuracy at k (global mu, per-channel sigma_c).
    outlier_acc:
        Full per-channel outlier ablation accuracy at k.
    output_path:
        Destination PNG path.
    title:
        Override plot title.
    sigma_k:
        Sigma threshold for the annotation.
    """
    labels = ["Baseline", "Global\nμ + global σ",
              "Per-ch. μ +\nglobal σ",
              "Global μ +\n per-ch. σ",
              "Per-ch. μ +\n per-ch. σ"]
    values = [baseline, global_acc, mean_only_acc, var_only_acc, outlier_acc]
    diffs = [0.0] + [v - baseline for v in values[1:]]

    colors = [
        _PALETTE["gray"],
        _PALETTE["coral"],
        _PALETTE["yellow"],
        _PALETTE["blue"],
        _PALETTE["teal"],
    ]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _poster_style(ax, fontsize=15)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.55, zorder=3)

    # Value annotations on bars.
    for i, (val, diff) in enumerate(zip(values, diffs)):
        ax.text(i, val + 1.5, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=_PALETTE["dark"])
        if i > 0:
            sign = "+" if diff > 0 else ""
            ax.annotate(f"{sign}{diff:.2f} pp vs baseline",
                        xy=(i, val), xytext=(i, val + 6),
                        fontsize=11,
                        color=_PALETTE["red"] if diff < 0 else _PALETTE["teal"],
                        ha="center")

    # Baseline reference line.
    ax.axhline(baseline, color=_PALETTE["gray"], linestyle="--", linewidth=1.5,
               zorder=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, color=_PALETTE["dark"])
    ax.set_ylabel(LABELS["accuracy"], fontsize=15, color=_PALETTE["dark"])
    ax.set_title(title or f"Ablation Condition Comparison at k={sigma_k}",
                 fontsize=17, color=_PALETTE["dark"], fontweight="bold")
    ax.set_ylim(0, baseline + 15)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved ablation comparison to %s", output_path)


# ===========================================================================
# 7. Hinton diagram — per-channel mean
# ===========================================================================


def plot_per_channel_mean_histogram(
    per_channel_means: dict[str, list[float]],
    block_idx: int,
    output_path: Path,
    *,
    bins: int = 80,
    title: str | None = None,
) -> None:
    """Plot a histogram of per-channel means for a single block.

    The x-axis is the per-channel mean value and the y-axis is
    frequency (channel count).  Outliers like a single -71.2 channel
    sit visibly isolated, and the overall distribution shape around
    zero is immediately apparent.

    Parameters
    ----------
    per_channel_means:
        Mapping from site_identifier to per-channel mean list.
    block_idx:
        Which encoder block to visualise (0-based).
    output_path:
        Destination PNG path.
    bins:
        Number of histogram bins.
    title:
        Override plot title.
    """
    site_id = f"blocks.{block_idx}/pre_gelu"
    if site_id not in per_channel_means:
        logger.warning("No per-channel mean data for %s; skipping histogram.", site_id)
        return

    means = np.asarray(per_channel_means[site_id], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _poster_style(ax, fontsize=14)

    ax.hist(means, bins=bins, color=_PALETTE["teal"], alpha=0.7,
            edgecolor="white", linewidth=0.3, zorder=2)

    mean_val = float(np.mean(means))
    median_val = float(np.median(means))
    ax.axvline(mean_val, color=_PALETTE["red"], linestyle="--", linewidth=1.5,
               zorder=4, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color=_PALETTE["dark"], linestyle=":", linewidth=1.5,
               zorder=4, label=f"Median = {median_val:.2f}")

    # Annotate the most extreme value if it's far out.
    min_val = float(np.min(means))
    max_val = float(np.max(means))
    ax.annotate(
        f"Min: {min_val:.1f}\nMax: {max_val:.1f}",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=11, color=_PALETTE["dark"], ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#DDDDDD", alpha=0.9),
    )

    ax.set_xlabel("Per-channel mean", fontsize=14, color=_PALETTE["dark"])
    ax.set_ylabel("Number of channels", fontsize=14, color=_PALETTE["dark"])
    ax.set_title(title or f"Per-Channel μ Distribution — Block {block_idx} pre-GELU",
                 fontsize=16, color=_PALETTE["dark"], fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", frameon=True,
              facecolor="white", edgecolor="#DDDDDD")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved per-channel mean histogram to %s", output_path)


# ===========================================================================
# 9. Effective gain vs σ_c scatter — architectural mechanism figure
# ===========================================================================


def plot_effective_gain_scatter(
    effective_gains: list[np.ndarray],
    pc_stds: list[np.ndarray],
    output_path: Path,
    *,
    block_indices: list[int] | None = None,
    title: str | None = None,
) -> None:
    """Plot effective per-channel gain |w_c x y|_2 against per-channel σ_c.

    One scatter panel per block, one point per MLP hidden channel.
    A regression line and annotated Pearson r quantify the
    architectural encoding of activation spread.
    """
    if block_indices is None:
        block_indices = [8, 9, 10]
    n = len(block_indices)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.2),
                              facecolor="white", squeeze=False)

    for i, (ax, bidx) in enumerate(zip(axes[0], block_indices)):
        gain = effective_gains[i]
        stds = pc_stds[i]
        r = float(np.corrcoef(gain, stds)[0, 1])

        ax.scatter(gain, stds, s=3, alpha=0.25,
                   color=_PALETTE["blue"], edgecolors="none", zorder=2)

        if len(stds) > 1 and np.std(gain) > 0:
            slope, intercept = np.polyfit(gain, stds, 1)
            xs = np.linspace(gain.min(), gain.max(), 100)
            ax.plot(xs, slope * xs + intercept, "-",
                    color=_PALETTE["red"], linewidth=1.5, zorder=3)

        ax.text(0.97, 0.05,
                f"x: [{gain.min():.1f}, {gain.max():.1f}]\n"
                f"y: [{stds.min():.1f}, {stds.max():.1f}]",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=_PALETTE["gray"])

        ax.text(0.97, 0.97, f"r = {r:+.3f}  (n={len(gain):,})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.85, edgecolor="#CCCCCC"))

        _poster_style(ax, fontsize=13)
        ax.set_xlabel(
            r"Effective per-channel gain  $\|\mathbf{w}_c \odot \gamma\|_2$",
            fontsize=12, color=_PALETTE["dark"])
        ax.set_ylabel(
            r"Per-channel $\sigma_c$",
            fontsize=12, color=_PALETTE["dark"])
        ax.set_title(f"Block {bidx}", fontsize=14, fontweight="bold",
                     color=_PALETTE["dark"])
        ax.grid(True, alpha=0.15, linewidth=0.3)

    fig.suptitle(
        title or "Learned Weights Encode Activation Spread: Late Blocks",
        fontsize=15, fontweight="bold", color=_PALETTE["dark"])

    fig.text(0.5, -0.04,
             r"$\|\mathbf{w}_c \odot \gamma\|_2$ = L2 norm of fc1.weight row "
             r"$c$ element-wise multiplied by LayerNorm scale $\gamma$",
             ha="center", fontsize=9, color=_PALETTE["gray"],
             transform=fig.transFigure)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved effective gain-σ scatter to %s", output_path)
