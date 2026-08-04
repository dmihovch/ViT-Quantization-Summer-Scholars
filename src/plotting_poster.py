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
import re
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.ablation import AblationResult  # noqa: E402

logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"blocks\.(\d+)")


# ---------------------------------------------------------------------------
# Shared poster styling
# ---------------------------------------------------------------------------

# Paul Tol-inspired qualitative palette (colourblind-safe, print-friendly).
_PALETTE: dict[str, str] = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "gray": "#BBBBBB",
    "dark": "#222222",
    "coral": "#CC3311",
    "teal": "#009988",
}

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

    This is the hero figure: it shows the raw activation distribution
    overlaid with the global 3σ threshold (dashed red), the range of
    per-channel thresholds (orange band), and the zeroed regions (light
    red shading).  A single glance communicates: global thresholds
    over-zero activations that fall within their own channel's normal
    range.

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
        color=_PALETTE["gray"], alpha=0.6, density=False,
        edgecolor="white", linewidth=0.2, zorder=1,
    )

    # --- Global threshold lines ---
    lo_global = global_mean - sigma_k * global_std
    hi_global = global_mean + sigma_k * global_std
    ax.axvline(lo_global, color=_PALETTE["red"], linestyle="--", linewidth=2.0,
               zorder=4, label=f"Global ±{sigma_k}σ = [{lo_global:.1f}, {hi_global:.1f}]")
    ax.axvline(hi_global, color=_PALETTE["red"], linestyle="--", linewidth=2.0, zorder=4)

    # --- Per-channel threshold band ---
    if per_channel_stds is not None and per_channel_means is not None and len(per_channel_stds) > 0:
        pc_stds = np.asarray(per_channel_stds)
        pc_means = np.asarray(per_channel_means)
        pc_lo = pc_means - sigma_k * pc_stds
        pc_hi = pc_means + sigma_k * pc_stds
        # Shade the region between min per-channel lower and max per-channel upper.
        band_lo = float(np.min(pc_lo))
        band_hi = float(np.max(pc_hi))
        ax.axvspan(band_lo, band_hi, alpha=0.08, color=_PALETTE["yellow"], zorder=0)
        ax.axvline(band_lo, color=_PALETTE["yellow"], linestyle="-.", linewidth=1.5,
                   zorder=3, label=f"Per-channel ±{sigma_k}σ range [{band_lo:.1f}, {band_hi:.1f}]")
        ax.axvline(band_hi, color=_PALETTE["yellow"], linestyle="-.", linewidth=1.5, zorder=3)

    # --- Zeroed regions ---
    if xlim is None:
        xlim = (float(np.percentile(flat, 0.1)), float(np.percentile(flat, 99.9)))
    ax.set_xlim(xlim)
    ymax = ax.get_ylim()[1]
    ax.fill_between([xlim[0], lo_global], 0, ymax, color=_PALETTE["red"], alpha=0.04, zorder=0)
    ax.fill_between([hi_global, xlim[1]], 0, ymax, color=_PALETTE["red"], alpha=0.04, zorder=0)

    # --- Mean line ---
    ax.axvline(global_mean, color=_PALETTE["dark"], linestyle="-", linewidth=1.0,
               zorder=5, label=f"μ = {global_mean:.2f}")

    # --- Labels ---
    ax.set_xlabel("Activation value", fontsize=16, color=_PALETTE["dark"])
    ax.set_ylabel("Count", fontsize=16, color=_PALETTE["dark"])
    ax.set_title(layer_name, fontsize=18, color=_PALETTE["dark"], fontweight="bold")
    ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#DDDDDD")

    # Add annotation: what fraction lives in the zeroed regions.
    frac_zeroed = float(np.mean((flat < lo_global) | (flat > hi_global)))
    ax.annotate(
        f"{frac_zeroed * 100:.1f}% zeroed\nby global ±{sigma_k}σ",
        xy=(lo_global, ymax * 0.7),
        fontsize=11, color=_PALETTE["red"],
        ha="right", va="top",
    )

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
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
) -> None:
    """Plot a small-multiples grid of outlier fractions across all sites.

    Renders a 12×6 grid of coloured tiles where each tile's intensity
    encodes the outlier fraction at that (block, site) pair.  This
    communicates "where are the outliers?" in a single glance — no
    colorbar cross-referencing needed at poster distance if the colour
    saturation is strong enough.

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

    # Build matrix: blocks 0..11 × sites.
    matrix = np.full((12, len(site_order)), np.nan)
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
        if 0 <= blk < 12:
            matrix[blk, site_order.index(site)] = fracs[sigma_key]

    fig, axes = plt.subplots(12, len(site_order),
                              figsize=(len(site_order) * 1.6, 12 * 0.45),
                              facecolor="white")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color=_PALETTE["dark"], y=0.98)

    vmax = float(np.nanmax(matrix)) if np.any(~np.isnan(matrix)) else 0.01
    vmin = 0.0

    for row in range(12):
        for col in range(len(site_order)):
            ax = axes[row, col]
            val = matrix[row, col]

            if np.isnan(val):
                ax.set_facecolor("#F0F0F0")
            else:
                # Map fraction to colour intensity.
                intensity = min(val / max(vmax, 1e-8), 1.0)
                ax.set_facecolor(mcolors.to_rgba(_PALETTE["red"], alpha=0.15 + 0.85 * intensity))

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
                        color=_PALETTE["dark"] if val < 0.5 * vmax else "white",
                        transform=ax.transAxes)

    # Row labels.
    for row in range(12):
        axes[row, 0].set_ylabel(f"Blk {row}", fontsize=8, color=_PALETTE["dark"],
                                rotation=0, ha="right", va="center",
                                labelpad=15)

    # Column labels.
    for col, site in enumerate(site_order):
        axes[0, col].set_title(site.replace("_", " "), fontsize=7,
                               color=_PALETTE["dark"], pad=3)

    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved outlier site grid to %s", output_path)


# ===========================================================================
# 3. Connected scatter — accuracy vs %-zeroed
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
    """Plot accuracy against fraction zeroed as a connected scatter.

    Each point is (mean %-zeroed across layers, top-1 accuracy).
    Points are connected in k-order and colour-coded by condition.
    The per-channel curve sits to the right of the global curve —
    same accuracy at higher sparsity — visually communicating
    efficiency without requiring the viewer to cross-reference axes.

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
    def _extract(results: list[AblationResult]) -> dict[float, tuple[float, float]]:
        """Extract (mean_pct_zeroed, accuracy) per k for non-random results."""
        by_k: dict[float, tuple[list[float], list[float]]] = {}
        for r in results:
            if r.is_random:
                continue
            k = r.sigma_threshold
            by_k.setdefault(k, ([], []))
            by_k[k][0].append(r.top1_accuracy)
            by_k[k][1].append(r.pct_zeroed)
        out: dict[float, tuple[float, float]] = {}
        for k in sorted(by_k.keys()):
            accs, pcts = by_k[k]
            out[k] = (float(np.mean(accs)), float(np.mean(pcts)))
        return out

    pts_a = _extract(results_a)
    pts_b = _extract(results_b)
    baseline = results_a[0].baseline_top1 if results_a else 0.0

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    _poster_style(ax, fontsize=16)

    for pts, color, label, marker in [
        (pts_a, _PALETTE["coral"], label_a, "o"),
        (pts_b, _PALETTE["teal"], label_b, "s"),
    ]:
        ks = sorted(pts.keys())
        xs = [pts[k][1] for k in ks]
        ys = [pts[k][0] for k in ks]
        ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.5, zorder=2)
        ax.scatter(xs, ys, c=color, s=120, marker=marker, zorder=3,
                   edgecolors="white", linewidth=1, label=label)
        for k, x, y in zip(ks, xs, ys):
            ax.annotate(f"k={k:.0f}", (x, y),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=10, color=color, alpha=0.9)

    # Baseline horizontal.
    ax.axhline(baseline, color=_PALETTE["gray"], linestyle="--", linewidth=1.5,
               zorder=1, label=f"Baseline {baseline:.1f}%")

    ax.set_xlabel("% Elements Zeroed", fontsize=16, color=_PALETTE["dark"])
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=16, color=_PALETTE["dark"])
    ax.set_title(title or f"Accuracy vs Sparsity — {results_a[0].site}",
                 fontsize=18, color=_PALETTE["dark"], fontweight="bold")
    ax.legend(fontsize=13, loc="lower left", frameon=True,
              facecolor="white", edgecolor="#DDDDDD")

    # Invert x-axis: moving right = more zeroed.
    ax.invert_xaxis()

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved accuracy vs sparsity scatter to %s", output_path)


# ===========================================================================
# 4. Ridgeline — per-channel σ distributions
# ===========================================================================


def plot_per_channel_sigma_ridgeline(
    per_channel_stds: dict[str, list[float]],
    output_path: Path,
    *,
    title: str | None = None,
    colourmap_name: str = "viridis",
    overlap: float = 0.6,
) -> None:
    """Plot ridgeline (joyplot) of per-channel σ distributions across blocks.

    Each block is a filled density curve, vertically offset and coloured
    by block depth.  Late blocks show bimodal, wide distributions; early
    blocks are tight Gaussian-like clusters.  Much more visceral than a
    heatmap.

    Parameters
    ----------
    per_channel_stds:
        Mapping from site identifier to per-channel σ list.
    output_path:
        Destination PNG path.
    title:
        Override plot title.
    colourmap_name:
        Matplotlib sequential colormap name.
    overlap:
        Vertical overlap between curves (0=no overlap, 1=full overlap).
    """
    # Filter to pre_gelu sites only and sort by block index.
    gelu_stds: list[tuple[int, np.ndarray]] = []
    for sid, stds in per_channel_stds.items():
        if "/pre_gelu" not in sid:
            continue
        m = _BLOCK_RE.search(sid)
        if not m:
            continue
        blk = int(m.group(1))
        gelu_stds.append((blk, np.asarray(stds, dtype=np.float32)))
    gelu_stds.sort(key=lambda x: x[0])

    if not gelu_stds:
        logger.warning("No pre_gelu per-channel σ data found; skipping ridgeline.")
        return

    n = len(gelu_stds)
    fig, ax = plt.subplots(figsize=(12, n * 0.65), facecolor="white")
    _poster_style(ax, fontsize=14)

    cmap = plt.get_cmap(colourmap_name)
    # Global x-range (shared across all blocks).
    all_vals = np.concatenate([arr for _, arr in gelu_stds])
    x_min = 0.0
    x_max = float(np.percentile(all_vals, 99))

    # Build KDE for each block.
    xs = np.linspace(x_min, x_max, 500)

    for idx, (blk, arr) in enumerate(gelu_stds):
        # Simple Gaussian KDE approximation using histogram smoothing.
        hist, edges = np.histogram(arr, bins=100, range=(x_min, x_max), density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        from scipy.ndimage import gaussian_filter1d
        density = gaussian_filter1d(hist.astype(np.float64), sigma=2.0)
        # Interpolate to common x grid.
        density_interp = np.interp(xs, centres, density)
        density_interp = density_interp / (density_interp.max() + 1e-10)  # normalise peak=1

        # Colour by block depth.
        color = cmap(idx / max(n - 1, 1))

        # Vertical offset.
        baseline = (n - 1 - idx) * overlap
        ax.fill_between(xs, baseline, baseline + density_interp,
                        color=color, alpha=0.85, zorder=idx, linewidth=0)

        # Block label.
        ax.text(x_max * 1.01, baseline + 0.1, f"Blk {blk}",
                fontsize=9, color=color, va="bottom", ha="left",
                fontweight="bold")

    # Per-channel σ mean reference line.
    mean_sigma = float(np.mean(all_vals))
    ax.axvline(mean_sigma, color=_PALETTE["gray"], linestyle=":", linewidth=1,
               zorder=n + 1, alpha=0.6)

    ax.set_xlim(x_min, x_max * 1.15)
    ax.set_ylim(-0.3, n * overlap + 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Per-channel σ", fontsize=14, color=_PALETTE["dark"])
    ax.set_title(title or "Per-Channel σ Distribution — pre-GELU",
                 fontsize=16, color=_PALETTE["dark"], fontweight="bold")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved per-channel σ ridgeline to %s", output_path)


# ===========================================================================
# 5. Streamgraph — attention entropy collapse
# ===========================================================================


def plot_attention_entropy_streamgraph(
    cls_entropies: dict[str, list[float]],
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot a streamgraph of CLS attention entropy across blocks.

    Each attention head is a coloured band.  The stream narrows in later
    blocks as entropy collapses — the attention sink phenomenon
    (Zhai et al. 2023, ICML).  At poster distance, the narrowing stream
    communicates the collapse more viscerally than a heatmap.

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
    # Build (num_blocks, num_heads) matrix.
    entries: list[tuple[int, list[float]]] = []
    for sid, ent in cls_entropies.items():
        m = _BLOCK_RE.search(sid)
        if not m:
            continue
        blk = int(m.group(1))
        if blk < 0 or blk > 11:
            continue
        entries.append((blk, ent))

    if not entries:
        logger.warning("No CLS entropy data; skipping streamgraph.")
        return

    entries.sort(key=lambda x: x[0])
    num_blocks = max(b for b, _ in entries) + 1
    num_heads = len(entries[0][1]) if entries else 1

    # Build (heads, blocks) matrix.
    matrix = np.zeros((num_heads, num_blocks))
    for blk, ent in entries:
        for h in range(min(num_heads, len(ent))):
            matrix[h, blk] = ent[h]

    # Baseline: shift each head so that all curves are positive and stack.
    # Use a gallery-style baseline (wiggle).
    baseline = np.zeros(num_blocks)
    stacked = np.zeros((num_heads + 1, num_blocks))
    for h in range(num_heads):
        stacked[h + 1] = stacked[h] + matrix[h]
    centre = 0.5 * (stacked[0] + stacked[-1])
    centred = matrix - centre[np.newaxis, :]  # centre each column at 0

    # Head colour palette.
    head_colors = [plt.get_cmap("tab20")(i % 20) for i in range(num_heads)]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    _poster_style(ax, fontsize=14)

    x = np.arange(num_blocks)
    cumsum = np.zeros(num_blocks)
    for h in range(num_heads):
        vals = centred[h]
        ax.fill_between(x, cumsum, cumsum + vals,
                        color=head_colors[h], alpha=0.85, linewidth=0,
                        zorder=num_heads - h)
        cumsum += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in range(num_blocks)], fontsize=11)
    ax.set_xlabel("Encoder Block", fontsize=14, color=_PALETTE["dark"])
    ax.set_ylabel("CLS Entropy (nats)", fontsize=14, color=_PALETTE["dark"])
    ax.set_title(title or "CLS Attention Entropy — Streamgraph",
                 fontsize=16, color=_PALETTE["dark"], fontweight="bold")
    ax.set_yticks([])

    # Annotate the collapse.
    mid_entropy = float(np.mean(matrix[:, :4]))
    late_entropy = float(np.mean(matrix[:, -4:]))
    ax.annotate(
        f"Early blocks\nmean: {mid_entropy:.2f} nats",
        xy=(2, 0), fontsize=11, color=_PALETTE["dark"],
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
    )
    ax.annotate(
        f"Late blocks\nmean: {late_entropy:.2f} nats",
        xy=(9, 0), fontsize=11, color=_PALETTE["dark"],
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
    )

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved attention entropy streamgraph to %s", output_path)


# ===========================================================================
# 6. Waterfall chart — ablation mode decomposition
# ===========================================================================


def plot_ablation_waterfall(
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
    """Plot a waterfall chart decomposing the per-channel accuracy benefit.

    Shows how the accuracy degrades from baseline through each ablation
    variant, revealing whether the per-channel benefit comes from mean
    correction, variance correction, or both.

    Parameters
    ----------
    baseline:
        Unablated top-1 accuracy (%).
    global_acc:
        Global-σ ablation accuracy at k.
    mean_only_acc:
        ``mean_only`` ablation accuracy at k (per-channel μ_c, global σ).
    var_only_acc:
        ``var_only`` ablation accuracy at k (global μ, per-channel σ_c).
    outlier_acc:
        Full per-channel outlier ablation accuracy at k.
    output_path:
        Destination PNG path.
    title:
        Override plot title.
    sigma_k:
        Sigma threshold for the annotation.
    """
    labels = ["Baseline", "Global\n±kσ",
              "+ Per-ch.\nμ_c", "+ Per-ch.\nσ_c",
              "+ Interaction",
              "Per-channel\n(total)"]
    values = [baseline,
              global_acc - baseline,
              mean_only_acc - global_acc,
              var_only_acc - mean_only_acc,
              outlier_acc - var_only_acc,
              0.0]  # final is sum of increments

    # Compute running total for bar positions.
    running = [baseline]
    for v in values[1:-1]:
        running.append(running[-1] + v)
    running.append(outlier_acc)

    # Bar bottoms and heights.
    bottoms = [running[0]] + [min(running[i], running[i + 1]) for i in range(len(running) - 1)]
    heights = [0] + [values[i] for i in range(1, len(values) - 1)] + [0]
    # The final "total" bar should span from 0 to outlier_acc (not from the running sum).
    bottoms[-1] = 0
    heights[-1] = outlier_acc

    colors = [_PALETTE["gray"],
              _PALETTE["red"],
              _PALETTE["yellow"],
              _PALETTE["blue"],
              _PALETTE["green"],
              _PALETTE["teal"]]
    color_labels = ["Starting\nvalue", "Change",
                    "Change", "Change",
                    "Change", "Final\nvalue"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _poster_style(ax, fontsize=15)

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, bottom=bottoms, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.55, zorder=3)

    # Value annotations on bars.
    for i in range(len(labels)):
        val = running[i]
        bar_top = bottoms[i] + heights[i]
        mid = (bottoms[i] + bar_top) / 2
        ax.text(i, mid + 1.5, f"{val:.2f}%",
                ha="center", va="center", fontsize=14,
                fontweight="bold", color="white" if i > 0 else _PALETTE["dark"])

    # Delta annotations.
    deltas = [0,
              global_acc - baseline,
              mean_only_acc - global_acc,
              var_only_acc - mean_only_acc,
              outlier_acc - var_only_acc,
              outlier_acc - baseline]
    for i in range(1, len(labels)):
        d = deltas[i]
        if abs(d) > 0.01:
            sign = "+" if d > 0 else ""
            ax.annotate(f"{sign}{d:.2f} pp",
                        xy=(i, running[i]), xytext=(i, running[i] + 3.5 * np.sign(d)),
                        fontsize=13, fontweight="bold",
                        color=_PALETTE["red"] if d < 0 else _PALETTE["teal"],
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, color=_PALETTE["dark"])
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=15, color=_PALETTE["dark"])
    ax.set_title(title or f"Decomposing the Per-Channel Benefit at k={sigma_k}σ",
                 fontsize=17, color=_PALETTE["dark"], fontweight="bold")
    ax.set_ylim(0, baseline + 10)

    # Legend for bar colors.
    legend_patches = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(colors, color_labels)]
    ax.legend(handles=legend_patches, fontsize=11, loc="upper right",
              frameon=True, facecolor="white", edgecolor="#DDDDDD")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved ablation waterfall to %s", output_path)


# ===========================================================================
# 7. Hinton diagram — per-channel mean
# ===========================================================================


def plot_per_channel_mean_hinton(
    per_channel_means: dict[str, list[float]],
    block_idx: int,
    output_path: Path,
    *,
    max_squares_per_row: int = 40,
    title: str | None = None,
) -> None:
    """Plot a Hinton diagram of per-channel means for a single block.

    Each channel is a square whose area encodes |μ| and colour encodes sign
    (blue = positive, red = negative).  Ideal for showing the asymmetric
    mean structure of block 10 pre-GELU where μ ranges −71 to +26.

    Parameters
    ----------
    per_channel_means:
        Mapping from site_identifier to per-channel mean list.
    block_idx:
        Which encoder block to visualise (0-based).
    output_path:
        Destination PNG path.
    max_squares_per_row:
        Number of squares per row in the grid layout.
    title:
        Override plot title.
    """
    site_id = f"blocks.{block_idx}/pre_gelu"
    if site_id not in per_channel_means:
        logger.warning("No per-channel mean data for %s; skipping Hinton.", site_id)
        return

    means = np.asarray(per_channel_means[site_id], dtype=np.float64)
    D = len(means)

    # Grid layout.
    n_cols = max_squares_per_row
    n_rows = int(np.ceil(D / n_cols))

    fig, ax = plt.subplots(figsize=(n_cols * 0.33, n_rows * 0.33 + 1),
                           facecolor="white")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.axis("off")

    max_abs = float(max(np.max(np.abs(means)), 1e-10))

    for i in range(D):
        row = n_rows - 1 - (i // n_cols)
        col = i % n_cols
        val = means[i]

        # Square size proportional to |μ|.
        size = 0.85 * np.sqrt(abs(val) / max_abs)
        half = size / 2
        color = _PALETTE["blue"] if val > 0 else _PALETTE["red"]
        alpha = 0.3 + 0.7 * (abs(val) / max_abs)

        rect = mpatches.Rectangle(
            (col + 0.5 - half, row + 0.5 - half),
            size, size,
            facecolor=color, edgecolor="none", alpha=alpha,
        )
        ax.add_patch(rect)

    ax.set_title(title or f"Per-Channel μ — Block {block_idx} pre-GELU\n"
                 f"Blue = +μ, Red = −μ, area ∝ |μ|, max |μ| = {max_abs:.1f}",
                 fontsize=16, color=_PALETTE["dark"], fontweight="bold", pad=10)

    # Legend.
    legend_patches = [
        mpatches.Patch(color=_PALETTE["blue"], alpha=0.6, label="μ > 0"),
        mpatches.Patch(color=_PALETTE["red"], alpha=0.6, label="μ < 0"),
    ]
    ax.legend(handles=legend_patches, fontsize=12, loc="lower right",
              frameon=True, facecolor="white", edgecolor="#DDDDDD")

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved per-channel mean Hinton diagram to %s", output_path)
