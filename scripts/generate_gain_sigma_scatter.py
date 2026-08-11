"""Generate effective-gain vs per-channel-σ scatter plots for the poster.

Computes ‖w_c ⊙ γ‖₂ — the effective per-channel gain — for every MLP hidden
channel and plots it against per-channel pre-GELU σ_c.

Produces:
  - ``gain_sigma_scatter_blocks_8_9_10.png`` — 3-panel grid (Blocks 8, 9, 10)
  - ``gain_sigma_scatter_block10.png``      — single Block 10
  - ``gain_sigma_scatter_all_blocks.png``   — 4×3 grid, all 12 blocks

Usage:
    python scripts/generate_gain_sigma_scatter.py \\
        --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json \\
        --output-dir all-plots2/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting_utils import POSTER_PALETTE


def _effective_gain(fc1_weight: np.ndarray, ln_gamma: np.ndarray) -> np.ndarray:
    """‖w_c ⊙ γ‖₂ per channel, shape (3072,)."""
    weighted = fc1_weight * ln_gamma[np.newaxis, :]
    return np.linalg.norm(weighted, axis=1)


def _scatter_one_block(ax, block_idx, gain, stds):
    """Draw one scatter panel: effective gain vs σ_c."""
    r = float(np.corrcoef(gain, stds)[0, 1])

    ax.scatter(gain, stds, s=3, alpha=0.25,
               color=POSTER_PALETTE["blue"], edgecolors="none", zorder=2)

    if len(stds) > 1 and np.std(gain) > 0:
        slope, intercept = np.polyfit(gain, stds, 1)
        xs = np.linspace(gain.min(), gain.max(), 100)
        ax.plot(xs, slope * xs + intercept, "-",
                color=POSTER_PALETTE["red"], linewidth=1.5, zorder=3)

    # Per-panel axis range annotation.
    ax.text(
        0.97, 0.05,
        f"x: [{gain.min():.1f}, {gain.max():.1f}]\ny: [{stds.min():.1f}, {stds.max():.1f}]",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.5, color=POSTER_PALETTE["gray"],
    )

    # Pearson r annotation.
    ax.text(
        0.97, 0.97,
        f"r = {r:+.3f}  (n={len(gain):,})",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   alpha=0.85, edgecolor="#CCCCCC"),
    )

    # Poster style.
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#CCCCCC")
    ax.tick_params(labelsize=10, colors=POSTER_PALETTE["dark"])
    ax.grid(True, alpha=0.15, linewidth=0.3)

    ax.set_xlabel(
        r"Effective per-channel gain  $\|\mathbf{w}_c \odot \gamma\|_2$",
        fontsize=12, color=POSTER_PALETTE["dark"],
    )
    ax.set_ylabel(
        r"Per-channel $\sigma_c$",
        fontsize=12, color=POSTER_PALETTE["dark"],
    )
    ax.set_title(f"Block {block_idx}", fontsize=14, fontweight="bold",
                 color=POSTER_PALETTE["dark"])


# -----------------------------------------------------------------
# Variants
# -----------------------------------------------------------------

def _make_three_panel(data, out_path):
    """Blocks 8, 9, 10 side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2),
                              facecolor="white", constrained_layout=True)
    for ax, bidx in zip(axes, [8, 9, 10]):
        blk = data[bidx]
        _scatter_one_block(ax, bidx, blk["gain"], blk["std"])
    fig.suptitle("Learned Weights Encode Activation Spread: Late Blocks",
                 fontsize=15, fontweight="bold", color=POSTER_PALETTE["dark"])
    # Equation annotation.
    fig.text(0.5, -0.04,
             r"$\|\mathbf{w}_c \odot \gamma\|_2$ = L2 norm of fc1.weight "
             r"row $c$ element-wise multiplied by LayerNorm scale $\gamma$",
             ha="center", fontsize=9, color=POSTER_PALETTE["gray"],
             transform=fig.transFigure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _make_single_block(data, out_path, block_idx=10):
    """Single block at larger size."""
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="white",
                            constrained_layout=True)
    blk = data[block_idx]
    _scatter_one_block(ax, block_idx, blk["gain"], blk["std"])
    ax.set_title(f"Effective Gain vs. Activation σ — Block {block_idx}",
                 fontsize=12, fontweight="bold", color=POSTER_PALETTE["dark"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _make_all_blocks(data, out_path):
    """4×3 grid for all 12 blocks."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), facecolor="white",
                              constrained_layout=True)
    for bidx in range(12):
        ax = axes[bidx // 3][bidx % 3]
        blk = data[bidx]
        _scatter_one_block(ax, bidx, blk["gain"], blk["std"])
    fig.suptitle("Effective Per-Channel Gain vs. Pre-GELU σ — All 12 Blocks",
                 fontsize=14, fontweight="bold", color=POSTER_PALETTE["dark"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate effective-gain vs σ_c scatter plots.",
    )
    parser.add_argument(
        "--layer-stats", type=Path,
        default=Path("outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json"),
        help="Path to profiling_result.json from Phase 1.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("all-plots2/analysis"),
        help="Directory for output PNGs.",
    )
    args = parser.parse_args()

    # 1. Load profiling data.
    with open(args.layer_stats, "r", encoding="utf-8") as f:
        profiling = json.load(f)
    stats = profiling["stats"]

    # 2. Load model weights.
    print("Loading ViT-B/16 to extract fc1.weight …")
    from src.model import load_vit
    from src.utils import get_device, seed_everything

    seed_everything(42)
    device = get_device()
    model, _ = load_vit(device)

    fc1_weights = {}
    for bidx in range(12):
        fc1_weights[bidx] = model.blocks[bidx].mlp.fc1.weight.detach().cpu().numpy()

    del model
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()

    # 3. Build per-block data.
    data = {}
    for bidx in range(12):
        ln_gamma = np.array(
            stats[f"blocks.{bidx}/post_layernorm_2"]["layernorm_gamma"],
            dtype=np.float64,
        )
        pc_std = np.array(
            stats[f"blocks.{bidx}/pre_gelu"]["per_channel_std"],
            dtype=np.float64,
        )
        data[bidx] = {
            "gain": _effective_gain(fc1_weights[bidx], ln_gamma),
            "std": pc_std,
        }

    # 4. Generate all variants.
    out = args.output_dir
    _make_three_panel(data, out / "gain_sigma_scatter_blocks_8_9_10.png")
    _make_single_block(data, out / "gain_sigma_scatter_block10.png", block_idx=10)
    _make_all_blocks(data, out / "gain_sigma_scatter_all_blocks.png")

    print("Done — 3 scatter variants generated.")


if __name__ == "__main__":
    main()