"""Analyse effective per-channel gain ‖fc1.weight[c, :] ⊙ γ‖ vs per-channel pre-GELU σ.

Hypothesis (SmoothQuant, Xiao et al. 2023, ICML):
    High-γ LayerNorm channels amplify the residual stream into the MLP, and
    each fc1 row selectively weights those amplified channels.  The effective
    per-channel gain — the L2 norm of the Hadamard product of fc1.weight[c, :]
    with the LN2 γ vector — captures the combined scaling that determines how
    strongly channel c responds.  If this effective gain correlates with
    per_channel_std (both 3072-dim), then pre-GELU outliers are a deliberate
    consequence of the trained weights, not an anomaly.

This replaces the naive correlation of LN2 γ (768-dim) with pre-GELU σ_c
(3072-dim) in ``analyze_layernorm_gamma.py``, which found r≈0.0003 because
the two vectors live in different spaces.

Usage:
    python scripts/analyze_effective_gain.py \
        --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/effective-gain-analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse ‖fc1.weight ⊙ γ‖ vs per-channel pre-GELU σ correlation.",
    )
    parser.add_argument(
        "--layer-stats",
        type=Path,
        default=Path("outputs/phase1-profiling/seed_42/profiling_result.json"),
        help="Path to profiling_result.json from Phase 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/effective-gain-analysis"),
        help="Directory for output table and bar chart.",
    )
    return parser.parse_args()


def pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length lists.

    Parameters
    ----------
    x, y:
        Equal-length lists of float values.

    Returns
    -------
    float
        Pearson r in [-1, 1].  Returns 0.0 if either list has zero variance.
    """
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _compute_effective_gain(
    fc1_weight: np.ndarray, ln_gamma: np.ndarray
) -> np.ndarray:
    """Compute the effective per-channel gain for every MLP hidden channel.

    For each row *c* of ``fc1_weight`` (shape [D_mlp, D]), compute the L2
    norm of the Hadamard product with ``ln_gamma`` (shape [D]):

        effective_gain[c] = ‖fc1_weight[c, :] ⊙ ln_gamma‖₂

    Parameters
    ----------
    fc1_weight:
        MLP fc1 weight matrix, shape ``(3072, 768)`` for ViT-B/16.
    ln_gamma:
        LayerNorm γ (scale) vector, shape ``(768,)``.

    Returns
    -------
    np.ndarray
        Effective gain per MLP hidden channel, shape ``(3072,)``.
    """
    # Element-wise multiply each row by ln_gamma: (3072, 768) ⊙ (768,) → (3072, 768)
    weighted = fc1_weight * ln_gamma[np.newaxis, :]
    # L2 norm along the embedding axis: (3072,)
    return np.linalg.norm(weighted, axis=1)


def main() -> None:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    # ------------------------------------------------------------------
    # 1. Load Phase 1 stats.
    # ------------------------------------------------------------------
    with open(args.layer_stats, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats: dict[str, dict] = data["stats"]

    # ------------------------------------------------------------------
    # 2. Load the ViT model (briefly — only need fc1.weight per block).
    # ------------------------------------------------------------------
    logger.info("Loading model to extract fc1.weight (no data pass needed)...")
    from src.model import load_vit
    from src.utils import get_device, seed_everything

    seed_everything(42)
    device = get_device()
    model, _transform = load_vit(device)

    fc1_weights: dict[int, np.ndarray] = {}
    for block_idx in range(12):
        w = model.blocks[block_idx].mlp.fc1.weight.detach().cpu().numpy()
        fc1_weights[block_idx] = w
        logger.debug("Block %d fc1.weight shape: %s", block_idx, w.shape)

    # Model no longer needed — let it be garbage collected.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Extracted fc1.weight for all %d blocks; model released.", len(fc1_weights))

    # ------------------------------------------------------------------
    # 3. Build per-block correlation data.
    # ------------------------------------------------------------------
    results: list[dict[str, Any]] = []
    for block_idx in range(12):
        ln2_sid = f"blocks.{block_idx}/post_layernorm_2"
        gelu_sid = f"blocks.{block_idx}/pre_gelu"

        if ln2_sid not in stats or gelu_sid not in stats:
            logger.warning("Missing data for block %d; skipping.", block_idx)
            continue

        ln_gamma = stats[ln2_sid].get("layernorm_gamma")
        pc_std = stats[gelu_sid].get("per_channel_std")
        pc_mean = stats[gelu_sid].get("per_channel_mean")
        global_std = stats[gelu_sid].get("std")
        global_mean = stats[gelu_sid].get("mean")

        if ln_gamma is None or pc_std is None:
            logger.warning(
                "Missing γ or per_channel_std for block %d; skipping.", block_idx,
            )
            continue

        # Validate shapes.
        fc1_w = fc1_weights[block_idx]  # (3072, 768)
        gamma_arr = np.array(ln_gamma, dtype=np.float64)
        assert fc1_w.shape == (3072, 768), f"Unexpected fc1 shape: {fc1_w.shape}"
        assert gamma_arr.shape == (768,), f"Unexpected γ shape: {gamma_arr.shape}"
        assert len(pc_std) == 3072, f"Unexpected σ_c length: {len(pc_std)}"

        # Compute effective gain per channel (3072-dim).
        effective_gain = _compute_effective_gain(fc1_w, gamma_arr)

        r = pearson_r(effective_gain.tolist(), pc_std)

        results.append({
            "block": block_idx,
            "pearson_r": r,
            "global_mean": global_mean,
            "global_std": global_std,
            "pc_std_min": min(pc_std),
            "pc_std_max": max(pc_std),
            "pc_mean_min": min(pc_mean) if pc_mean else None,
            "pc_mean_max": max(pc_mean) if pc_mean else None,
            "effective_gain_min": float(effective_gain.min()),
            "effective_gain_max": float(effective_gain.max()),
            "effective_gain_mean": float(effective_gain.mean()),
            "effective_gain_std": float(effective_gain.std()),
        })
        logger.info(
            "Block %2d: Pearson r(‖fc1⊙γ‖, σ_c) = %+.4f  "
            "(global μ=%.2f, σ=%.2f; gain μ=%.2f, σ=%.2f)",
            block_idx, r, global_mean, global_std,
            effective_gain.mean(), effective_gain.std(),
        )

    # ------------------------------------------------------------------
    # 4. Print summary table.
    # ------------------------------------------------------------------
    if not results:
        logger.error("No valid blocks found; aborting.")
        return

    print()
    header = (
        f"{'Block':>5s}  {'r(gain,σ_c)':>12s}  {'Global μ':>10s}  {'Global σ':>10s}  "
        f"{'σ_c range':>20s}  {'Gain range':>20s}"
    )
    sep = (
        f"{'':->5s}  {'':->12s}  {'':->10s}  {'':->10s}  "
        f"{'':->20s}  {'':->20s}"
    )
    print(header)
    print(sep)
    for r in results:
        sigma_range = f"{r['pc_std_min']:.2f} – {r['pc_std_max']:.2f}"
        gain_range = f"{r['effective_gain_min']:.2f} – {r['effective_gain_max']:.2f}"
        print(
            f"{r['block']:5d}  {r['pearson_r']:+12.4f}  "
            f"{r['global_mean']:10.2f}  {r['global_std']:10.2f}  "
            f"{sigma_range:>20s}  {gain_range:>20s}"
        )

    # Mean r across blocks.
    mean_r = sum(r["pearson_r"] for r in results) / len(results)
    print(f"\nMean Pearson r across all blocks: {mean_r:+.4f}")

    # ------------------------------------------------------------------
    # 5. Bar chart of per-block Pearson r.
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    blocks = [r["block"] for r in results]
    rs = [r["pearson_r"] for r in results]
    colors = ["steelblue" if r >= 0 else "coral" for r in rs]
    ax.bar(blocks, rs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Encoder Block")
    ax.set_ylabel("Pearson r(‖fc1⊙γ‖, pre-GELU σ_c)")
    ax.set_title(
        "Effective per-Channel Gain vs Per-Channel pre-GELU σ Correlation\n"
        r"(‖fc1.weight[c, :] $\odot$ LN2 $\gamma$‖₂  —  both 3072-dim)"
    )
    ax.set_xticks(blocks)
    fig.tight_layout()
    bar_path = args.output_dir / "effective_gain_correlation_bars.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart to %s", bar_path)

    # ------------------------------------------------------------------
    # 6. Save JSON summary.
    # ------------------------------------------------------------------
    summary_path = args.output_dir / "effective_gain_correlation.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean_pearson_r": mean_r,
            "per_block": results,
        }, f, indent=2)
    logger.info("Saved correlation summary to %s", summary_path)


if __name__ == "__main__":
    main()
