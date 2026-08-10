"""Analyse correlation between LayerNorm γ weights and per-channel pre-GELU σ.

Hypothesis (SmoothQuant, Xiao et al. 2023, ICML):
    High-γ LayerNorm channels amplify the residual stream into the MLP, creating
    the per-channel variance pattern observed in Phase 1 profiling.  If the
    correlation is strong (Pearson r ≫ 0.5), then pre-GELU outliers are not an
    anomaly — they are a deliberate architectural feature of the trained model.

Usage:
    python scripts/analyze_layernorm_gamma.py \
        --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/layernorm-gamma-analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path so `src` imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib.use("Agg")

from src.plotting_utils import ANALYTICAL_COLORS, LABELS

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse LN γ vs per-channel pre-GELU σ correlation.",
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
        default=Path("outputs/layernorm-gamma-analysis"),
        help="Directory for output table and scatter plots.",
    )
    # --run-dir convenience flag
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Convenience: auto-discover files from a run directory "
             "(e.g. outputs/full-run-2026-8-4).",
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


def _discover_layer_stats(run_dir: Path) -> Path | None:
    """Auto-discover profiling_result.json from a run directory."""
    phase1_dir = run_dir / "phase1-profiling"
    if phase1_dir.is_dir():
        for seed_dir in sorted(phase1_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                candidate = seed_dir / "profiling_result.json"
                if candidate.is_file():
                    return candidate
    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    # --- Handle --run-dir auto-discovery ---
    if args.run_dir is not None:
        discovered = _discover_layer_stats(args.run_dir)
        if discovered is not None:
            args.layer_stats = discovered

    # Load Phase 1 stats.
    with open(args.layer_stats, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats: dict[str, dict] = data["stats"]

    # Build per-block correlation data.
    # LN2 (post_layernorm_2) γ weights → pre_gelu per-channel σ.
    results: list[dict] = []
    for block_idx in range(12):
        ln2_sid = f"blocks.{block_idx}/post_layernorm_2"
        gelu_sid = f"blocks.{block_idx}/pre_gelu"

        if ln2_sid not in stats or gelu_sid not in stats:
            logger.warning("Missing data for block %d; skipping.", block_idx)
            continue

        gamma = stats[ln2_sid].get("layernorm_gamma")
        pc_std = stats[gelu_sid].get("per_channel_std")
        pc_mean = stats[gelu_sid].get("per_channel_mean")
        global_std = stats[gelu_sid].get("std")
        global_mean = stats[gelu_sid].get("mean")

        if gamma is None or pc_std is None:
            logger.warning(
                "Missing γ or per_channel_std for block %d; skipping.", block_idx,
            )
            continue

        r = pearson_r(gamma, pc_std)
        results.append({
            "block": block_idx,
            "pearson_r": r,
            "global_mean": global_mean,
            "global_std": global_std,
            "pc_std_min": min(pc_std),
            "pc_std_max": max(pc_std),
            "pc_mean_min": min(pc_mean) if pc_mean else None,
            "pc_mean_max": max(pc_mean) if pc_mean else None,
        })
        logger.info(
            "Block %2d: Pearson r(γ, σ_c) = %+.4f  (global μ=%.2f, σ=%.2f)",
            block_idx, r, global_mean, global_std,
        )

    # Print summary table.
    print()
    print(f"{'Block':>5s}  {'r(γ,σ_c)':>10s}  {'Global μ':>10s}  {'Global σ':>10s}  {'σ_c range':>20s}")
    print(f"{'':->5s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->20s}")
    for r in results:
        sigma_range = f"{r['pc_std_min']:.2f} – {r['pc_std_max']:.2f}"
        print(
            f"{r['block']:5d}  {r['pearson_r']:+10.4f}  "
            f"{r['global_mean']:10.2f}  {r['global_std']:10.2f}  "
            f"{sigma_range:>20s}"
        )

    # Compute mean r across blocks.
    mean_r = sum(r["pearson_r"] for r in results) / len(results) if results else 0.0
    print(f"\nMean Pearson r across all blocks: {mean_r:+.4f}")

    # NOTE: We do NOT generate a bar chart of per-block Pearson r here.
    # LN2 γ is 768-dim (embedding) while pre-GELU σ_c is 3072-dim
    # (MLP hidden).  These are different spaces — a valid element-wise
    # Pearson r cannot be computed without arbitrary broadcasting, and
    # the resulting r ≈ 0.0003 is just statistical noise.  Plotting
    # microscopic noise as giant coloured bars misleads the reader.
    #
    # The correct analysis is in ``analyze_effective_gain.py``, which
    # computes ‖fc1.weight[c,:] ⊙ γ‖₂ (3072-dim) vs σ_c (3072-dim).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Skipping LN2 γ correlation bar chart: γ (768-dim) and σ_c "
        "(3072-dim) live in different spaces.  See analyze_effective_gain.py "
        "for the correct analysis."
    )

    # Save JSON summary.
    summary_path = args.output_dir / "ln_gamma_correlation.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean_pearson_r": mean_r,
            "per_block": results,
        }, f, indent=2)
    logger.info("Saved correlation summary to %s", summary_path)


if __name__ == "__main__":
    main()