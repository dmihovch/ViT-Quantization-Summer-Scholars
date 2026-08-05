"""Entry point for Phase 2 — Outlier Ablation (Zeroing).

Parses command-line arguments, constructs an :class:`~src.config.AblationConfig`,
and delegates all experiment logic to :func:`src.exp2_ablation.run`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import AblationConfig
from src.exp2_ablation import run
from src.utils import get_device, seed_everything


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 2.

    Returns
    -------
    argparse.Namespace
        Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2: Outlier Ablation sweep for ViT-B/16.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory of the ImageNet validation split (ImageFolder layout).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2-ablation"),
        help="Directory where results CSV and accuracy/zeroed plots are written.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50000,
        help="Number of validation images to evaluate (default: full val set).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for the DataLoader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of independent runs with different seeds. "
        "Results saved to output_dir/seed_{s}/ for each seed.",
    )
    parser.add_argument(
        "--sigma-thresholds",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 6.0],
        metavar="K",
        help="Sigma multipliers k to sweep (default: 3 4 6, matching Phase 1 OUTLIER_SIGMAS).",
    )
    parser.add_argument(
        "--layer-stats",
        type=Path,
        default=Path("outputs/phase1-profiling/seed_42/profiling_result.json"),
        help="Path to the profiling_result.json produced by Phase 1.",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="global",
        choices=["global", "per_channel", "all"],
        help="Zeroing granularity: 'global' (per-layer μ,σ), 'per_channel' "
        "(per-channel μ_c,σ_c for pre_gelu only), or 'all' (run both back-to-back).",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="outlier",
        choices=["outlier", "mean_only", "var_only"],
        help="Per-channel ablation variant (only with --granularity per_channel): "
        "'outlier' (full per-channel), 'mean_only' (per-channel μ_c + global σ), "
        "'var_only' (global μ + per-channel σ_c).",
    )
    parser.add_argument(
        "--layer-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Only ablate blocks in this inclusive range (0-based).  "
        "E.g. '--layer-range 10 10' for block 10 only.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace, granularity: str) -> AblationConfig:
    """Build an AblationConfig for a specific granularity mode.

    Parameters
    ----------
    args:
        Parsed command-line arguments.
    granularity:
        One of ``"global"`` or ``"per_channel"``.

    Returns
    -------
    AblationConfig
    """
    layer_range: tuple[int, int] | None = None
    if args.layer_range is not None:
        layer_range = (args.layer_range[0], args.layer_range[1])

    return AblationConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=get_device(),
        sigma_thresholds=tuple(args.sigma_thresholds),
        layer_stats_path=args.layer_stats,
        seed=args.seed,
        num_seeds=args.num_seeds,
        granularity=granularity,
        ablation_mode=args.ablation_mode,
        layer_range=layer_range,
    )


def main() -> None:
    """Configure logging, seed RNGs, build config(s), and run Phase 2."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    seed_everything(args.seed)

    if args.granularity == "all":
        # Run global first, then per_channel, back-to-back.
        logger.info("=== Running global granularity ===")
        run(_build_config(args, "global"))

        logger.info("=== Running per_channel granularity ===")
        run(_build_config(args, "per_channel"))
    else:
        run(_build_config(args, args.granularity))


if __name__ == "__main__":
    main()