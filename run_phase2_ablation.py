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
        help="Global random seed for reproducibility.",
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
        default=Path("outputs/phase1-profiling/profiling_result.json"),
        help="Path to the profiling_result.json produced by Phase 1.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging, seed RNGs, build config, and run Phase 2."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    seed_everything(args.seed)

    config = AblationConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=get_device(),
        sigma_thresholds=tuple(args.sigma_thresholds),
        layer_stats_path=args.layer_stats,
        seed=args.seed,
    )
    run(config)


if __name__ == "__main__":
    main()
