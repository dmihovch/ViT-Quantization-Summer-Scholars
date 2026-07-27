"""Entry point for Phase 1 — Baseline Pre-GELU Profiling.

Parses command-line arguments, constructs a :class:`~src.config.ProfilingConfig`,
logs system information, and delegates all experiment logic to
:func:`src.exp1_profiling.run`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import ProfilingConfig
from src.exp1_profiling import run
from src.utils import get_device, log_system_info


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 1.

    Returns
    -------
    argparse.Namespace
        Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        description="Phase 1: Baseline Pre-GELU Profiling for ViT-B/16.",
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
        default=Path("outputs/phase1-profiling"),
        help="Directory where layer stats JSON and histogram PNGs are written.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1024,
        help="Number of validation images to profile (subset for speed). "
        "Use --all to profile the entire dataset.",
    )
    parser.add_argument(
        "--all",
        dest="use_all_images",
        action="store_true",
        help="Profile the entire dataset (overrides --num-images).",
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
        "--skip-outlier-recount",
        dest="skip_outlier_recount",
        action="store_true",
        help=(
            "Skip the second-pass global-σ outlier recount. "
            "Outlier fractions in the output JSON will be approximate (per-batch-σ). "
            "Use only for fast iteration, not for publishable results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging, log system info, build config, and run Phase 1."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    log_system_info()

    config = ProfilingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=None if args.use_all_images else args.num_images,
        batch_size=args.batch_size,
        device=get_device(),
        seed=args.seed,
        num_seeds=args.num_seeds,
        skip_outlier_recount=args.skip_outlier_recount,
    )
    run(config)


if __name__ == "__main__":
    main()