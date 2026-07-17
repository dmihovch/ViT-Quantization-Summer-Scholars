"""Entry point for Phase 1 — Baseline Pre-GELU Profiling.

Parses command-line arguments, constructs a :class:`~src.config.ProfilingConfig`,
and delegates all experiment logic to :func:`src.exp1_profiling.run`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import ProfilingConfig
from src.exp1_profiling import run
from src.utils import get_device, seed_everything


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
        default=Path("data/imagenet-val"),
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
        help="Number of validation images to profile (subset for speed).",
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
    return parser.parse_args()


def main() -> None:
    """Configure logging, seed RNGs, build config, and run Phase 1."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    seed_everything(args.seed)

    config = ProfilingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=get_device(),
    )
    run(config)


if __name__ == "__main__":
    main()
