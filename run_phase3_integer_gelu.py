"""Entry point for Phase 3 — Integer GELU Exploration.

Parses command-line arguments, constructs an
:class:`~src.config.IntegerGELUConfig`, and delegates all experiment logic
to :func:`src.exp3_integer_gelu.run`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import IntegerGELUConfig
from src.exp3_integer_gelu import run
from src.utils import get_device, seed_everything


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 3.

    Returns
    -------
    argparse.Namespace
        Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        description="Phase 3: Integer GELU LUT exploration for ViT-B/16.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase3-integer-gelu"),
        help="Directory where LUT comparison plots and metrics JSON are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--layer-stats",
        type=Path,
        default=Path("outputs/phase1-profiling/profiling_result.json"),
        help="Path to the profiling_result.json produced by Phase 1.",
    )
    parser.add_argument(
        "--ablation-stats",
        type=Path,
        default=Path("outputs/phase2-ablation/ablation_results.csv"),
        help="Path to the ablation_results.csv produced by Phase 2.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging, seed RNGs, build config, and run Phase 3."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    seed_everything(args.seed)

    config = IntegerGELUConfig(
        output_dir=args.output_dir,
        device=get_device(),
        layer_stats_path=args.layer_stats,
        ablation_stats_path=args.ablation_stats,
    )
    run(config)


if __name__ == "__main__":
    main()
