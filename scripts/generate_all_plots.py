#!/usr/bin/env python
"""Master orchestration script: regenerate all plots from existing data.

Reads data from a 5-seed full run directory and regenerates every plot,
organized into sub-directories under a single configurable output root::

    {output_root}/
    ├── phase1/     # Phase 1 profiling plots (heatmaps, kurtosis, etc.)
    ├── phase2/     # Phase 2 ablation plots (accuracy, %-zeroed, comparisons)
    ├── analysis/   # Post-hoc analysis plots (CI, effective channels, correlation)
    └── poster/     # 6 final poster-quality figures

Usage
-----
    # Default: read from outputs/5-seed-full-run-2026-08-05, write to plots/
    python scripts/generate_all_plots.py

    # Custom data dir and output dir:
    python scripts/generate_all_plots.py \\
        --data-dir outputs/5-seed-full-run-2026-08-05 \\
        --output-root my_plots

    # Include live activation overlay (requires GPU + ImageNet val data):
    python scripts/generate_all_plots.py \\
        --run-live-overlay \\
        --imagenet-val-dir data/imagenet-val

    # Skip certain groups:
    python scripts/generate_all_plots.py --skip-phase1 --skip-analysis
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path("outputs/5-seed-full-run-2026-08-05")
_DEFAULT_OUTPUT_ROOT = Path("plots")


def _run(cmd: list[str], *, label: str, timeout: int = 1800) -> bool:
    """Run a subprocess command with logging. Returns True on success."""
    logger.info("--- %s ---", label)
    logger.debug("  $ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  %s", line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.warning("  [stderr] %s", line)
    if result.returncode != 0:
        logger.error("%s FAILED (exit code %d)", label, result.returncode)
        return False
    logger.info("%s ✓", label)
    return True


def _phase1(data_dir: Path, output_dir: Path) -> bool:
    profiling_json = data_dir / "phase1-profiling" / "seed_42" / "profiling_result.json"
    if not profiling_json.exists():
        logger.error("Phase 1 profiling JSON not found: %s", profiling_json)
        return False
    ensure_dir(output_dir)
    return _run(
        [
            sys.executable, "scripts/regenerate_plots.py",
            "--phase1-json", profiling_json,
            "--output-dir", output_dir,
        ],
        label=f"Phase 1 plots → {output_dir}",
    )


def _phase2(data_dir: Path, output_dir: Path) -> bool:
    """Phase 2: single-seed comparison (global vs per-channel) to cover all
    ablation plots (accuracy, %-zeroed, comparisons)."""
    csv_global = sorted(data_dir.glob("phase2-global/seed_*/ablation_results.csv"))
    csv_pc = sorted(data_dir.glob("phase2-per-channel/seed_*/ablation_results.csv"))

    if not csv_global or not csv_pc:
        logger.error("Phase 2 CSVs not found in %s", data_dir)
        return False

    ensure_dir(output_dir)
    return _run(
        [
            sys.executable, "scripts/regenerate_plots.py",
            "--phase2-csv-a", csv_global[0],
            "--phase2-csv-b", csv_pc[0],
            "--output-dir", output_dir,
        ],
        label=f"Phase 2 plots → {output_dir}",
    )


def _analysis(data_dir: Path, output_dir: Path) -> bool:
    """Post-hoc analysis: CI deltas, effective channels, degradation,
    effective gain correlation."""
    profiling_json = data_dir / "phase1-profiling" / "seed_42" / "profiling_result.json"

    csv_global = sorted(data_dir.glob("phase2-global/seed_*/ablation_results.csv"))
    csv_pc = sorted(data_dir.glob("phase2-per-channel/seed_*/ablation_results.csv"))

    ensure_dir(output_dir)
    ok = True

    if csv_global and csv_pc:
        ok &= _run(
            [
                sys.executable, "scripts/analyze_ablation_results.py",
                "--csv-a", csv_global[0],
                "--csv-b", csv_pc[0],
                "--output-dir", output_dir,
            ],
            label=f"Ablation analysis → {output_dir}",
        )

    if profiling_json.exists():
        ok &= _run(
            [
                sys.executable, "scripts/analyze_effective_gain.py",
                "--layer-stats", profiling_json,
                "--output-dir", output_dir,
            ],
            label=f"Effective gain analysis → {output_dir}",
        )
        ok &= _run(
            [
                sys.executable, "scripts/analyze_layernorm_gamma.py",
                "--layer-stats", profiling_json,
                "--output-dir", output_dir,
            ],
            label=f"LayerNorm γ analysis → {output_dir}",
        )

    return ok


def _poster(
    data_dir: Path,
    output_dir: Path,
    *,
    run_live_overlay: bool = False,
    imagenet_val_dir: Path | None = None,
) -> bool:
    """6 final poster figures via generate_final_report_plots.py."""
    ensure_dir(output_dir)
    cmd = [
        sys.executable, "scripts/generate_final_report_plots.py",
        "--input-dir", data_dir,
        "--output-dir", output_dir,
    ]
    if run_live_overlay:
        cmd.append("--run-live-overlay")
        if imagenet_val_dir:
            cmd.extend(["--imagenet-val-dir", imagenet_val_dir])
    return _run(cmd, label=f"Poster figures → {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all plots from a 5-seed full run.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
        help=f"Path to 5-seed-full-run directory (default: {_DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT,
        help=f"Top-level output directory (default: {_DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 profiling plots.",
    )
    parser.add_argument(
        "--skip-phase2", action="store_true",
        help="Skip Phase 2 ablation plots.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip analysis plots.",
    )
    parser.add_argument(
        "--skip-poster", action="store_true",
        help="Skip poster figures.",
    )
    parser.add_argument(
        "--run-live-overlay", action="store_true",
        help="Run live model pass for activation overlay (needs GPU).",
    )
    parser.add_argument(
        "--imagenet-val-dir", type=Path, default=Path("data/imagenet-val"),
        help="Path to ImageNet val set for live overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_root: Path = args.output_root.resolve()

    if not data_dir.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    logger.info("Data source:   %s", data_dir)
    logger.info("Output root:   %s", output_root)

    all_ok = True

    if not args.skip_phase1:
        ok = _phase1(data_dir, output_root / "phase1")
        all_ok = all_ok and ok
        if not ok:
            logger.warning("Phase 1 generation had errors; continuing...")

    if not args.skip_phase2:
        ok = _phase2(data_dir, output_root / "phase2")
        all_ok = all_ok and ok
        if not ok:
            logger.warning("Phase 2 generation had errors; continuing...")

    if not args.skip_analysis:
        ok = _analysis(data_dir, output_root / "analysis")
        all_ok = all_ok and ok
        if not ok:
            logger.warning("Analysis generation had errors; continuing...")

    if not args.skip_poster:
        ok = _poster(
            data_dir, output_root / "poster",
            run_live_overlay=args.run_live_overlay,
            imagenet_val_dir=args.imagenet_val_dir,
        )
        all_ok = all_ok and ok
        if not ok:
            logger.warning("Poster generation had errors; continuing...")

    logger.info("=" * 60)
    if all_ok:
        logger.info("All plot groups generated successfully in %s", output_root)
    else:
        logger.warning(
            "Some plot groups had errors. See log above. "
            "Output is in %s", output_root,
        )


if __name__ == "__main__":
    main()
