#!/usr/bin/env python
"""Generate poster-quality plots from existing Phase 1 and Phase 2 data.

Reads ``profiling_result.json`` and up to two ``ablation_results.csv`` files
and produces publication/poster-ready figures.  All plots use the poster
styling conventions from :mod:`src.plotting_poster` (white background, custom
colour palette, direct annotation, ≥14 pt fonts).

Usage
-----
    # Phase 1 only (profiling stats, no ablation data needed):
    python scripts/generate_poster_plots.py \\
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \\
        --output-dir outputs/poster-plots

    # Phase 1 + Phase 2 comparison (global vs per-channel):
    python scripts/generate_poster_plots.py \\
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \\
        --phase2-csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \\
        --phase2-csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \\
        --output-dir outputs/poster-plots

    # Full suite including activation distribution overlay (needs model + GPU):
    python scripts/generate_poster_plots.py \\
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \\
        --phase2-csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \\
        --phase2-csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \\
        --output-dir outputs/poster-plots \\
        --histogram-data-dir data

Plots generated
---------------
* ``poster_outlier_grid_3.0_sigma.png`` (and 4.0, 6.0)
* ``poster_sigma_ridgeline.png``
* ``poster_entropy_streamgraph.png``
* ``poster_mean_hinton_blk10.png``
* ``poster_accuracy_vs_sparsity.png``
* ``poster_ablation_waterfall.png``
* ``poster_activation_overlay_blk10.png`` (requires --histogram-data-dir)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from nnsight import NNsight

from src.ablation import AblationResult
from src.data_loader import build_val_loader
from src.model import load_vit
from src.plotting_poster import (
    plot_ablation_waterfall,
    plot_accuracy_vs_sparsity_scatter,
    plot_activation_distribution_overlay,
    plot_attention_entropy_streamgraph,
    plot_outlier_site_grid,
    plot_per_channel_mean_hinton,
    plot_per_channel_sigma_ridgeline,
)
from src.profiler import LayerStats, SiteId, histogram_profile_vit, load_profiling_result
from src.utils import ensure_dir, get_device, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_ablation_results(path: Path) -> list[AblationResult]:
    """Load ablation results from a CSV file."""
    results: list[AblationResult] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(AblationResult(
                site=row["site"],
                sigma_threshold=float(row["sigma_threshold"]),
                site_identifier=row["site_identifier"],
                pct_zeroed=float(row["pct_zeroed"]),
                top1_accuracy=float(row["top1_accuracy"]),
                top5_accuracy=float(row["top5_accuracy"]),
                baseline_top1=float(row["baseline_top1"]),
                baseline_top5=float(row["baseline_top5"]),
                is_random=row.get("is_random", "False") == "True",
                granularity=row.get("granularity", "global"),
                ablation_mode=row.get("ablation_mode", "outlier"),
                cls_entropy=json.loads(row.get("cls_entropy", "[]")),
                patch_entropy=json.loads(row.get("patch_entropy", "[]")),
                baseline_cls_entropy=json.loads(
                    row.get("baseline_cls_entropy", "[]"),
                ),
                baseline_patch_entropy=json.loads(
                    row.get("baseline_patch_entropy", "[]"),
                ),
            ))
    logger.info("Loaded %d ablation results from %s", len(results), path)
    return results


# ---------------------------------------------------------------------------
# Phase 1 poster plots
# ---------------------------------------------------------------------------


def _generate_phase1_poster_plots(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> list[Path]:
    """Generate all Phase 1 poster plots."""
    written: list[Path] = []

    # 1. Outlier site grid at 3σ, 4σ, 6σ.
    outlier_fracs: dict[str, dict[str, float]] = {
        k: s.outlier_fractions for k, s in stats.items()
        if s.outlier_fractions
    }
    if outlier_fracs:
        for sigma_key in ["3.0_sigma", "4.0_sigma", "6.0_sigma"]:
            p = output_dir / f"poster_outlier_grid_{sigma_key}.png"
            plot_outlier_site_grid(outlier_fracs, p, sigma_key=sigma_key)
            written.append(p)

    # 2. Per-channel σ ridgeline.
    pc_stds: dict[str, list[float]] = {
        k: s.per_channel_std for k, s in stats.items()
        if s.per_channel_std is not None
    }
    if pc_stds:
        p = output_dir / "poster_sigma_ridgeline.png"
        plot_per_channel_sigma_ridgeline(pc_stds, p)
        written.append(p)

    # 3. Attention entropy streamgraph.
    cls_ent: dict[str, list[float]] = {
        k: s.attention_entropy_cls for k, s in stats.items()
        if s.attention_entropy_cls is not None
    }
    if cls_ent:
        p = output_dir / "poster_entropy_streamgraph.png"
        plot_attention_entropy_streamgraph(cls_ent, p)
        written.append(p)

    # 4. Per-channel mean Hinton (block 10).
    pc_means: dict[str, list[float]] = {
        k: s.per_channel_mean for k, s in stats.items()
        if s.per_channel_mean is not None
    }
    if pc_means:
        p = output_dir / "poster_mean_hinton_blk10.png"
        plot_per_channel_mean_hinton(pc_means, 10, p)
        written.append(p)

    return written


# ---------------------------------------------------------------------------
# Phase 2 poster plots
# ---------------------------------------------------------------------------


def _generate_phase2_poster_plots(
    results_a: list[AblationResult],
    results_b: list[AblationResult],
    output_dir: Path,
    label_a: str,
    label_b: str,
    stats: dict[SiteId, LayerStats] | None = None,
    histogram_data_dir: Path | None = None,
) -> list[Path]:
    """Generate all Phase 2 poster plots (comparison and overlay)."""
    written: list[Path] = []

    # 5. Accuracy vs sparsity scatter.
    pre_gelu_a = [r for r in results_a if r.site == "pre_gelu" and not r.is_random]
    pre_gelu_b = [r for r in results_b if r.site == "pre_gelu" and not r.is_random]
    if pre_gelu_a and pre_gelu_b:
        p = output_dir / "poster_accuracy_vs_sparsity.png"
        plot_accuracy_vs_sparsity_scatter(
            pre_gelu_a, pre_gelu_b, p, label_a=label_a, label_b=label_b,
        )
        written.append(p)

    # 6. Ablation waterfall at k=3.
    # Requires ablation data for all modes: global, mean_only, var_only, outlier.
    # If any mode's data is missing, the waterfall chart is skipped rather than
    # filled with fake numbers.
    def _acc_at_k(results, k, granularity, mode="outlier"):
        subset = [r for r in results
                  if r.site == "pre_gelu" and not r.is_random
                  and r.sigma_threshold == k
                  and r.granularity == granularity
                  and r.ablation_mode == mode]
        return float(np.mean([r.top1_accuracy for r in subset])) if subset else 0.0

    baseline = pre_gelu_a[0].baseline_top1 if pre_gelu_a else 85.0
    global_k3 = _acc_at_k(results_a, 3.0, "global")
    pc_outlier_k3 = _acc_at_k(results_b, 3.0, "per_channel", mode="outlier")
    mean_only_k3 = _acc_at_k(results_b, 3.0, "per_channel", mode="mean_only")
    var_only_k3 = _acc_at_k(results_b, 3.0, "per_channel", mode="var_only")

    if global_k3 > 0 and pc_outlier_k3 > 0 and mean_only_k3 > 0 and var_only_k3 > 0:
        p = output_dir / "poster_ablation_waterfall.png"
        plot_ablation_waterfall(
            baseline, global_k3, mean_only_k3, var_only_k3, pc_outlier_k3, p,
            sigma_k=3.0,
        )
        written.append(p)
    else:
        missing = []
        if global_k3 <= 0:
            missing.append("global")
        if pc_outlier_k3 <= 0:
            missing.append("per-channel outlier")
        if mean_only_k3 <= 0:
            missing.append("per-channel mean_only")
        if var_only_k3 <= 0:
            missing.append("per-channel var_only")
        logger.info(
            "Skipping ablation waterfall: missing data for %s. "
            "Run mean_only + var_only ablation experiments (RQ2) to populate.",
            ", ".join(missing),
        )

    # 7. Activation distribution overlay (requires model + GPU).
    if histogram_data_dir is not None and stats is not None:
        _generate_activation_overlay(stats, histogram_data_dir, output_dir, written)

    return written


def _generate_activation_overlay(
    stats: dict[SiteId, LayerStats],
    data_dir: Path,
    output_dir: Path,
    written: list[Path],
) -> None:
    """Generate the activation distribution overlay for block 10 pre-GELU.

    Requires the model and GPU to collect raw activation tensors.
    """
    device = get_device()
    model, transform = load_vit(device)
    wrapped = NNsight(model)

    loader = build_val_loader(
        data_dir, transform, 64, None, device, shuffle=True,
    )
    images, _ = next(iter(loader))

    with torch.no_grad():
        raw_tensors = histogram_profile_vit(wrapped, images.to(device), (10,))

    block10_key = "blocks.10/pre_gelu"
    if block10_key not in raw_tensors:
        logger.warning("No block 10 pre_gelu tensor in histogram output; skipping overlay.")
        return

    tensor = raw_tensors[block10_key].detach().cpu().numpy().ravel().astype(np.float32)
    gelu_stats = stats.get(block10_key)
    if gelu_stats is None:
        logger.warning("No Phase 1 stats for block 10 pre_gelu; skipping overlay.")
        return

    p = output_dir / "poster_activation_overlay_blk10.png"
    plot_activation_distribution_overlay(
        tensor,
        "Block 10 — pre-GELU Activation Distribution",
        p,
        global_mean=gelu_stats.mean,
        global_std=gelu_stats.std,
        per_channel_stds=gelu_stats.per_channel_std,
        per_channel_means=gelu_stats.per_channel_mean,
        sigma_k=3.0,
    )
    written.append(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate poster-quality plots from existing data.",
    )
    parser.add_argument(
        "--phase1-json", type=Path, default=None,
        help="Path to profiling_result.json (Phase 1).",
    )
    parser.add_argument(
        "--phase2-csv-a", type=Path, default=None,
        help="Path to first ablation_results.csv (e.g. global).",
    )
    parser.add_argument(
        "--phase2-csv-b", type=Path, default=None,
        help="Path to second ablation_results.csv (e.g. per-channel).",
    )
    parser.add_argument(
        "--label-a", type=str, default="Global",
        help="Legend label for CSV A.",
    )
    parser.add_argument(
        "--label-b", type=str, default="Per-channel",
        help="Legend label for CSV B.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write poster PNGs.",
    )
    parser.add_argument(
        "--histogram-data-dir", type=Path, default=None,
        help=(
            "ImageNet val directory for the activation distribution overlay. "
            "Requires model + GPU.  If omitted, that plot is skipped."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    if args.phase1_json is None and args.phase2_csv_a is None:
        logger.error(
            "No input files specified.  Provide --phase1-json and/or "
            "--phase2-csv-a/--phase2-csv-b."
        )
        sys.exit(1)

    total_written: list[Path] = []

    # --- Phase 1 ---
    stats: dict[SiteId, LayerStats] | None = None
    if args.phase1_json is not None:
        logger.info("=== Phase 1 poster plots ===")
        result = load_profiling_result(args.phase1_json)
        stats = result.stats
        logger.info("Loaded %d sites from %s", len(stats), args.phase1_json)
        total_written.extend(_generate_phase1_poster_plots(stats, args.output_dir))

    # --- Phase 2 ---
    if args.phase2_csv_a is not None and args.phase2_csv_b is not None:
        logger.info("=== Phase 2 poster plots ===")
        results_a = _load_ablation_results(args.phase2_csv_a)
        results_b = _load_ablation_results(args.phase2_csv_b)
        total_written.extend(
            _generate_phase2_poster_plots(
                results_a, results_b, args.output_dir,
                label_a=args.label_a, label_b=args.label_b,
                stats=stats,
                histogram_data_dir=args.histogram_data_dir,
            ),
        )
    elif args.phase2_csv_a is not None or args.phase2_csv_b is not None:
        logger.error("Both --phase2-csv-a and --phase2-csv-b are required for comparison.")
        sys.exit(1)

    logger.info("Generated %d poster plots in %s", len(total_written), args.output_dir)
    for p in total_written:
        logger.info("  %s", p.name)


if __name__ == "__main__":
    main()
