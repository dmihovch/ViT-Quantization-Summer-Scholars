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

    # Phase 1 + Phase 2 comparison via CSV files (used by regenerate_all.sh):
    python scripts/generate_poster_plots.py \\
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \\
        --csv-a outputs/phase2-global/seed_42/ablation_results.csv \\
        --csv-b outputs/phase2-per-channel/seed_42/ablation_results.csv \\
        --csv-b outputs/phase2-per-channel-mean-only/seed_42/ablation_results.csv \\
        --csv-b outputs/phase2-per-channel-var-only/seed_42/ablation_results.csv \\
        --output-dir outputs/poster-plots

    # Multi-seed directory mode:
    python scripts/generate_poster_plots.py \\
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \\
        --phase2-dir-a outputs/phase2-global \\
        --phase2-dir-b outputs/phase2-per-channel \\
        --phase2-dir-mean-only outputs/phase2-per-channel-mean-only \\
        --phase2-dir-var-only outputs/phase2-per-channel-var-only \\
        --output-dir outputs/poster-plots

    # Full suite including activation distribution overlay (needs model + GPU):
    python scripts/generate_poster_plots.py \\
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \\
        --csv-a outputs/phase2-global/seed_42/ablation_results.csv \\
        --csv-b outputs/phase2-per-channel/seed_42/ablation_results.csv \\
        --csv-b outputs/phase2-per-channel-mean-only/seed_42/ablation_results.csv \\
        --csv-b outputs/phase2-per-channel-var-only/seed_42/ablation_results.csv \\
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
    plot_ablation_comparison,
    plot_accuracy_vs_sparsity_scatter,
    plot_activation_distribution_overlay,
    plot_attention_entropy_heatmap,
    plot_effective_gain_scatter,
    plot_outlier_site_grid,
    plot_per_channel_mean_histogram,
    plot_per_channel_sigma_line,
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


def _load_ablation_results(paths: list[Path]) -> list[AblationResult]:
    """Load ablation results from a list of CSV files."""
    results: list[AblationResult] = []
    for path in paths:
        if not path.exists():
            logger.warning("Skipping non-existent file: %s", path)
            continue
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
    logger.info("Loaded %d total ablation results from %d files", len(results), len(paths))
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
        plot_per_channel_sigma_line(pc_stds, p)
        written.append(p)

    # 3. Attention entropy streamgraph.
    cls_ent: dict[str, list[float]] = {
        k: s.attention_entropy_cls for k, s in stats.items()
        if s.attention_entropy_cls is not None
    }
    if cls_ent:
        p = output_dir / "poster_entropy_streamgraph.png"
        plot_attention_entropy_heatmap(cls_ent, p)
        written.append(p)

    # 4. Per-channel mean Hinton (block 10).
    pc_means: dict[str, list[float]] = {
        k: s.per_channel_mean for k, s in stats.items()
        if s.per_channel_mean is not None
    }
    if pc_means:
        p = output_dir / "poster_mean_hinton_blk10.png"
        plot_per_channel_mean_histogram(pc_means, 10, p)
        written.append(p)

    # 5. Effective gain vs σ_c scatter (requires model weights — skip if unavailable).
    try:
        p = output_dir / "poster_gain_sigma_scatter.png"
        _generate_gain_sigma_scatter(stats, p)
        written.append(p)
    except Exception:
        logger.info("Skipping gain-σ scatter (model weights unavailable).")

    return written


# Helper: gain-sigma scatter (requires model + profiling data)

def _generate_gain_sigma_scatter(
    raw_stats: dict[str, dict],
    output_path: Path,
    block_indices: tuple[int, ...] = (8, 9, 10),
) -> None:
    """Generate effective-gain vs σ_c scatter — 3-panel, late blocks."""
    from src.model import load_vit
    from src.utils import get_device, seed_everything

    seed_everything(42)
    device = get_device()
    model, _ = load_vit(device)

    fc1_weights = {}
    for bidx in range(12):
        fc1_weights[bidx] = (
            model.blocks[bidx].mlp.fc1.weight.detach().cpu().numpy()
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gains: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for bidx in block_indices:
        ln_gamma = np.array(
            raw_stats[f"blocks.{bidx}/post_layernorm_2"]["layernorm_gamma"],
            dtype=np.float64,
        )
        pc_std = np.array(
            raw_stats[f"blocks.{bidx}/pre_gelu"]["per_channel_std"],
            dtype=np.float64,
        )
        weighted = fc1_weights[bidx] * ln_gamma[np.newaxis, :]
        gains.append(np.linalg.norm(weighted, axis=1))
        stds.append(pc_std)

    from src.plotting_poster import plot_effective_gain_scatter
    plot_effective_gain_scatter(
        gains, stds, output_path,
        block_indices=list(block_indices),
    )


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
    results_mean_only: list[AblationResult] | None = None,
    results_var_only: list[AblationResult] | None = None,
) -> list[Path]:
    """Generate all Phase 2 poster plots (comparison and overlay).

    Parameters
    ----------
    results_mean_only:
        Optional separate results for the mean_only ablation mode.
        If not provided, mean_only rows are looked up in ``results_b``
        (CSV-file mode where all --csv-b files are merged).
    results_var_only:
        Optional separate results for the var_only ablation mode.
    """
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
    # If any mode's data is missing, the waterfall chart is skipped.
    def _acc_at_k(results, k, granularity, mode="outlier"):
        subset = [r for r in results
                  if r.site == "pre_gelu" and not r.is_random
                  and r.sigma_threshold == k
                  and r.granularity == granularity
                  and r.ablation_mode == mode]
        return float(np.mean([r.top1_accuracy for r in subset])) if subset else 0.0

    # Verify baselines are consistent.
    baseline_a = pre_gelu_a[0].baseline_top1 if pre_gelu_a else None
    baseline_b = pre_gelu_b[0].baseline_top1 if pre_gelu_b else None
    if baseline_a is None or baseline_b is None:
        logger.warning("Skipping waterfall: missing baseline data.")
        return written
    if not np.isclose(baseline_a, baseline_b):
        logger.error(
            f"Baseline mismatch: A={baseline_a:.2f}%, B={baseline_b:.2f}%. "
            "Cannot generate waterfall."
        )
        return written
    baseline = baseline_a

    global_k3 = _acc_at_k(results_a, 3.0, "global")
    pc_outlier_k3 = _acc_at_k(results_b, 3.0, "per_channel", mode="outlier")

    # Look up mean_only and var_only: prefer the dedicated result lists
    # (multi-seed directory mode), otherwise fall back to results_b
    # (CSV-file mode where all --csv-b files are merged).
    mean_source = results_mean_only if results_mean_only else results_b
    var_source = results_var_only if results_var_only else results_b
    mean_only_k3 = _acc_at_k(mean_source, 3.0, "per_channel", mode="mean_only")
    var_only_k3 = _acc_at_k(var_source, 3.0, "per_channel", mode="var_only")

    if global_k3 > 0 and pc_outlier_k3 > 0 and mean_only_k3 > 0 and var_only_k3 > 0:
        p = output_dir / "poster_ablation_waterfall.png"
        plot_ablation_comparison(
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
    # --csv-a: single CSV file (used by regenerate_all.sh).
    parser.add_argument(
        "--csv-a", type=Path, default=None,
        help="Path to a single global ablation_results.csv.",
    )
    # --csv-b: may be repeated for outlier, mean_only, and var_only CSVs.
    parser.add_argument(
        "--csv-b", type=Path, action="append", default=None,
        help=(
            "Path to a per-channel ablation_results.csv.  Repeat for each "
            "ablation mode (outlier, mean_only, var_only)."
        ),
    )
    # --phase2-dir-a / --phase2-dir-b: multi-seed directory mode.
    parser.add_argument(
        "--phase2-dir-a", type=Path, default=None,
        help="Root directory for multi-seed global run (contains seed_*/).",
    )
    parser.add_argument(
        "--phase2-dir-b", type=Path, default=None,
        help="Root directory for multi-seed per-channel outlier run.",
    )
    parser.add_argument(
        "--phase2-dir-mean-only", type=Path, default=None,
        help="Root directory for multi-seed per-channel mean_only run.",
    )
    parser.add_argument(
        "--phase2-dir-var-only", type=Path, default=None,
        help="Root directory for multi-seed per-channel var_only run.",
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

    # Determine which input mode we're in.
    has_csv_mode = args.csv_a is not None
    has_dir_mode = (
        args.phase2_dir_a is not None
        or args.phase2_dir_b is not None
        or args.phase2_dir_mean_only is not None
        or args.phase2_dir_var_only is not None
    )

    if args.phase1_json is None and not has_csv_mode and not has_dir_mode:
        logger.error(
            "No input files specified.  Provide --phase1-json and/or "
            "--csv-a/--csv-b or --phase2-dir-a/--phase2-dir-b."
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

    # --- Phase 2: CSV-file mode (used by regenerate_all.sh) ---
    if has_csv_mode:
        logger.info("=== Phase 2 poster plots (CSV-file mode) ===")
        csv_b_list: list[Path] = args.csv_b if args.csv_b else []

        if not csv_b_list:
            logger.error("--csv-b is required when --csv-a is provided.")
            sys.exit(1)

        results_a = _load_ablation_results([args.csv_a])
        # Merge all --csv-b files into one results list so that _acc_at_k
        # can find outlier, mean_only, and var_only rows.
        results_b = _load_ablation_results(csv_b_list)

        total_written.extend(
            _generate_phase2_poster_plots(
                results_a, results_b, args.output_dir,
                label_a=args.label_a, label_b=args.label_b,
                stats=stats,
                histogram_data_dir=args.histogram_data_dir,
            ),
        )

    # --- Phase 2: multi-seed directory mode ---
    elif has_dir_mode:
        logger.info("=== Phase 2 poster plots (multi-seed directory mode) ===")

        if args.phase2_dir_a is None or args.phase2_dir_b is None:
            logger.error(
                "Both --phase2-dir-a and --phase2-dir-b are required "
                "for multi-seed mode."
            )
            sys.exit(1)

        paths_a = sorted(args.phase2_dir_a.glob("**/ablation_results.csv"))
        paths_b = sorted(args.phase2_dir_b.glob("**/ablation_results.csv"))

        if not paths_a or not paths_b:
            logger.error(
                "Could not find ablation_results.csv in the provided directories."
            )
            sys.exit(1)

        results_a = _load_ablation_results(paths_a)
        results_b_outlier = _load_ablation_results(paths_b)

        # Load mean_only and var_only if provided.
        results_b_mean_only: list[AblationResult] = []
        results_b_var_only: list[AblationResult] = []
        if args.phase2_dir_mean_only is not None:
            paths_mean = sorted(
                args.phase2_dir_mean_only.glob("**/ablation_results.csv")
            )
            if paths_mean:
                results_b_mean_only = _load_ablation_results(paths_mean)
        if args.phase2_dir_var_only is not None:
            paths_var = sorted(
                args.phase2_dir_var_only.glob("**/ablation_results.csv")
            )
            if paths_var:
                results_b_var_only = _load_ablation_results(paths_var)

        total_written.extend(
            _generate_phase2_poster_plots(
                results_a, results_b_outlier, args.output_dir,
                label_a=args.label_a, label_b=args.label_b,
                stats=stats,
                histogram_data_dir=args.histogram_data_dir,
                results_mean_only=results_b_mean_only,
                results_var_only=results_b_var_only,
            ),
        )

    logger.info("Generated %d poster plots in %s", len(total_written), args.output_dir)
    for p in total_written:
        logger.info("  %s", p.name)


if __name__ == "__main__":
    main()
