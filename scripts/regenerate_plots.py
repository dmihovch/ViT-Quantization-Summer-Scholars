"""Regenerate all plots from existing data files without re-running experiments.

Reads ``profiling_result.json`` (Phase 1) and/or ``ablation_results.csv``
(Phase 2) and regenerates every plot.  This is the single entry point for
visualisation — experiment orchestrators produce data; this script produces
plots.

Usage
-----
    # Phase 1 only:
    python scripts/regenerate_plots.py \
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/phase1-profiling/seed_42/

    # Phase 2 only (single CSV):
    python scripts/regenerate_plots.py \
        --phase2-csv outputs/phase2-ablation/ablation_results.csv \
        --output-dir outputs/phase2-ablation/

    # Phase 2 comparison (two CSVs, e.g. global vs per-channel):
    python scripts/regenerate_plots.py \
        --phase2-csv-a outputs/phase2-ablation-global/ablation_results.csv \
        --phase2-csv-b outputs/phase2-ablation-per-channel/ablation_results.csv \
        --output-dir outputs/phase2-comparison/

    # Both phases:
    python scripts/regenerate_plots.py \
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \
        --phase2-csv-a outputs/phase2-ablation-global/ablation_results.csv \
        --phase2-csv-b outputs/phase2-ablation-per-channel/ablation_results.csv \
        --output-dir outputs/all-plots/

    # Histograms (requires model + GPU):
    python scripts/regenerate_plots.py \
        --phase1-json outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/phase1-profiling/seed_42/ \
        --histograms --data-dir data
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
from src.plotting import (
    plot_ablation_mode_comparison,
    plot_accuracy_comparison,
    plot_accuracy_vs_threshold,
    plot_activation_histogram,
    plot_attention_entropy_heatmap,
    plot_entropy_delta_heatmap,
    plot_kurtosis_heatmap,
    plot_ln2_amplification_ratio,
    plot_outlier_fraction_heatmap,
    plot_pct_zeroed_per_layer,
    plot_per_channel_mean_heatmap,
    plot_per_channel_std_heatmap,
)
from src.profiler import (
    LayerStats,
    SiteId,
    histogram_profile_vit,
    load_profiling_result,
)
from src.utils import ensure_dir, get_device, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Sigma thresholds used for outlier fraction plots.
_OUTLIER_SIGMAS: tuple[float, ...] = (3.0, 4.0, 6.0)


# ===========================================================================
# Phase 1 plot generators
# ===========================================================================


def _regenerate_phase1_plots(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate all Phase 1 plots from profiling stats.

    Parameters
    ----------
    stats:
        Mapping from site_identifier to :class:`LayerStats`.
    output_dir:
        Directory where PNGs are written.
    """
    _regenerate_per_channel_heatmaps(stats, output_dir)
    _regenerate_entropy_heatmaps(stats, output_dir)
    _regenerate_kurtosis_heatmap(stats, output_dir)
    _regenerate_outlier_fraction_heatmaps(stats, output_dir)
    _regenerate_ln2_amplification(stats, output_dir)


def _regenerate_per_channel_heatmaps(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate per-channel σ and μ heatmaps."""
    # --- Per-channel σ ---
    per_channel_std: dict[str, list[float]] = {
        key: s.per_channel_std
        for key, s in stats.items()
        if s.per_channel_std is not None
    }
    if per_channel_std:
        by_dim: dict[int, dict[str, list[float]]] = {}
        for key, stds in per_channel_std.items():
            d = len(stds)
            by_dim.setdefault(d, {})[key] = stds

        for d, group in sorted(by_dim.items()):
            suffix = f"_d{d}" if len(by_dim) > 1 else ""
            out_path = output_dir / f"per_channel_std_heatmap{suffix}.png"
            plot_per_channel_std_heatmap(group, out_path)
            logger.info("Per-channel σ heatmap (D=%d) written to %s", d, out_path)
    else:
        logger.warning("No per_channel_std data found; skipping σ heatmaps.")

    # --- Per-channel μ ---
    per_channel_mean: dict[str, list[float]] = {
        key: s.per_channel_mean
        for key, s in stats.items()
        if s.per_channel_mean is not None
    }
    if per_channel_mean:
        by_dim_m: dict[int, dict[str, list[float]]] = {}
        for key, means in per_channel_mean.items():
            d = len(means)
            by_dim_m.setdefault(d, {})[key] = means

        for d, group in sorted(by_dim_m.items()):
            suffix = f"_d{d}" if len(by_dim_m) > 1 else ""
            out_path = output_dir / f"per_channel_mean_heatmap{suffix}.png"
            plot_per_channel_mean_heatmap(group, out_path)
            logger.info("Per-channel μ heatmap (D=%d) written to %s", d, out_path)
    else:
        logger.warning("No per_channel_mean data found; skipping μ heatmaps.")


def _regenerate_entropy_heatmaps(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate attention entropy heatmaps for CLS and patch queries."""
    cls_entropies: dict[str, list[float]] = {
        key: s.attention_entropy_cls
        for key, s in stats.items()
        if s.attention_entropy_cls is not None
    }
    patch_entropies: dict[str, list[float]] = {
        key: s.attention_entropy_patches
        for key, s in stats.items()
        if s.attention_entropy_patches is not None
    }
    if cls_entropies:
        plot_attention_entropy_heatmap(
            cls_entropies,
            output_dir / "attention_entropy_cls_heatmap.png",
            title="CLS query attention entropy per head (nats)",
        )
        logger.info("CLS entropy heatmap written.")
    if patch_entropies:
        plot_attention_entropy_heatmap(
            patch_entropies,
            output_dir / "attention_entropy_patches_heatmap.png",
            title="Patch query mean attention entropy per head (nats)",
        )
        logger.info("Patch entropy heatmap written.")
    if not cls_entropies and not patch_entropies:
        logger.warning("No attention entropy data found; skipping entropy heatmaps.")


def _regenerate_kurtosis_heatmap(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate per-site kurtosis heatmap."""
    kurtosis: dict[str, float] = {
        key: s.kurtosis for key, s in stats.items()
    }
    if kurtosis:
        plot_kurtosis_heatmap(kurtosis, output_dir / "kurtosis_heatmap.png")
        logger.info("Kurtosis heatmap written.")
    else:
        logger.warning("No kurtosis data found; skipping.")


def _regenerate_outlier_fraction_heatmaps(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate outlier fraction heatmaps at 3σ, 4σ, 6σ."""
    outlier_fracs: dict[str, dict[str, float]] = {
        key: s.outlier_fractions for key, s in stats.items()
        if s.outlier_fractions
    }
    if not outlier_fracs:
        logger.warning("No outlier fraction data found; skipping.")
        return

    for k in _OUTLIER_SIGMAS:
        sigma_key = f"{k}_sigma"
        plot_outlier_fraction_heatmap(
            outlier_fracs, sigma_key,
            output_dir / f"outlier_fraction_{sigma_key}_heatmap.png",
        )
    logger.info("Outlier fraction heatmaps written.")


def _regenerate_ln2_amplification(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate LN2 amplification ratio bar chart."""
    ratios: dict[str, float] = {
        key: s.ln2_amplification_ratio
        for key, s in stats.items()
        if s.ln2_amplification_ratio is not None
    }
    if ratios:
        plot_ln2_amplification_ratio(ratios, output_dir / "ln2_amplification_ratio.png")
        logger.info("LN2 amplification ratio plot written.")
    else:
        logger.warning("No LN2 amplification ratio data found; skipping.")


def _regenerate_histograms(
    data_dir: Path,
    batch_size: int,
    block_indices: tuple[int, ...],
    output_dir: Path,
) -> None:
    """Regenerate activation histograms (requires model load + GPU)."""
    device = get_device()
    model, transform = load_vit(device)
    wrapped = NNsight(model)

    loader = build_val_loader(
        data_dir, transform, batch_size, None, device, shuffle=True,
    )
    images, _ = next(iter(loader))

    with torch.no_grad():
        raw_tensors = histogram_profile_vit(
            wrapped, images.to(device), block_indices,
        )

    hist_dir = output_dir / "histograms"
    ensure_dir(hist_dir)
    for key, tensor in raw_tensors.items():
        activations = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        safe_key = key.replace("/", "_").replace(".", "_")
        plot_activation_histogram(
            activations=activations,
            layer_name=key,
            output_path=hist_dir / f"{safe_key}.png",
            log_scale=True,
        )
    logger.info("Wrote %d histogram PNGs to %s", len(raw_tensors), hist_dir)


# ===========================================================================
# Phase 2 plot generators
# ===========================================================================


def _load_ablation_results(path: Path) -> list[AblationResult]:
    """Load ablation results from a CSV file.

    Parameters
    ----------
    path:
        Path to ``ablation_results.csv``.

    Returns
    -------
    list[AblationResult]
        Deserialised results.
    """
    import json as _json

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
                cls_entropy=_json.loads(row.get("cls_entropy", "[]")),
                patch_entropy=_json.loads(row.get("patch_entropy", "[]")),
                baseline_cls_entropy=_json.loads(row.get("baseline_cls_entropy", "[]")),
                baseline_patch_entropy=_json.loads(row.get("baseline_patch_entropy", "[]")),
            ))
    logger.info("Loaded %d ablation results from %s", len(results), path)
    return results


def _regenerate_phase2_single(
    results: list[AblationResult],
    output_dir: Path,
) -> None:
    """Regenerate plots for a single Phase 2 run.

    Parameters
    ----------
    results:
        All ablation results from one CSV.
    output_dir:
        Directory where PNGs are written.
    """
    sites = sorted({r.site for r in results})
    sigma_ks = sorted({r.sigma_threshold for r in results})

    for site in sites:
        site_results = [r for r in results if r.site == site and not r.is_random]
        if not site_results:
            continue

        # Accuracy vs threshold.
        plot_accuracy_vs_threshold(
            site_results,
            output_dir / f"accuracy_vs_threshold_{site}.png",
        )

        # Per-layer %-zeroed at each k.
        for k in sigma_ks:
            plot_pct_zeroed_per_layer(
                site_results, k,
                output_dir / f"pct_zeroed_{site}_k{k:.1f}.png",
            )

        # Random control comparison.
        random_results = [r for r in results if r.site == site and r.is_random]
        if random_results:
            plot_accuracy_vs_threshold(
                random_results,
                output_dir / f"accuracy_vs_threshold_{site}_random.png",
            )

        # Entropy delta (pre_softmax only).
        if site == "pre_softmax":
            entropy_deltas: dict[str, dict[str, float]] = {}
            for r in site_results:
                if r.cls_entropy and r.baseline_cls_entropy:
                    cls_delta = sum(a - b for a, b in zip(r.cls_entropy, r.baseline_cls_entropy)) / len(r.cls_entropy)
                    patch_delta = sum(a - b for a, b in zip(r.patch_entropy, r.baseline_patch_entropy)) / len(r.patch_entropy) if r.patch_entropy and r.baseline_patch_entropy else 0.0
                    entropy_deltas[r.site_identifier] = {
                        "mean_cls_delta": cls_delta,
                        "mean_patch_delta": patch_delta,
                    }
            if entropy_deltas:
                plot_entropy_delta_heatmap(
                    entropy_deltas,
                    output_dir / f"entropy_delta_cls_{site}.png",
                    delta_key="mean_cls_delta",
                    title=f"CLS entropy delta after {site} ablation",
                )
                plot_entropy_delta_heatmap(
                    entropy_deltas,
                    output_dir / f"entropy_delta_patch_{site}.png",
                    delta_key="mean_patch_delta",
                    title=f"Patch entropy delta after {site} ablation",
                )

    # Ablation mode comparison (if multiple granularity/mode values exist).
    granularities = sorted({r.granularity for r in results if not r.is_random})
    if len(granularities) > 1:
        for site in sites:
            for k in sigma_ks:
                mode_results: dict[str, list[AblationResult]] = {}
                for r in results:
                    if r.site == site and not r.is_random and r.sigma_threshold == k:
                        label = f"{r.granularity}"
                        mode_results.setdefault(label, []).append(r)
                if len(mode_results) > 1:
                    plot_ablation_mode_comparison(
                        mode_results,
                        output_dir / f"ablation_mode_{site}_k{k:.1f}.png",
                        sigma_k=k,
                    )

    logger.info("Phase 2 single-run plots written to %s", output_dir)


def _regenerate_phase2_comparison(
    results_a: list[AblationResult],
    results_b: list[AblationResult],
    output_dir: Path,
    label_a: str = "Global",
    label_b: str = "Per-channel",
) -> None:
    """Regenerate comparison plots between two Phase 2 runs.

    Parameters
    ----------
    results_a:
        First set of ablation results.
    results_b:
        Second set of ablation results.
    output_dir:
        Directory where PNGs are written.
    label_a:
        Legend label for the first condition.
    label_b:
        Legend label for the second condition.
    """
    sites_a = {r.site for r in results_a if not r.is_random}
    sites_b = {r.site for r in results_b if not r.is_random}
    common_sites = sites_a & sites_b

    for site in sorted(common_sites):
        a_site = [r for r in results_a if r.site == site and not r.is_random]
        b_site = [r for r in results_b if r.site == site and not r.is_random]
        if not a_site or not b_site:
            continue

        plot_accuracy_comparison(
            a_site, b_site,
            output_dir / f"accuracy_comparison_{site}.png",
            label_a=label_a, label_b=label_b,
        )

    logger.info("Phase 2 comparison plots written to %s", output_dir)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all plots from existing data files.",
    )
    # Phase 1
    parser.add_argument(
        "--phase1-json", type=Path, default=None,
        help="Path to profiling_result.json (Phase 1).",
    )
    # Phase 2
    parser.add_argument(
        "--phase2-csv", type=Path, default=None,
        help="Path to a single ablation_results.csv (Phase 2 single-run plots).",
    )
    parser.add_argument(
        "--phase2-csv-a", type=Path, default=None,
        help="Path to first ablation_results.csv for comparison (e.g. global).",
    )
    parser.add_argument(
        "--phase2-csv-b", type=Path, default=None,
        help="Path to second ablation_results.csv for comparison (e.g. per-channel).",
    )
    parser.add_argument(
        "--label-a", type=str, default="Global",
        help="Legend label for CSV A (default: 'Global').",
    )
    parser.add_argument(
        "--label-b", type=str, default="Per-channel",
        help="Legend label for CSV B (default: 'Per-channel').",
    )
    # Output
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write all regenerated plots.",
    )
    # Histograms (requires model + GPU)
    parser.add_argument(
        "--histograms", action="store_true",
        help="Also regenerate activation histograms (requires model + GPU).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="ImageNet val directory (only needed with --histograms).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for histogram pass.",
    )
    parser.add_argument(
        "--all-blocks", action="store_true",
        help="Generate histograms for all 12 blocks (default: 0, 5, 11 only).",
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

    # --- Phase 1 ---
    if args.phase1_json is not None:
        logger.info("=== Phase 1 plots ===")
        result = load_profiling_result(args.phase1_json)
        logger.info("Loaded %d sites from %s", len(result.stats), args.phase1_json)
        _regenerate_phase1_plots(result.stats, args.output_dir)

        if args.histograms:
            block_indices = tuple(range(12)) if args.all_blocks else (0, 5, 11)
            _regenerate_histograms(
                args.data_dir, args.batch_size, block_indices, args.output_dir,
            )

    # --- Phase 2 single ---
    if args.phase2_csv is not None:
        logger.info("=== Phase 2 single-run plots ===")
        results = _load_ablation_results(args.phase2_csv)
        _regenerate_phase2_single(results, args.output_dir)

    # --- Phase 2 comparison ---
    if args.phase2_csv_a is not None and args.phase2_csv_b is not None:
        logger.info("=== Phase 2 comparison plots ===")
        results_a = _load_ablation_results(args.phase2_csv_a)
        results_b = _load_ablation_results(args.phase2_csv_b)
        _regenerate_phase2_comparison(
            results_a, results_b, args.output_dir,
            label_a=args.label_a, label_b=args.label_b,
        )

    if args.phase1_json is None and args.phase2_csv is None and args.phase2_csv_a is None:
        logger.warning(
            "No input files specified.  Use --phase1-json, --phase2-csv, "
            "and/or --phase2-csv-a/--phase2-csv-b."
        )

    logger.info("Done. Plots in %s", args.output_dir)


if __name__ == "__main__":
    main()