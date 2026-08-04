"""Post-hoc analysis of Phase 2 ablation results.

Performs three analyses on existing ablation CSV data (zero GPU cost) and
delegates all plotting to ``src/plotting.py``:

1. **Bootstrap CI on global vs per-channel delta**: Computes 95% bootstrap
   confidence intervals on the accuracy difference between two conditions at
   each sigma threshold.

2. **Effective channels preserved**: Translates %-zeroed into "effective
   channels preserved" per block, comparing two conditions at each sigma
   threshold.

3. **Accuracy degradation per %-zeroed**: Normalises accuracy drop by the
   fraction of elements zeroed — "how much accuracy is lost per unit of
   sparsity?"

Usage:
    python scripts/analyze_ablation_results.py \
        --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
        --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
        --output-dir outputs/ablation-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.plotting import (
    plot_bootstrap_ci_delta,
    plot_degradation_efficiency,
    plot_effective_channels,
)

logger = logging.getLogger(__name__)

# ViT-B/16 MLP hidden dimension.
_D_MLP: int = 3072


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of Phase 2 ablation results.",
    )
    parser.add_argument(
        "--csv-a",
        type=Path,
        default=Path("outputs/phase2-ablation-global-50k/ablation_results.csv"),
        help="Path to first ablation results CSV (e.g. global).",
    )
    parser.add_argument(
        "--csv-b",
        type=Path,
        default=Path("outputs/phase2-ablation-per-channel-50k/ablation_results.csv"),
        help="Path to second ablation results CSV (e.g. per-channel).",
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
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation-analysis"),
        help="Directory for output plots and JSON.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for CI estimation.",
    )
    return parser.parse_args()


def _load_accuracy_by_k(
    path: Path, site: str = "pre_gelu",
) -> dict[float, float]:
    """Load top-1 accuracy per sigma threshold for a given site.

    Returns mapping from sigma_k (float) to top1_accuracy (float).
    Only includes non-random rows.
    """
    acc: dict[float, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["site"] != site:
                continue
            if row["is_random"] == "True":
                continue
            k = float(row["sigma_threshold"])
            acc[k] = float(row["top1_accuracy"])
    return acc


def _load_pct_zeroed_by_block(
    path: Path, sigma_k: float, site: str = "pre_gelu",
) -> dict[str, float]:
    """Load per-layer %-zeroed for a given sigma threshold.

    Returns mapping from site_identifier to pct_zeroed.
    """
    pct: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["site"] != site:
                continue
            if row["is_random"] == "True":
                continue
            if float(row["sigma_threshold"]) != sigma_k:
                continue
            pct[row["site_identifier"]] = float(row["pct_zeroed"])
    return pct


# ---------------------------------------------------------------------------
# Analysis 1: Bootstrap CI on delta
# ---------------------------------------------------------------------------


def _bootstrap_ci_delta(
    acc_a: dict[float, float],
    acc_b: dict[float, float],
    n_images: int = 50000,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[float, dict[str, float]]:
    """Estimate 95% bootstrap CI for (B − A) accuracy delta.

    Uses vectorised numpy sampling: each bootstrap resample draws from a
    Binomial(n_images, p) distribution.  For n_images=50000 and
    n_bootstrap=10000, this runs in < 1 second.

    Parameters
    ----------
    acc_a:
        Mapping from sigma_k to condition A top-1 accuracy (%).
    acc_b:
        Mapping from sigma_k to condition B top-1 accuracy (%).
    n_images:
        Number of images evaluated.
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[float, dict[str, float]]
        Mapping from sigma_k to {"delta_point_estimate", "delta_ci_low_pct",
        "delta_ci_high_pct", "acc_a", "acc_b"}.
    """
    rng = np.random.default_rng(seed)
    results: dict[float, dict[str, float]] = {}

    for k in sorted(acc_a.keys()):
        p_a = acc_a[k] / 100.0
        p_b = acc_b.get(k, p_a) / 100.0

        correct_a = rng.binomial(n_images, p_a, size=n_bootstrap)
        correct_b = rng.binomial(n_images, p_b, size=n_bootstrap)
        acc_a_boot = 100.0 * correct_a / n_images
        acc_b_boot = 100.0 * correct_b / n_images
        deltas = acc_b_boot - acc_a_boot

        ci_low = float(np.percentile(deltas, 2.5))
        ci_high = float(np.percentile(deltas, 97.5))
        mean_delta = float(np.mean(deltas))

        results[k] = {
            "delta_point_estimate": round(acc_b.get(k, acc_a[k]) - acc_a[k], 4),
            "delta_mean_pct": round(mean_delta, 4),
            "delta_ci_low_pct": round(ci_low, 4),
            "delta_ci_high_pct": round(ci_high, 4),
            "acc_a": acc_a[k],
            "acc_b": acc_b.get(k, acc_a[k]),
        }

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Effective channels preserved
# ---------------------------------------------------------------------------


def _effective_channels(
    pct_a: dict[str, float],
    pct_b: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute effective channels preserved per block.

    effective_channels = (1 - pct_zeroed/100) * D_MLP

    Returns mapping from site_identifier to:
        {"channels_a", "channels_b", "delta_channels"}
    """
    results: dict[str, dict[str, float]] = {}
    for sid in sorted(pct_a.keys()):
        a_pct = pct_a[sid]
        b_pct = pct_b.get(sid, a_pct)
        a_ch = (1.0 - a_pct / 100.0) * _D_MLP
        b_ch = (1.0 - b_pct / 100.0) * _D_MLP
        results[sid] = {
            "channels_a": round(a_ch, 1),
            "channels_b": round(b_ch, 1),
            "delta_channels": round(b_ch - a_ch, 1),
            "pct_zeroed_a": a_pct,
            "pct_zeroed_b": b_pct,
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Accuracy degradation per unit sparsity
# ---------------------------------------------------------------------------


def _degradation_per_sparsity(
    acc_a: dict[float, float],
    acc_b: dict[float, float],
    pct_a: dict[str, float],
    pct_b: dict[str, float],
    baseline_top1: float,
) -> dict[float, dict[str, float]]:
    """Compute accuracy degradation normalised by mean %-zeroed.

    degradation_per_pct = (baseline - accuracy) / mean_pct_zeroed

    Returns mapping from sigma_k to:
        {"degradation_a_per_pct", "degradation_b_per_pct", "efficiency_ratio"}
    """
    results: dict[float, dict[str, float]] = {}
    for k in sorted(acc_a.keys()):
        a_mean_pct = sum(v for v in pct_a.values()) / max(len(pct_a), 1)
        b_mean_pct = sum(v for v in pct_b.values()) / max(len(pct_b), 1)

        a_degrad = (baseline_top1 - acc_a[k]) / a_mean_pct if a_mean_pct > 0 else 0.0
        b_degrad = (baseline_top1 - acc_b.get(k, baseline_top1)) / b_mean_pct if b_mean_pct > 0 else 0.0

        results[k] = {
            "mean_pct_zeroed_a": round(a_mean_pct, 4),
            "mean_pct_zeroed_b": round(b_mean_pct, 4),
            "degradation_a_per_pct": round(a_degrad, 4),
            "degradation_b_per_pct": round(b_degrad, 4),
            "efficiency_ratio": round(a_degrad / b_degrad, 4) if b_degrad > 0 else float("inf"),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data.
    acc_a = _load_accuracy_by_k(args.csv_a)
    acc_b = _load_accuracy_by_k(args.csv_b)

    # Determine baseline from CSV A.
    with open(args.csv_a, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        baseline_top1 = float(first_row["baseline_top1"])

    logger.info("Baseline top-1: %.2f%%", baseline_top1)
    logger.info("%s acc: %s", args.label_a, {f"k={k}": f"{v:.2f}%" for k, v in sorted(acc_a.items())})
    logger.info("%s acc: %s", args.label_b, {f"k={k}": f"{v:.2f}%" for k, v in sorted(acc_b.items())})

    # --- Analysis 1: Bootstrap CI ---
    logger.info("=== Bootstrap CI on delta (%s − %s) ===", args.label_b, args.label_a)
    ci_results = _bootstrap_ci_delta(
        acc_a, acc_b,
        n_bootstrap=args.num_bootstrap,
    )
    for k in sorted(ci_results.keys()):
        r = ci_results[k]
        logger.info(
            "  k=%.1f: delta=%.2f%%  [95%% CI: %.2f%%, %.2f%%]",
            k, r["delta_point_estimate"], r["delta_ci_low_pct"], r["delta_ci_high_pct"],
        )

    plot_bootstrap_ci_delta(ci_results, args.output_dir / "bootstrap_ci_delta.png")

    # --- Analysis 2: Effective channels ---
    logger.info("=== Effective channels preserved ===")
    for k in sorted(acc_a.keys()):
        pct_a = _load_pct_zeroed_by_block(args.csv_a, k)
        pct_b = _load_pct_zeroed_by_block(args.csv_b, k)
        if not pct_b:
            continue
        channels = _effective_channels(pct_a, pct_b)

        total_a = sum(ch["channels_a"] for ch in channels.values())
        total_b = sum(ch["channels_b"] for ch in channels.values())
        logger.info(
            "  k=%.1f: %s total=%.0f, %s total=%.0f, delta=%.0f",
            k, args.label_a, total_a, args.label_b, total_b, total_b - total_a,
        )

        # Map to legacy keys for the plotting function.
        channels_for_plot: dict[str, dict[str, float]] = {
            sid: {
                "global_channels": ch["channels_a"],
                "pc_channels": ch["channels_b"],
            }
            for sid, ch in channels.items()
        }
        plot_effective_channels(
            channels_for_plot, k,
            args.output_dir / f"effective_channels_k{k:.1f}.png",
        )

    # --- Analysis 3: Degradation per sparsity ---
    logger.info("=== Accuracy degradation per unit sparsity ===")
    for k in sorted(acc_a.keys()):
        pct_a = _load_pct_zeroed_by_block(args.csv_a, k)
        pct_b = _load_pct_zeroed_by_block(args.csv_b, k)
        if not pct_b:
            continue
        deg_results = _degradation_per_sparsity(
            acc_a, acc_b, pct_a, pct_b, baseline_top1,
        )
        for k2, r in deg_results.items():
            logger.info(
                "  k=%.1f: %s=%.4f pp/%%, %s=%.4f pp/%%, efficiency=%.2fx",
                k2,
                args.label_a,
                r["degradation_a_per_pct"],
                args.label_b,
                r["degradation_b_per_pct"],
                r["efficiency_ratio"],
            )

        plot_degradation_efficiency(
            deg_results, args.output_dir / "degradation_efficiency.png",
        )
        break  # Only one plot needed (same structure across k values)

    # Save all results as JSON.
    summary: dict[str, object] = {
        "baseline_top1": baseline_top1,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "num_bootstrap": args.num_bootstrap,
        "bootstrap_ci": {
            str(k): v for k, v in ci_results.items()
        },
    }
    # Add effective channels for k=3.
    pct_a_3 = _load_pct_zeroed_by_block(args.csv_a, 3.0)
    pct_b_3 = _load_pct_zeroed_by_block(args.csv_b, 3.0)
    if pct_b_3:
        channels_3 = _effective_channels(pct_a_3, pct_b_3)
        summary["effective_channels_k3"] = {
            str(k): v for k, v in channels_3.items()
        }

    json_path = args.output_dir / "ablation_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved full analysis to %s", json_path)


if __name__ == "__main__":
    main()