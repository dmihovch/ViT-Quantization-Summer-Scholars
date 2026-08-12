#!/usr/bin/env python
"""
Generate the final, publication-quality figures for the ViT outlier study.

This script orchestrates the generation of all plots, including the six key
figures for the poster narrative, from the 5-seed full experimental run.
"""

import argparse
import collections
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.plotting_poster import (
    _PALETTE,
    _poster_style,
    plot_activation_distribution_overlay,
    plot_outlier_site_grid,
    plot_per_channel_sigma_line,
    plot_ablation_comparison,
)
from src.profiler import load_profiling_result
from src.utils import ensure_dir
from src.ablation import AblationResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
                    seed=int(row.get("seed", 0)),
                    is_random=row.get("is_random", "False") == "True",
                    granularity=row.get("granularity", "global"),
                    ablation_mode=row.get("ablation_mode", "outlier"),
                ))
    logger.info("Loaded %d total ablation results from %d files", len(results), len(paths))
    return results

def plot_accuracy_cost_vs_sparsity(
    results_a: list[AblationResult],
    results_b: list[AblationResult],
    output_path: Path,
    label_a: str = "Global",
    label_b: str = "Per-channel",
) -> None:
    """Plot accuracy drop vs. induced sparsity as a line plot."""
    
    def aggregate_by_k(results: list[AblationResult]):
        grouped = collections.defaultdict(lambda: {'accs': [], 'sparsities': []})
        for r in results:
            if r.site == "pre_gelu" and not r.is_random:
                grouped[r.sigma_threshold]['accs'].append(r.top1_accuracy)
                grouped[r.sigma_threshold]['sparsities'].append(r.pct_zeroed)
        
        agg = {}
        for k, data in sorted(grouped.items()):
            agg[k] = {
                'mean_acc': np.mean(data['accs']),
                'mean_sparsity': np.mean(data['sparsities']),
            }
        return agg

    agg_a = aggregate_by_k(results_a)
    agg_b = aggregate_by_k(results_b)
    
    baseline = results_a[0].baseline_top1 if results_a else 85.03

    sparsity_a = [v['mean_sparsity'] for v in agg_a.values()]
    cost_a = [baseline - v['mean_acc'] for v in agg_a.values()]
    sparsity_b = [v['mean_sparsity'] for v in agg_b.values()]
    cost_b = [baseline - v['mean_acc'] for v in agg_b.values()]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    _poster_style(ax, fontsize=16)

    ax.plot(sparsity_a, cost_a, 'o-', color=_PALETTE["coral"], label=label_a, linewidth=2.5, markersize=8)
    ax.plot(sparsity_b, cost_b, 'o-', color=_PALETTE["teal"], label=label_b, linewidth=2.5, markersize=8)

    # Annotate each point with its sigma threshold k
    for k, sx, cx in zip(agg_a.keys(), sparsity_a, cost_a):
        ax.annotate(f"{int(k)}σ", (sx, cx), textcoords="offset points",
                    xytext=(8, -4), fontsize=14, color=_PALETTE["coral"],
                    ha="left", va="top")
        for k, sx, cx in zip(agg_b.keys(), sparsity_b, cost_b):
            ax.annotate(f"{int(k)}σ", (sx, cx), textcoords="offset points",
                        xytext=(8, -4), fontsize=14, color=_PALETTE["teal"],
                    ha="left", va="top")

    ax.set_xlabel("Induced Activation Sparsity (%)", fontsize=16)
    ax.set_ylabel("Accuracy Drop (pp)", fontsize=16)
    ax.set_title("Accuracy Cost of Sparsification", fontsize=18, fontweight="bold")
    ax.legend(fontsize=16, loc="upper left")
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#CCCCCC')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy cost vs sparsity line plot to %s", output_path)





def plot_main_accuracy_bars(
    results_a: list[AblationResult],
    results_b: list[AblationResult],
    output_path: Path,
    label_a: str = "Global",
    label_b: str = "Per-channel",
) -> None:
    """The main grouped bar chart for accuracy comparison."""
    sigma_ks = (3.0, 4.0, 6.0)
    
    def _acc_at_k(results: list[AblationResult], k: float):
        accs = [r.top1_accuracy for r in results if r.sigma_threshold == k and not r.is_random and r.site == 'pre_gelu']
        return (np.mean(accs), np.std(accs)) if accs else (0, 0)

    baseline = results_a[0].baseline_top1 if results_a else 85.03
    
    means_a = [_acc_at_k(results_a, k)[0] for k in sigma_ks]
    stds_a = [_acc_at_k(results_a, k)[1] for k in sigma_ks]
    means_b = [_acc_at_k(results_b, k)[0] for k in sigma_ks]
    stds_b = [_acc_at_k(results_b, k)[1] for k in sigma_ks]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    _poster_style(ax, fontsize=16)
    
    x = np.arange(len(sigma_ks))
    width = 0.35
    
    bars1 = ax.bar(x - width / 2, means_a, width, yerr=stds_a, label=label_a, color=_PALETTE["coral"], capsize=5)
    bars2 = ax.bar(x + width / 2, means_b, width, yerr=stds_b, label=label_b, color=_PALETTE["teal"], capsize=5)
    
    ax.bar_label(bars1, fmt='%.2f', padding=3, fontsize=18)
    ax.bar_label(bars2, fmt='%.2f', padding=3, fontsize=18)

    ax.axhline(baseline, color=_PALETTE["gray"], linestyle="--", linewidth=2.0, label=f"Baseline ({baseline:.2f}%)")
    
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy After Outlier Clipping", fontweight="bold", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}σ Threshold" for k in sigma_ks])
    ax.legend(fontsize=16, loc="lower right")
    ax.set_ylim(bottom=min(0, np.min(means_a)-5 if means_a else 0), top=baseline+2)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved main accuracy bar chart to %s", output_path)


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Generate all final report plots.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to 5-seed-full-run directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to the new_plots output directory.")
    parser.add_argument("--run-live-overlay", action="store_true", help="Run the live model pass to generate the activation overlay.")
    parser.add_argument("--imagenet-val-dir", type=Path, default="data/imagenet-val", help="Path to ImageNet validation set for live overlay.")

    args = parser.parse_args()

    data_root = args.input_dir
    global_dir = data_root / "phase2-ablation-global-10k"
    per_channel_dir = data_root / "phase2-ablation-per-channel-10k"
    mean_only_dir = data_root / "phase2-ablation-mean-only-10k"
    var_only_dir = data_root / "phase2-ablation-var-only-10k"
    profiling_dir = data_root / "phase1-profiling"

    output_root = args.output_dir
    poster_dir = output_root
    
    ensure_dir(output_root)

    global_paths = sorted(global_dir.glob("**/ablation_results.csv"))
    pc_paths = sorted(per_channel_dir.glob("**/ablation_results.csv"))
    mean_only_paths = sorted(mean_only_dir.glob("**/ablation_results.csv"))
    var_only_paths = sorted(var_only_dir.glob("**/ablation_results.csv"))
    
    if not global_paths or not pc_paths or not mean_only_paths or not var_only_paths:
        logger.error("Could not find all required ablation_results.csv files in %s", data_root)
        sys.exit(1)

    results_global = _load_ablation_results(global_paths)
    results_pc = _load_ablation_results(pc_paths)
    results_mean_only = _load_ablation_results(mean_only_paths)
    results_var_only = _load_ablation_results(var_only_paths)

    profiling_path = profiling_dir / "seed_42" / "profiling_result.json"
    if not profiling_path.exists():
        logger.error("Could not find profiling_result.json in %s", profiling_dir)
        sys.exit(1)
    profiling_data = load_profiling_result(profiling_path)
    stats = profiling_data.stats

    # --- Generate 6 Key Poster Figures ---
    logger.info("--- Generating Poster Figures ---")

    if args.run_live_overlay:
        logger.info("Generating Figure 1 (Activation Overlay)... Requires GPU and data.")
        from src.model import load_vit
        from src.data_loader import build_val_loader
        from nnsight import NNsight
        from src.profiler import histogram_profile_vit
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, transform = load_vit(device)
        loader = build_val_loader(args.imagenet_val_dir, transform, 64, None, device, shuffle=True)
        images, _ = next(iter(loader))
        with torch.no_grad():
            raw_tensors = histogram_profile_vit(NNsight(model), images.to(device), (10,))
        tensor = raw_tensors['blocks.10/pre_gelu'].detach().cpu().numpy().ravel().astype(np.float32)
        gelu_stats = stats.get('blocks.10/pre_gelu')
        plot_activation_distribution_overlay(
            tensor, "Block 10: pre-GELU Activation Distribution",
            poster_dir / "fig1_activation_overlay.png",
            global_mean=gelu_stats.mean, global_std=gelu_stats.std,
            per_channel_stds=gelu_stats.per_channel_std,
            per_channel_means=gelu_stats.per_channel_mean, sigma_k=3.0
        )
    else:
        logger.warning("Skipping Figure 1 (Activation Overlay). Use --run-live-overlay to generate.")

    pc_stds = {k: s.per_channel_std for k, s in stats.items() if s.per_channel_std}
    plot_per_channel_sigma_line(pc_stds, poster_dir / "fig2_sigma_ridgeline.png", title="Per-Channel Standard Deviation by Block: pre-GELU")

    outlier_fracs = {k: s.outlier_fractions for k, s in stats.items() if s.outlier_fractions}
    plot_outlier_site_grid(outlier_fracs, poster_dir / "fig3_outlier_grid.png", sigma_key="3.0_sigma", title="Outlier Fraction at 3σ Threshold")

    plot_main_accuracy_bars(results_global, results_pc, poster_dir / "fig4_accuracy_bars.png")

    plot_accuracy_cost_vs_sparsity(results_global, results_pc, poster_dir / "fig5_accuracy_cost_vs_sparsity.png")

    logger.info("Generating Figure 6 (Waterfall Plot)...")
    try:
        baseline = results_global[0].baseline_top1
        global_k3 = np.mean([r.top1_accuracy for r in results_global if r.sigma_threshold == 3.0 and not r.is_random])
        pc_k3 = np.mean([r.top1_accuracy for r in results_pc if r.sigma_threshold == 3.0 and not r.is_random])
        mean_only_k3 = np.mean([r.top1_accuracy for r in results_mean_only if r.sigma_threshold == 3.0 and not r.is_random])
        var_only_k3 = np.mean([r.top1_accuracy for r in results_var_only if r.sigma_threshold == 3.0 and not r.is_random])
        plot_ablation_comparison(baseline, global_k3, mean_only_k3, var_only_k3, pc_k3, poster_dir / "fig6_ablation_waterfall.png", sigma_k=3.0)
    except Exception as e:
        logger.error("Could not generate waterfall plot: %s", e)

    # --- Figure 7: Effective gain vs σ_c scatter ---
    logger.info("Generating Figure 7 (Gain-σ Scatter)...")
    try:
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
            import torch
            torch.cuda.empty_cache()

        # Re-read profiling JSON for raw per-channel data + LN γ.
        with open(profiling_path, "r") as f:
            raw_stats = json.load(f)["stats"]

        gains, stds_list = [], []
        for bidx in (8, 9, 10):
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
            stds_list.append(pc_std)

        from src.plotting_poster import plot_effective_gain_scatter
        plot_effective_gain_scatter(
            gains, stds_list, poster_dir / "fig7_gain_sigma_scatter.png",
            block_indices=[8, 9, 10],
        )
    except Exception as e:
        logger.error("Could not generate gain-σ scatter: %s", e)

    logger.info("--- Poster Figure Generation Complete ---")

if __name__ == "__main__":
    main()
