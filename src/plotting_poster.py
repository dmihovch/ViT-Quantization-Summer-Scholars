
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_PALETTE = {
    "coral": "#FF6F61",
    "teal": "#00A99D",
    "yellow": "#FFC107",
    "blue": "#007BFF",
    "gray": "#868e96",
    "dark": "#343a40",
    "red": "#DC3545",
}

def _poster_style(ax, fontsize=12):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)

def plot_activation_distribution_overlay(tensor, title, output_path, global_mean, global_std, per_channel_stds, per_channel_means, sigma_k):
    logger.info(f"Plotting activation distribution overlay to {output_path}")

def plot_outlier_site_grid(outlier_fracs, output_path, *args, **kwargs):
    logger.info(f"Plotting outlier site grid to {output_path}")

def plot_per_channel_sigma_line(pc_stds, output_path, *args, **kwargs):
    logger.info(f"Plotting per-channel sigma line to {output_path}")

def plot_attention_entropy_heatmap(cls_ent, output_path, *args, **kwargs):
    logger.info(f"Plotting attention entropy heatmap to {output_path}")

def plot_per_channel_mean_histogram(pc_means, block_idx, output_path, *args, **kwargs):
    logger.info(f"Plotting per-channel mean histogram to {output_path}")

def plot_ablation_comparison(
    baseline: float,
    global_acc: float,
    mean_only_acc: float,
    var_only_acc: float,
    outlier_acc: float,
    output_path: Path,
    *,
    title: str | None = None,
    sigma_k: float = 3.0,
) -> None:
    labels = ["Baseline", "Global\nμ + global σ",
              "Per-ch. μ +\nglobal σ",
              "Global μ +\n per-ch. σ",
              "Per-ch. μ +\n per-ch. σ"]
    values = [baseline, global_acc, mean_only_acc, var_only_acc, outlier_acc]
    diffs = [0.0] + [v - baseline for v in values[1:]]

    colors = [
        _PALETTE["gray"],
        _PALETTE["coral"],
        _PALETTE["yellow"],
        _PALETTE["blue"],
        _PALETTE["teal"],
    ]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _poster_style(ax, fontsize=15)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.55, zorder=3)

    for i, (val, diff) in enumerate(zip(values, diffs)):
        ax.text(i, val + 2.5, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=_PALETTE["dark"])
        if i > 0:
            sign = "+" if diff > 0 else ""
            ax.annotate(f"{sign}{diff:.2f} pp vs baseline",
                        xy=(i, val), xytext=(i, val + 9),
                        fontsize=15,
                        color=_PALETTE["red"] if diff < 0 else _PALETTE["teal"],
                        ha="center")

    ax.axhline(baseline, color=_PALETTE["gray"], linestyle="--", linewidth=1.5,
               zorder=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_ylim(bottom=0, top=100)
    ax.set_title(title or f"Ablation Condition Comparison at k={sigma_k}", fontsize=18, fontweight="bold")
    ax.grid(axis='y', linestyle='--', color='#cccccc', zorder=0)

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation comparison to %s", output_path)

def plot_effective_gain_scatter(gains, stds_list, output_path, block_indices):
    logger.info(f"Plotting effective gain scatter to {output_path}")


def plot_accuracy_vs_sparsity_scatter(
    results_a: list,
    results_b: list,
    output_path: Path,
    label_a: str = "Global",
    label_b: str = "Per-channel",
) -> None:
    """Plot accuracy drop vs. induced sparsity as a connected scatter."""
    import collections

    def _aggregate_by_k(results):
        grouped = collections.defaultdict(lambda: {"accs": [], "sparsities": []})
        for r in results:
            if r.site == "pre_gelu" and not r.is_random:
                grouped[r.sigma_threshold]["accs"].append(r.top1_accuracy)
                grouped[r.sigma_threshold]["sparsities"].append(r.pct_zeroed)
        agg = {}
        for k, data in sorted(grouped.items()):
            agg[k] = {
                "mean_acc": np.mean(data["accs"]),
                "mean_sparsity": np.mean(data["sparsities"]),
            }
        return agg

    agg_a = _aggregate_by_k(results_a)
    agg_b = _aggregate_by_k(results_b)

    baseline = results_a[0].baseline_top1 if results_a else 85.03

    sparsity_a = [v["mean_sparsity"] for v in agg_a.values()]
    cost_a = [baseline - v["mean_acc"] for v in agg_a.values()]
    sparsity_b = [v["mean_sparsity"] for v in agg_b.values()]
    cost_b = [baseline - v["mean_acc"] for v in agg_b.values()]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    _poster_style(ax, fontsize=16)

    ax.plot(sparsity_a, cost_a, "o-", color=_PALETTE["coral"],
            label=label_a, linewidth=2.5, markersize=8)
    ax.plot(sparsity_b, cost_b, "o-", color=_PALETTE["teal"],
            label=label_b, linewidth=2.5, markersize=8)

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
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="#CCCCCC")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy vs sparsity scatter to %s", output_path)
