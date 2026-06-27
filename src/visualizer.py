"""
visualizer.py
=============

Turns the per-layer outlier summaries from Experiment 1 into simple bar charts,
so the raw numbers become easy to interpret at a glance.

Every chart draws one bar per linear layer, in network order (early layers on
the left, late layers on the right), and colours each bar by its layer type so
the attention-vs-MLP story is visible immediately.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

# Use the non-interactive "Agg" backend: we only ever save figures to disk, never
# pop up a window. This must be selected BEFORE importing pyplot, and it lets the
# script run on a headless server with no display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import must follow backend selection)
from matplotlib.patches import Patch  # noqa: E402

from src.hooks import LayerOutlierSummary
from src.model_utils import LayerType

# A fixed colour per layer type, so every chart is visually consistent. This is
# a constant lookup table (configuration), not mutable program state.
LAYER_TYPE_COLORS: dict[LayerType, str] = {
    LayerType.ATTENTION: "tab:blue",
    LayerType.FEEDFORWARD: "tab:red",
    LayerType.OTHER: "tab:gray",
}


def shorten_layer_name(layer_name: str) -> str:
    """
    Make long module paths readable on a chart axis, e.g.
    'encoder.layers.encoder_layer_5.mlp.0' -> 'L5.mlp.0'.
    """
    short_name = layer_name.replace("encoder.layers.encoder_layer_", "L")
    short_name = short_name.replace(".self_attention", ".attn")
    return short_name


def _plot_metric_bars(
    summaries: Sequence[LayerOutlierSummary],
    values: Sequence[float],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """
    Draw one bar per layer for a single metric, colour the bars by layer type,
    add a legend, and save the figure to `output_path`.
    """
    bar_labels: list[str] = [shorten_layer_name(s.layer_name) for s in summaries]
    bar_colors: list[str] = [LAYER_TYPE_COLORS[s.layer_type] for s in summaries]
    bar_positions: list[int] = list(range(len(summaries)))

    # Widen the figure when there are many layers so bars do not overlap.
    figure_width: float = max(8.0, len(summaries) * 0.3)
    figure, axes = plt.subplots(figsize=(figure_width, 5.0))

    axes.bar(bar_positions, list(values), color=bar_colors)
    axes.set_title(title)
    axes.set_ylabel(y_label)
    axes.set_xticks(bar_positions)
    axes.set_xticklabels(bar_labels, rotation=90, fontsize=7)

    # Build a small legend that maps each colour back to its layer type.
    legend_handles = [Patch(color=color) for color in LAYER_TYPE_COLORS.values()]
    legend_labels = [layer_type.value for layer_type in LAYER_TYPE_COLORS]
    axes.legend(legend_handles, legend_labels)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)  # release the figure's memory now that it is saved


def generate_all_plots(
    summaries: Sequence[LayerOutlierSummary], output_dir: Path
) -> None:
    """Render every Experiment-1 chart into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_bars(
        summaries,
        values=[s.fixed_outlier_density for s in summaries],
        title="Outlier density per layer (|activation| > 6.0)",
        y_label="Fraction of values routed to FP16",
        output_path=output_dir / "outlier_density_fixed.png",
    )
    _plot_metric_bars(
        summaries,
        values=[s.statistical_outlier_density for s in summaries],
        title="Outlier density per layer (> 3 std-dev from mean)",
        y_label="Fraction of values flagged as outliers",
        output_path=output_dir / "outlier_density_statistical.png",
    )
    _plot_metric_bars(
        summaries,
        values=[s.max_magnitude for s in summaries],
        title="Maximum activation magnitude per layer",
        y_label="Max |activation|",
        output_path=output_dir / "max_magnitude.png",
    )
    _plot_metric_bars(
        summaries,
        values=[s.channel_persistence_variance for s in summaries],
        title="Channel persistence per layer (variance of per-channel outlier counts)",
        y_label="Variance  (high = persistent, low = scattered)",
        output_path=output_dir / "channel_persistence.png",
    )
