"""
Regenerate all plots from the Experiment 1 outlier statistics.

Reads the JSON written by ``run_experiment1_mapping.py``, reconstructs
``LayerOutlierSummary`` objects (converting the serialised string ``layer_type``
back to the ``LayerType`` enum), and re-renders every chart.
"""

import json
from pathlib import Path

from src.hooks import LayerOutlierSummary
from src.model_utils import LayerType, classify_linear_layer
from src.visualizer import generate_all_plots

# Map the string values written by ``LayerOutlierSummary.to_dict()`` back to
# their enum members.  This is the inverse of ``layer_type.value``.
_LAYER_TYPE_BY_VALUE: dict[str, LayerType] = {
    member.value: member for member in LayerType
}


def _reconstruct_summary(entry: dict[str, object]) -> LayerOutlierSummary:
    """Convert one JSON dict back into a ``LayerOutlierSummary``.

    The JSON stores ``layer_type`` as a plain string (the enum's ``.value``);
    we convert it back to the enum so the dataclass constructor accepts it.
    If the stored string does not match any known ``LayerType`` value (e.g.
    legacy JSON with ``"FeedForward_MLP"`` or ``"Attention_QKV"`` used as a
    catch-all), we fall back to classifying by the layer name.
    """
    raw_type: str = str(entry.get("layer_type", ""))
    layer_type: LayerType = _LAYER_TYPE_BY_VALUE.get(
        raw_type, classify_linear_layer(str(entry["layer_name"]))
    )
    # Build a clean dict with the enum value so ``**unpack`` works.
    clean: dict[str, object] = {**entry, "layer_type": layer_type}
    return LayerOutlierSummary(**clean)  # type: ignore[arg-type]


def main() -> None:
    """Load the stats and regenerate the plots."""
    output_dir = Path("outputs/exp1_outlier_maps")
    stats_path = output_dir / "outlier_stats.json"

    with open(stats_path, "r") as f:
        data = json.load(f)

    summaries = [_reconstruct_summary(entry) for entry in data]

    generate_all_plots(summaries, output_dir)
    print(f"All plots regenerated in {output_dir}")


if __name__ == "__main__":
    main()
