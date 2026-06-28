"""
run_experiment1_mapping.py
==========================

Driver for Experiment 1: map activation-outlier behaviour across every linear
layer of ViT-B/16.

The pipeline, end to end:
  1. Load the model and its matching preprocessing transform.
  2. PASS 1: stream the images and compute each layer's EXACT global mean and
     standard deviation (these define the statistical 3-sigma threshold).
  3. PASS 2: stream the SAME images again and, using the frozen pass-1
     statistics, measure outliers on the activation entering each matmul - the
     LLM.int8() routing decision point - for both the fixed 6.0 and the 3-sigma
     thresholds.
  4. Every hook reduces its layer's input activations to scalar statistics on
     the fly (no raw activations are stored).
  5. Save the per-layer summaries to JSON and render charts.

We read the data twice (two passes) on purpose: it lets the 3-sigma threshold
use exact global statistics instead of a per-batch approximation. The data
loader is deterministic, so both passes see identical inputs.

This file is intentionally lightweight: all reusable logic lives in `src/`.

How many images to run is set on the command line, so you can shift gears
without editing code. The three workflows from the thesis plan:

    # "Debugging" run - just make sure nothing crashes (seconds).
    python run_experiment1_mapping.py --num-images 128

    # "Characterization" run - the default; generate meaningful heatmaps.
    python run_experiment1_mapping.py --num-images 4096

    # "Thesis print" run - final, high-confidence numbers.
    python run_experiment1_mapping.py --num-images 50000

Run `python run_experiment1_mapping.py --help` to see every option.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from src import visualizer
from src.data_loader import build_image_dataloader
from src.hooks import (
    ActivationRecorder,
    LayerOutlierSummary,
    LayerThresholds,
    MomentCollector,
    OutlierStatsCollector,
    make_measurement_hook,
)
from src.model_utils import (
    ImageTransform,
    iter_measured_modules,
    load_vit_b_16,
)

# --- workflow presets (just the image counts named in the thesis plan) -------
DEBUG_RUN_IMAGES: int = 128  # quick "does it crash?" pass while coding
CHARACTERIZATION_RUN_IMAGES: int = 4096  # the everyday heatmap-generating run
THESIS_PRINT_RUN_IMAGES: int = 50000  # the final, full-validation run


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Every knob for one run of Experiment 1, gathered into a single typed object
    instead of loose module-level globals. `frozen=True` makes it read-only once
    parsed, so no part of the pipeline can accidentally change it mid-run.
    """

    num_images: int
    batch_size: int
    data_dir: Path
    output_dir: Path
    device: torch.device

    @property
    def json_output_path(self) -> Path:
        """Where the per-layer statistics JSON is written."""
        return self.output_dir / "outlier_stats.json"


def parse_config() -> ExperimentConfig:
    """
    Read command-line arguments and build the immutable `ExperimentConfig`.

    Anything the user does not specify falls back to a sensible default (the
    4,096-image characterization run, batch size 64, reading from `data/`).
    """
    parser = argparse.ArgumentParser(
        description="Experiment 1: map activation outliers across ViT-B/16 linear layers.",
        # Shows each option's default value in --help automatically.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=CHARACTERIZATION_RUN_IMAGES,
        help=(
            "How many images to stream through the model. Suggested values: "
            f"{DEBUG_RUN_IMAGES} (debugging), "
            f"{CHARACTERIZATION_RUN_IMAGES} (characterization / heatmaps), "
            f"{THESIS_PRINT_RUN_IMAGES} (final thesis-print run)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images processed per forward pass (raise/lower to fit VRAM).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing the input images (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp1_outlier_maps"),
        help="Folder where the JSON stats and charts are written.",
    )
    arguments = parser.parse_args()

    # Prefer the GPU when available; fall back to CPU otherwise.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ExperimentConfig(
        num_images=arguments.num_images,
        batch_size=arguments.batch_size,
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        device=device,
    )


def attach_measurement_hooks(
    model: VisionTransformer, recorder: ActivationRecorder
) -> list[RemovableHandle]:
    """
    Register one measurement PRE-hook per linear layer and tell the recorder
    about each layer up front. A pre-hook fires just before the matmul runs, so
    it sees the layer's input activation - the exact tensor LLM.int8() inspects
    to decide INT8-vs-FP16 routing. The same wiring drives both passes; only the
    recorder (a `MomentCollector` or an `OutlierStatsCollector`) differs.

    Returns the list of handles so we can detach the hooks afterwards. Leaving
    hooks attached would keep measuring during any later use of the model.
    """
    handles: list[RemovableHandle] = []
    for layer_name, module, layer_type in iter_measured_modules(model):
        recorder.register_layer(layer_name, layer_type)

        hook = make_measurement_hook(recorder, layer_name)
        handle: RemovableHandle = module.register_forward_pre_hook(hook)
        handles.append(handle)
    return handles


def run_forward_passes(
    model: VisionTransformer,
    dataloader: DataLoader[Tensor],
    config: ExperimentConfig,
) -> None:
    """
    Push every batch of images through the model. We discard the predictions -
    the hooks are doing the real work of recording statistics in the background.
    """
    # `torch.no_grad()` switches off gradient bookkeeping. We never call
    # `.backward()`, so this roughly halves VRAM use and speeds things up.
    with torch.no_grad():
        for batch_index, images in enumerate(dataloader):
            images = images.to(config.device)
            _ = model(images)  # the forward pass triggers every hook

            images_done = (batch_index + 1) * config.batch_size
            print(f"  processed ~{images_done} / {config.num_images} images", end="\r")
    print()  # finish the in-place progress line with a newline


def compute_thresholds(
    model: VisionTransformer,
    dataloader: DataLoader[Tensor],
    config: ExperimentConfig,
) -> LayerThresholds:
    """PASS 1: stream the data to compute each layer's exact global mean/std."""
    moment_collector = MomentCollector()
    handles = attach_measurement_hooks(model, moment_collector)
    run_forward_passes(model, dataloader, config)
    for handle in handles:
        handle.remove()
    return moment_collector.build_thresholds()


def characterize_outliers(
    model: VisionTransformer,
    dataloader: DataLoader[Tensor],
    config: ExperimentConfig,
    thresholds: LayerThresholds,
) -> list[LayerOutlierSummary]:
    """PASS 2: stream the SAME data, counting outliers against frozen thresholds."""
    collector = OutlierStatsCollector(thresholds)
    handles = attach_measurement_hooks(model, collector)
    run_forward_passes(model, dataloader, config)
    for handle in handles:
        handle.remove()
    return collector.build_summaries()


def export_summaries_to_json(summaries: list[LayerOutlierSummary], path: Path) -> None:
    """Write the per-layer summaries to a pretty-printed JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_summaries = [summary.to_dict() for summary in summaries]
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(serializable_summaries, json_file, indent=2)


def main() -> None:
    config = parse_config()
    print(f"Device: {config.device}")
    print(f"Run size: {config.num_images} images, batch size {config.batch_size}")

    # 1. Model + the exact preprocessing it was trained with.
    print("Loading ViT-B/16 ...")
    model: VisionTransformer
    transform: ImageTransform
    model, transform = load_vit_b_16(config.device)

    # 2. Build the (deterministic) image stream, reused identically by both passes.
    print(f"Loading up to {config.num_images} images from '{config.data_dir}' ...")
    dataloader: DataLoader[Tensor] = build_image_dataloader(
        image_dir=config.data_dir,
        transform=transform,
        batch_size=config.batch_size,
        max_images=config.num_images,
    )

    # 3. PASS 1 - exact global mean/std per layer (defines the 3-sigma threshold).
    print("Pass 1/2: computing exact global mean & std per layer ...")
    thresholds = compute_thresholds(model, dataloader, config)

    # 4. PASS 2 - outlier characterization against the frozen thresholds.
    print("Pass 2/2: characterizing outliers (fixed 6.0 and 3-sigma) ...")
    summaries: list[LayerOutlierSummary] = characterize_outliers(
        model, dataloader, config, thresholds
    )

    # 5. Save the results and charts.
    export_summaries_to_json(summaries, config.json_output_path)
    print(f"Wrote stats for {len(summaries)} layers to '{config.json_output_path}'.")

    visualizer.generate_all_plots(summaries, config.output_dir)
    print(f"Saved charts to '{config.output_dir}'.")


if __name__ == "__main__":
    main()
