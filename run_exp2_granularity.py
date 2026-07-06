"""
Run Experiment 2: Accuracy by Quantization Granularity.

This script quantizes the ENTIRE ViT-B/16 model to INT8 using four different
weight × activation scaling strategies and measures the resulting Top-1 accuracy
on the ImageNet validation set. It also records per-layer quantization error
(MSE) so we can correlate damage with the outlier maps from Experiment 1.

The four configurations form a 2×2 grid:

    Config A: per-tensor weights  × per-tensor activations  (the floor)
    Config B: per-channel weights × per-tensor activations  (finer weights)
    Config C: per-tensor weights  × per-token activations   (finer activations)
    Config D: per-channel weights × per-token activations   (the ceiling)

Each config is evaluated 3 times and results are reported as mean ± std.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from src.data_loader import create_imagenet_val_loader
from src.model_utils import evaluate_top1_accuracy, get_linear_layers, load_vit_model
from src.quantization import (
    MSEAccumulator,
    make_activation_quant_hook,
    quantize_all_weights,
    restore_weights,
)

# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

WEIGHT_STRATEGIES = ("per_tensor", "per_channel")
ACTIVATION_STRATEGIES = ("per_tensor", "per_token")

# The four configs: (weight_strategy, activation_strategy, display_name)
CONFIGS: List[tuple[str, str, str]] = [
    ("per_tensor", "per_tensor", "per_tensor"),
    ("per_channel", "per_tensor", "per_channel_weights"),
    ("per_tensor", "per_token", "per_token_activations"),
    ("per_channel", "per_token", "per_channel_per_token"),
]

DEFAULT_NUM_RUNS = 3


# ---------------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Results from a single evaluation run for one config."""

    run_index: int
    top1_accuracy: float
    accuracy_drop: float
    per_layer_mse: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfigResult:
    """Aggregated results across all runs for one config."""

    config_name: str
    weight_strategy: str
    activation_strategy: str
    runs: List[RunResult] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.top1_accuracy for r in self.runs) / len(self.runs)

    @property
    def std_accuracy(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_accuracy
        variance = sum((r.top1_accuracy - mean) ** 2 for r in self.runs) / (
            len(self.runs) - 1
        )
        return variance**0.5

    @property
    def mean_drop(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.accuracy_drop for r in self.runs) / len(self.runs)

    @property
    def std_drop(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_drop
        variance = sum((r.accuracy_drop - mean) ** 2 for r in self.runs) / (
            len(self.runs) - 1
        )
        return variance**0.5


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_with_quantization(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    linear_layers: Dict[str, nn.Linear],
    weight_strategy: str,
    activation_strategy: str,
) -> tuple[float, Dict[str, float]]:
    """
    Quantize the model, evaluate accuracy, and return (top1, per_layer_mse).

    The model's weights are restored to their original state before returning.
    """
    # Quantize weights in-place, saving originals.
    originals = quantize_all_weights(model, weight_strategy, linear_layers)

    # Register activation quant hooks with MSE tracking.
    mse_tracker = MSEAccumulator()
    handles: List[RemovableHandle] = []
    for name, module in linear_layers.items():
        hook = make_activation_quant_hook(activation_strategy, name, mse_tracker)
        handles.append(module.register_forward_pre_hook(hook))

    try:
        accuracy = evaluate_top1_accuracy(model, data_loader, device)
    finally:
        # Always clean up hooks and restore weights, even on error.
        for handle in handles:
            handle.remove()
        restore_weights(model, originals, linear_layers)

    return accuracy, mse_tracker.get_all()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment_2(
    num_images: int | None,
    batch_size: int,
    data_dir: str,
    output_dir: str,
    num_runs: int,
) -> None:
    """Main function to run the full quantization granularity experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run size: {num_images or 'all'} images, batch size {batch_size}")
    print(f"Number of runs per config: {num_runs}")

    # 1. Load model and data.
    print("Loading ViT-B/16 ...")
    model, _ = load_vit_model()
    model.to(device)
    linear_layers = get_linear_layers(model)
    print(f"Found {len(linear_layers)} linear layers.")

    data_loader = create_imagenet_val_loader(
        batch_size=batch_size, data_dir=data_dir, max_images=num_images
    )

    # 2. Baseline FP32 accuracy.
    print("Evaluating baseline FP32 accuracy ...")
    baseline_accuracy = evaluate_top1_accuracy(model, data_loader, device)
    print(f"Baseline FP32 accuracy: {baseline_accuracy:.2f}%")

    # 3. Evaluate each config.
    all_config_results: List[ConfigResult] = []

    for weight_strategy, activation_strategy, config_name in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Config: {config_name}")
        print(f"  Weights: {weight_strategy}, Activations: {activation_strategy}")
        print(f"{'=' * 60}")

        config_result = ConfigResult(
            config_name=config_name,
            weight_strategy=weight_strategy,
            activation_strategy=activation_strategy,
        )

        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs} ...")

            accuracy, per_layer_mse = evaluate_with_quantization(
                model,
                data_loader,
                device,
                linear_layers,
                weight_strategy,
                activation_strategy,
            )

            drop = baseline_accuracy - accuracy
            print(f"    Accuracy: {accuracy:.2f}%, Drop: {drop:.2f}%")

            config_result.runs.append(
                RunResult(
                    run_index=run_idx + 1,
                    top1_accuracy=accuracy,
                    accuracy_drop=drop,
                    per_layer_mse=per_layer_mse,
                )
            )

        print(
            f"  Mean accuracy: {config_result.mean_accuracy:.2f}% "
            f"± {config_result.std_accuracy:.2f}%"
        )
        all_config_results.append(config_result)

    # 4. Save results.
    os.makedirs(output_dir, exist_ok=True)
    image_suffix = f"_{num_images}imgs" if num_images is not None else "_all_imgs"

    # 4a. Per-run accuracy CSV.
    accuracy_csv_path = os.path.join(output_dir, f"accuracy_results{image_suffix}.csv")
    with open(accuracy_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_name",
                "weight_strategy",
                "activation_strategy",
                "run",
                "top1_accuracy",
                "accuracy_drop",
            ],
        )
        writer.writeheader()
        for cr in all_config_results:
            for run in cr.runs:
                writer.writerow(
                    {
                        "config_name": cr.config_name,
                        "weight_strategy": cr.weight_strategy,
                        "activation_strategy": cr.activation_strategy,
                        "run": run.run_index,
                        "top1_accuracy": f"{run.top1_accuracy:.4f}",
                        "accuracy_drop": f"{run.accuracy_drop:.4f}",
                    }
                )
    print(f"\nPer-run accuracy saved to {accuracy_csv_path}")

    # 4b. Summary JSON.
    summary_path = os.path.join(output_dir, f"accuracy_summary{image_suffix}.json")
    summary = {
        "baseline_fp32_accuracy": baseline_accuracy,
        "num_images": num_images,
        "num_runs": num_runs,
        "configs": [
            {
                "name": cr.config_name,
                "weight_strategy": cr.weight_strategy,
                "activation_strategy": cr.activation_strategy,
                "mean_accuracy": cr.mean_accuracy,
                "std_accuracy": cr.std_accuracy,
                "mean_drop": cr.mean_drop,
                "std_drop": cr.std_drop,
            }
            for cr in all_config_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # 4c. Per-layer MSE CSV (one file with all configs).
    mse_csv_path = os.path.join(output_dir, f"per_layer_mse{image_suffix}.csv")
    # Collect all layer names across all configs (should be identical).
    all_layer_names: List[str] = []
    for cr in all_config_results:
        for run in cr.runs:
            for name in run.per_layer_mse:
                if name not in all_layer_names:
                    all_layer_names.append(name)

    with open(mse_csv_path, "w", newline="") as f:
        fieldnames = ["layer_name"] + [cr.config_name for cr in all_config_results]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer_name in all_layer_names:
            row: Dict[str, str] = {"layer_name": layer_name}
            for cr in all_config_results:
                # Average MSE across runs for this layer.
                mse_values = [run.per_layer_mse.get(layer_name, 0.0) for run in cr.runs]
                avg_mse = sum(mse_values) / len(mse_values) if mse_values else 0.0
                row[cr.config_name] = f"{avg_mse:.6f}"
            writer.writerow(row)
    print(f"Per-layer MSE saved to {mse_csv_path}")

    print("\nExperiment 2 complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2: Accuracy by Quantization Granularity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to use for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/imagenet-val",
        help="Path to the ImageNet validation directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp2_granularity",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of evaluation runs per config.",
    )
    args = parser.parse_args()
    run_experiment_2(
        num_images=args.num_images,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
    )
