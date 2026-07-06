"""
Run Experiment 3: Per-Layer Sensitivity Analysis.

This script identifies which layers of a Vision Transformer are most sensitive to
post-training quantization. It works by iterating through each linear layer of the
model, quantizing it to INT8 (while leaving all other layers in FP32), and then
measuring the resulting Top-1 accuracy on a subset of the ImageNet validation set.

This "one-at-a-time" analysis isolates the effect of quantizing each specific
layer, providing a clear ranking of layer sensitivity. The layers that cause the
largest drop in accuracy are the most sensitive and are the primary candidates for
being kept in higher precision in a selective quantization scheme.

Results, including the accuracy drop for each layer, are saved to a CSV file in
the `outputs/exp3_sensitivity/` directory. A PNG plot visualizing these results
is automatically generated and saved in the same location.

Usage:

    # For a quick debugging run on a small subset of images:
    python run_experiment3_sensitivity.py --num-images 128

    # For a more representative development run:
    python run_experiment3_sensitivity.py --num-images 4096

    # For a full, final validation run on the entire dataset:
    python run_experiment3_sensitivity.py

"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.data_loader import create_imagenet_val_loader
from src.model_utils import evaluate_top1_accuracy, get_linear_layers, load_vit_model
from src.quantization import quantize_per_tensor


def visualize_results(results_path: str):
    """Generate and save the sensitivity plot."""
    df = pd.read_csv(results_path)

    # Infer the baseline accuracy from the first row
    baseline_accuracy = df["quantized_accuracy"][0] + df["accuracy_drop"][0]

    plt.figure(figsize=(15, 8))
    sns.barplot(x="layer_name", y="quantized_accuracy", data=df, color="#4C72B0")

    # Add a horizontal line for the baseline accuracy
    plt.axhline(
        y=baseline_accuracy,
        color="r",
        linestyle="--",
        label=f"Baseline FP32 Acc: {baseline_accuracy:.2f}%",
    )

    plt.xticks(rotation=90)
    plt.title("Per-Layer Quantization Sensitivity")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.xlabel("Layer Name")

    # Set y-axis limits to better zoom in on the accuracy changes
    min_accuracy = df["quantized_accuracy"].min()
    plt.ylim(bottom=min_accuracy - 0.1, top=baseline_accuracy + 0.1)

    plt.legend()
    plt.tight_layout()

    # Save the plot to the same directory as the CSV
    output_path = results_path.replace(".csv", ".png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def run_experiment_3(num_images: int | None):
    """Main function to run the per-layer sensitivity analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, _ = load_vit_model()
    model.to(device)
    linear_layers = get_linear_layers(model)
    data_loader = create_imagenet_val_loader(batch_size=64, max_images=num_images)

    # Get baseline accuracy
    print(f"Evaluating baseline FP32 accuracy on {num_images or 'all'} images...")
    baseline_accuracy = evaluate_top1_accuracy(model, data_loader, device)
    print(f"Baseline FP32 accuracy: {baseline_accuracy:.2f}%")

    # --- Per-layer sensitivity analysis ---
    results = []
    for layer_name, layer in linear_layers.items():
        print(f"Quantizing layer: {layer_name}")

        # Store original weights
        original_weight = layer.weight.data.clone()

        # Quantize layer
        layer.weight.data = quantize_per_tensor(layer.weight.data)

        # Evaluate accuracy
        quantized_accuracy = evaluate_top1_accuracy(model, data_loader, device)
        accuracy_drop = baseline_accuracy - quantized_accuracy
        print(f"  -> Accuracy: {quantized_accuracy:.2f}%, Drop: {accuracy_drop:.2f}%")

        results.append(
            {
                "layer_name": layer_name,
                "quantized_accuracy": quantized_accuracy,
                "accuracy_drop": accuracy_drop,
            }
        )

        # Restore original weights
        layer.weight.data = original_weight

    # --- Save results ---
    output_dir = "outputs/exp3_sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    image_suffix = f"_{num_images}imgs" if num_images is not None else "_all_imgs"
    output_path = os.path.join(output_dir, f"sensitivity_results{image_suffix}.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer_name", "quantized_accuracy", "accuracy_drop"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Automatically visualize the results
    visualize_results(output_path)

    print("Experiment 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to use for the experiment.",
    )
    args = parser.parse_args()
    run_experiment_3(args.num_images)
