"""Tests for the poster plotting utilities in :mod:`src.plotting_poster`.

All tests write to a temporary directory and assert that the expected PNG
file is created.  No model weights are loaded; inputs are synthetic arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ablation import AblationResult
from src.plotting_poster import (
    plot_ablation_waterfall,
    plot_accuracy_vs_sparsity_scatter,
    plot_activation_distribution_overlay,
    plot_attention_entropy_streamgraph,
    plot_outlier_site_grid,
    plot_per_channel_mean_hinton,
    plot_per_channel_sigma_ridgeline,
)


# ---------------------------------------------------------------------------
# 1. Activation distribution overlay
# ---------------------------------------------------------------------------


def test_plot_activation_distribution_overlay_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=1)
    activations = rng.standard_normal(5000).astype(np.float32) * 5.0
    output_path = tmp_path / "overlay.png"
    plot_activation_distribution_overlay(
        activations, "Test Layer", output_path,
        global_mean=0.0, global_std=5.0,
        per_channel_stds=[3.0, 7.0, 5.0],
        per_channel_means=[0.0, 0.0, 0.0],
        sigma_k=3.0,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_activation_distribution_overlay_no_per_channel(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=2)
    activations = rng.standard_normal(2000).astype(np.float32)
    output_path = tmp_path / "overlay_no_pc.png"
    plot_activation_distribution_overlay(
        activations, "Test Layer", output_path,
        global_mean=0.0, global_std=1.0,
        sigma_k=3.0,
    )
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 2. Outlier site grid
# ---------------------------------------------------------------------------


def test_plot_outlier_site_grid_creates_file(tmp_path: Path) -> None:
    fracs: dict[str, dict[str, float]] = {
        f"blocks.{i}/pre_gelu": {
            "3.0_sigma": 0.01 + i * 0.005,
            "4.0_sigma": 0.001 + i * 0.001,
            "6.0_sigma": 0.0001 + i * 0.0001,
        }
        for i in range(12)
    }
    # Add a few other sites.
    for site in ["residual_stream", "post_layernorm_1"]:
        for i in range(12):
            fracs[f"blocks.{i}/{site}"] = {"3.0_sigma": 0.001}
    output_path = tmp_path / "outlier_grid.png"
    plot_outlier_site_grid(fracs, output_path, sigma_key="3.0_sigma")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_outlier_site_grid_empty(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_grid.png"
    plot_outlier_site_grid({}, output_path)
    # Grid with no data should still render (all gray tiles).
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 3. Accuracy vs sparsity scatter
# ---------------------------------------------------------------------------


def _make_fake_results(
    site: str = "pre_gelu",
    accuracies: tuple[float, ...] = (80.0, 85.0, 90.0),
    ks: tuple[float, ...] = (3.0, 4.0, 6.0),
    pct_zeroed: tuple[float, ...] = (10.0, 5.0, 1.0),
) -> list[AblationResult]:
    return [
        AblationResult(
            site=site, sigma_threshold=k, site_identifier=f"blocks.{i}/{site}",
            pct_zeroed=pct, top1_accuracy=acc, top5_accuracy=95.0,
            baseline_top1=91.0, baseline_top5=97.0,
        )
        for i, (k, acc, pct) in enumerate(zip(ks, accuracies, pct_zeroed))
    ]


def test_plot_accuracy_vs_sparsity_scatter_creates_file(tmp_path: Path) -> None:
    a = _make_fake_results(accuracies=(43.0, 75.0, 84.0), pct_zeroed=(30.0, 10.0, 2.0))
    b = _make_fake_results(accuracies=(47.0, 75.5, 84.1), pct_zeroed=(25.0, 8.0, 1.5))
    output_path = tmp_path / "sparsity_scatter.png"
    plot_accuracy_vs_sparsity_scatter(a, b, output_path, label_a="Global", label_b="Per-channel")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 4. Ridgeline
# ---------------------------------------------------------------------------


def test_plot_per_channel_sigma_ridgeline_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=3)
    pc_stds: dict[str, list[float]] = {}
    for i in range(12):
        # Early blocks: tight.  Late blocks: wide, bimodal-ish.
        if i < 4:
            arr = rng.gamma(2.0, 1.0, 3072).tolist()
        elif i < 8:
            arr = rng.gamma(3.0, 2.0, 3072).tolist()
        else:
            a1 = rng.gamma(2.0, 1.0, 1536).tolist()
            a2 = rng.gamma(6.0, 3.0, 1536).tolist()
            arr = a1 + a2
        pc_stds[f"blocks.{i}/pre_gelu"] = arr
    output_path = tmp_path / "ridgeline.png"
    plot_per_channel_sigma_ridgeline(pc_stds, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_per_channel_sigma_ridgeline_no_pre_gelu(tmp_path: Path) -> None:
    """Should skip gracefully if no pre_gelu data."""
    output_path = tmp_path / "empty_ridge.png"
    plot_per_channel_sigma_ridgeline({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 5. Streamgraph
# ---------------------------------------------------------------------------


def test_plot_attention_entropy_streamgraph_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=4)
    cls_ent: dict[str, list[float]] = {}
    for i in range(12):
        # Entropy starts high, collapses in later blocks.
        base = 3.0 - i * 0.22
        cls_ent[f"blocks.{i}/post_softmax"] = [
            max(0.05, base + rng.normal(0, 0.3)) for _ in range(12)
        ]
    output_path = tmp_path / "streamgraph.png"
    plot_attention_entropy_streamgraph(cls_ent, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_attention_entropy_streamgraph_empty(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_stream.png"
    plot_attention_entropy_streamgraph({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 6. Waterfall
# ---------------------------------------------------------------------------


def test_plot_ablation_waterfall_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "waterfall.png"
    plot_ablation_waterfall(
        baseline=85.03,
        global_acc=43.24,
        mean_only_acc=45.00,
        var_only_acc=46.20,
        outlier_acc=47.00,
        output_path=output_path,
        sigma_k=3.0,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 7. Hinton diagram
# ---------------------------------------------------------------------------


def test_plot_per_channel_mean_hinton_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=5)
    # Block 10-like: asymmetric, large negatives.
    means = rng.normal(-5.0, 15.0, 3072).tolist()
    pc_means: dict[str, list[float]] = {
        "blocks.10/pre_gelu": means,
    }
    output_path = tmp_path / "hinton.png"
    plot_per_channel_mean_hinton(pc_means, 10, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_per_channel_mean_hinton_missing_block(tmp_path: Path) -> None:
    output_path = tmp_path / "missing_hinton.png"
    plot_per_channel_mean_hinton({}, 10, output_path)
    assert not output_path.exists()
