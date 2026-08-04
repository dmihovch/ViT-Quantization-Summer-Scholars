"""Tests for the plotting utilities in :mod:`src.plotting`.

All tests write to a temporary directory and assert that the expected PNG
file is created.  No model weights are loaded; inputs are synthetic arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ablation import AblationResult
from src.plotting import (
    _site_sort_key,
    plot_ablation_mode_comparison,
    plot_accuracy_comparison,
    plot_accuracy_vs_threshold,
    plot_activation_histogram,
    plot_attention_entropy_heatmap,
    plot_bootstrap_ci_delta,
    plot_degradation_efficiency,
    plot_effective_channels,
    plot_entropy_delta_heatmap,
    plot_kurtosis_heatmap,
    plot_ln2_amplification_ratio,
    plot_outlier_fraction_heatmap,
    plot_pct_zeroed_per_layer,
    plot_per_channel_mean_heatmap,
    plot_per_channel_std_heatmap,
)


# ---------------------------------------------------------------------------
# Phase 1 — histogram
# ---------------------------------------------------------------------------


def test_plot_activation_histogram_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=1)
    activations = rng.standard_normal(1000).astype(np.float32)
    output_path = tmp_path / "histogram.png"
    plot_activation_histogram(activations, "blocks.0.mlp.act", output_path, log_scale=True)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Phase 1 — per-channel heatmaps
# ---------------------------------------------------------------------------


def test_plot_per_channel_std_heatmap_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=2)
    stds: dict[str, list[float]] = {
        f"blocks.{i}/pre_gelu": rng.random(16).tolist() for i in range(3)
    }
    output_path = tmp_path / "heatmap.png"
    plot_per_channel_std_heatmap(stds, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_per_channel_mean_heatmap_creates_file(tmp_path: Path) -> None:
    means: dict[str, list[float]] = {
        f"blocks.{i}/pre_gelu": [float(j) for j in range(16)] for i in range(3)
    }
    output_path = tmp_path / "mean_heatmap.png"
    plot_per_channel_mean_heatmap(means, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_per_channel_mean_heatmap_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_mean.png"
    plot_per_channel_mean_heatmap({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 1 — attention entropy
# ---------------------------------------------------------------------------


def test_plot_attention_entropy_heatmap_creates_file(tmp_path: Path) -> None:
    entropies: dict[str, list[float]] = {
        "blocks.0/post_softmax": [1.0, 1.5, 2.0],
        "blocks.1/post_softmax": [0.5, 0.8, 1.2],
    }
    output_path = tmp_path / "entropy.png"
    plot_attention_entropy_heatmap(entropies, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_attention_entropy_heatmap_title_in_file(tmp_path: Path) -> None:
    entropies: dict[str, list[float]] = {
        "blocks.0/post_softmax": [1.0, 1.5, 2.0],
    }
    output_path = tmp_path / "entropy_titled.png"
    plot_attention_entropy_heatmap(entropies, output_path, title="CLS entropy")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_attention_entropy_heatmap_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_entropy.png"
    plot_attention_entropy_heatmap({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 1 — kurtosis heatmap
# ---------------------------------------------------------------------------


def test_plot_kurtosis_heatmap_creates_file(tmp_path: Path) -> None:
    kurtosis: dict[str, float] = {
        f"blocks.{i}/pre_gelu": float(i) * 0.5 for i in range(12)
    }
    for i in range(12):
        kurtosis[f"blocks.{i}/residual_stream"] = 0.1
    output_path = tmp_path / "kurtosis.png"
    plot_kurtosis_heatmap(kurtosis, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_kurtosis_heatmap_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_kurtosis.png"
    plot_kurtosis_heatmap({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 1 — outlier fraction heatmap
# ---------------------------------------------------------------------------


def test_plot_outlier_fraction_heatmap_creates_file(tmp_path: Path) -> None:
    fracs: dict[str, dict[str, float]] = {
        f"blocks.{i}/pre_gelu": {"3.0_sigma": 0.01, "4.0_sigma": 0.001, "6.0_sigma": 0.0001}
        for i in range(12)
    }
    output_path = tmp_path / "outlier_frac.png"
    plot_outlier_fraction_heatmap(fracs, "3.0_sigma", output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_outlier_fraction_heatmap_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_frac.png"
    plot_outlier_fraction_heatmap({}, "3.0_sigma", output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 1 — LN2 amplification ratio
# ---------------------------------------------------------------------------


def test_plot_ln2_amplification_ratio_creates_file(tmp_path: Path) -> None:
    ratios: dict[str, float] = {
        f"blocks.{i}/residual_stream": 1.0 + i * 0.1 for i in range(12)
    }
    output_path = tmp_path / "ln2_ratio.png"
    plot_ln2_amplification_ratio(ratios, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ln2_amplification_ratio_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_ln2.png"
    plot_ln2_amplification_ratio({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — accuracy vs threshold
# ---------------------------------------------------------------------------


def _make_fake_results(
    site: str = "pre_gelu",
    accuracies: tuple[float, ...] = (80.0, 85.0, 90.0),
    ks: tuple[float, ...] = (3.0, 4.0, 6.0),
) -> list[AblationResult]:
    return [
        AblationResult(
            site=site, sigma_threshold=k, site_identifier=f"blocks.{i}/{site}",
            pct_zeroed=10.0, top1_accuracy=acc, top5_accuracy=95.0,
            baseline_top1=91.0, baseline_top5=97.0,
        )
        for i, (k, acc) in enumerate(zip(ks, accuracies))
    ]


def test_plot_accuracy_vs_threshold_creates_file(tmp_path: Path) -> None:
    results = _make_fake_results()
    output_path = tmp_path / "acc_vs_k.png"
    plot_accuracy_vs_threshold(results, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_accuracy_vs_threshold_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_acc.png"
    plot_accuracy_vs_threshold([], output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — pct zeroed per layer
# ---------------------------------------------------------------------------


def test_plot_pct_zeroed_per_layer_creates_file(tmp_path: Path) -> None:
    results = [
        AblationResult(
            site="pre_gelu", sigma_threshold=3.0, site_identifier=f"blocks.{i}/pre_gelu",
            pct_zeroed=float(i) * 5.0, top1_accuracy=80.0, top5_accuracy=95.0,
            baseline_top1=91.0, baseline_top5=97.0,
        )
        for i in range(12)
    ]
    output_path = tmp_path / "pct_zeroed.png"
    plot_pct_zeroed_per_layer(results, 3.0, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Phase 2 — accuracy comparison (overlay)
# ---------------------------------------------------------------------------


def test_plot_accuracy_comparison_creates_file(tmp_path: Path) -> None:
    a = _make_fake_results(accuracies=(43.0, 75.0, 84.0))
    b = _make_fake_results(accuracies=(47.0, 75.5, 84.1))
    output_path = tmp_path / "comparison.png"
    plot_accuracy_comparison(a, b, output_path, label_a="Global", label_b="Per-channel")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_accuracy_comparison_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_comp.png"
    plot_accuracy_comparison([], [], output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — ablation mode comparison
# ---------------------------------------------------------------------------


def test_plot_ablation_mode_comparison_creates_file(tmp_path: Path) -> None:
    mode_results: dict[str, list[AblationResult]] = {
        "outlier": _make_fake_results(accuracies=(43.0,)),
        "mean_only": _make_fake_results(accuracies=(45.0,)),
        "var_only": _make_fake_results(accuracies=(44.0,)),
    }
    output_path = tmp_path / "mode_comp.png"
    plot_ablation_mode_comparison(mode_results, output_path, sigma_k=3.0)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ablation_mode_comparison_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_mode.png"
    plot_ablation_mode_comparison({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — entropy delta
# ---------------------------------------------------------------------------


def test_plot_entropy_delta_heatmap_creates_file(tmp_path: Path) -> None:
    deltas: dict[str, dict[str, float]] = {
        f"blocks.{i}/pre_softmax": {"mean_cls_delta": float(i) * 0.1 - 0.5}
        for i in range(12)
    }
    output_path = tmp_path / "entropy_delta.png"
    plot_entropy_delta_heatmap(deltas, output_path, delta_key="mean_cls_delta")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_entropy_delta_heatmap_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_delta.png"
    plot_entropy_delta_heatmap({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — bootstrap CI
# ---------------------------------------------------------------------------


def test_plot_bootstrap_ci_delta_creates_file(tmp_path: Path) -> None:
    ci: dict[float, dict[str, float]] = {
        3.0: {"delta_point_estimate": 3.76, "delta_ci_low_pct": 3.12, "delta_ci_high_pct": 4.36},
        4.0: {"delta_point_estimate": 0.42, "delta_ci_low_pct": -0.11, "delta_ci_high_pct": 0.96},
        6.0: {"delta_point_estimate": -0.47, "delta_ci_low_pct": -0.93, "delta_ci_high_pct": -0.03},
    }
    output_path = tmp_path / "bootstrap_ci.png"
    plot_bootstrap_ci_delta(ci, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_bootstrap_ci_delta_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_ci.png"
    plot_bootstrap_ci_delta({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — effective channels
# ---------------------------------------------------------------------------


def test_plot_effective_channels_creates_file(tmp_path: Path) -> None:
    channels: dict[str, dict[str, float]] = {
        f"blocks.{i}/pre_gelu": {"global_channels": 3000.0 - i * 100, "pc_channels": 3050.0 - i * 100}
        for i in range(12)
    }
    output_path = tmp_path / "eff_channels.png"
    plot_effective_channels(channels, 3.0, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_effective_channels_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_eff.png"
    plot_effective_channels({}, 3.0, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Phase 2 — degradation efficiency
# ---------------------------------------------------------------------------


def test_plot_degradation_efficiency_creates_file(tmp_path: Path) -> None:
    deg: dict[float, dict[str, float]] = {
        3.0: {"global_degradation_per_pct": 100.97, "pc_degradation_per_pct": 53.43},
        4.0: {"global_degradation_per_pct": 23.94, "pc_degradation_per_pct": 13.33},
        6.0: {"global_degradation_per_pct": 1.07, "pc_degradation_per_pct": 1.28},
    }
    output_path = tmp_path / "degradation.png"
    plot_degradation_efficiency(deg, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_degradation_efficiency_empty_input(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_deg.png"
    plot_degradation_efficiency({}, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Sort key tests
# ---------------------------------------------------------------------------


def test_site_sort_key_patch_embed_first() -> None:
    assert _site_sort_key("patch_embed/residual_stream") < _site_sort_key("blocks.0/pre_gelu")


def test_site_sort_key_numeric_order() -> None:
    assert _site_sort_key("blocks.2/pre_gelu") < _site_sort_key("blocks.10/pre_gelu")


def test_site_sort_key_full_sequence() -> None:
    site_ids = [
        "blocks.10/pre_gelu", "blocks.2/pre_gelu", "patch_embed/residual_stream",
        "blocks.0/pre_gelu", "blocks.11/pre_gelu", "blocks.1/pre_gelu",
    ]
    sorted_ids = sorted(site_ids, key=_site_sort_key)
    assert sorted_ids == [
        "patch_embed/residual_stream",
        "blocks.0/pre_gelu", "blocks.1/pre_gelu", "blocks.2/pre_gelu",
        "blocks.10/pre_gelu", "blocks.11/pre_gelu",
    ]


def test_site_sort_key_unknown_sorts_last() -> None:
    assert _site_sort_key("blocks.11/residual_stream") < _site_sort_key("unknown/site")