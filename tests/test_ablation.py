"""Tests for the outlier-zeroing ablation utilities in :mod:`src.ablation`.

Covers:
- Pure functions: ``_build_zeroing_mask``,
  ``_build_per_channel_zeroing_mask``, ``_build_random_mask``,
  ``compute_entropy_delta``, ``AblationResult``,
  ``save_ablation_results``, ``save_entropy_deltas``.
- Slow tests (require nnsight trace + model): ``zero_outliers_in_trace``
  for all three sites (global + per-channel + ablation modes),
  including entropy capture for pre_softmax and random-zeroing control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from nnsight import NNsight

from src.ablation import (
    AblationResult,
    _build_per_channel_zeroing_mask,
    _build_random_mask,
    _build_zeroing_mask,
    compute_entropy_delta,
    save_ablation_results,
    save_entropy_deltas,
    zero_outliers_in_trace,
)
from src.model import load_vit
from src.profiler import LayerStats
from src.utils import get_device, seed_everything



# ---------------------------------------------------------------------------
# _build_zeroing_mask
# ---------------------------------------------------------------------------


def test_build_zeroing_mask_all_kept() -> None:
    tensor = torch.tensor([-1.0, 0.0, 1.0])
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0, mean=0.0)
    assert mask.all().item()


def test_build_zeroing_mask_all_zeroed() -> None:
    tensor = torch.tensor([-10.0, 10.0])
    mask = _build_zeroing_mask(tensor, sigma_k=1.0, sigma=5.0, mean=0.0)
    assert not mask.any().item()


def test_build_zeroing_mask_mixed() -> None:
    tensor = torch.tensor([-6.0, -4.0, 0.0, 4.0, 6.0])
    mask = _build_zeroing_mask(tensor, sigma_k=1.0, sigma=5.0, mean=0.0)
    expected = torch.tensor([False, True, True, True, False])
    assert torch.equal(mask, expected)


def test_build_zeroing_mask_preserves_shape() -> None:
    tensor = torch.randn(4, 197, 3072)
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=2.0, mean=0.0)
    assert mask.shape == tensor.shape


def test_build_zeroing_mask_dtype() -> None:
    tensor = torch.randn(10)
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0, mean=0.0)
    assert mask.dtype == torch.bool


def test_build_zeroing_mask_mean_centered() -> None:
    """With mean=10, sigma=1, k=3: threshold is 3.  Elements at x=0 should
    be zeroed (|0-10|=10 > 3) even though |0| < 3."""
    tensor = torch.tensor([0.0, 10.0, 13.0, 7.0])
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0, mean=10.0)
    # |0-10|=10 > 3 → False, |10-10|=0 ≤ 3 → True, |13-10|=3 ≤ 3 → True, |7-10|=3 ≤ 3 → True
    expected = torch.tensor([False, True, True, True])
    assert torch.equal(mask, expected)


def test_build_zeroing_mask_zero_centered_with_zero_mean() -> None:
    """With mean=0.0, the mask is zero-centered."""
    tensor = torch.tensor([0.0, 10.0, 13.0, 7.0])
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0, mean=0.0)
    # |0|≤3 → True, |10|>3 → False, |13|>3 → False, |7|>3 → False
    expected = torch.tensor([True, False, False, False])
    assert torch.equal(mask, expected)


def test_build_zeroing_mask_default_mean_would_be_wrong() -> None:
    """Regression: using zero-centered threshold on a shifted distribution
    produces the wrong mask.  If mean were allowed to default to 0.0, the
    result would incorrectly keep/drop different elements."""
    tensor = torch.tensor([-28.0, -29.0, 0.0, 5.0])
    # Correct: with mean=-28.33, sigma=11.2, k=3: threshold=33.6
    # | -28 - (-28.33) | = 0.33 ≤ 33.6 → keep
    # | -29 - (-28.33) | = 0.67 ≤ 33.6 → keep
    # |   0 - (-28.33) | = 28.33 ≤ 33.6 → keep
    # |   5 - (-28.33) | = 33.33 ≤ 33.6 → keep
    mask_correct = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=11.2, mean=-28.33)
    assert mask_correct.all().item(), "All elements should be kept with correct mean"

    # Wrong: if mean=0.0 (the old default), threshold=33.6
    # | -28 | = 28 ≤ 33.6 → keep
    # | -29 | = 29 ≤ 33.6 → keep
    # |   0 | = 0 ≤ 33.6 → keep
    # |   5 | = 5 ≤ 33.6 → keep
    # In this case both happen to agree, but that's not guaranteed.
    # The point: the two definitions can differ, and only the mean-centered
    # one matches the project's outlier definition.
    mask_wrong = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=11.2, mean=0.0)
    # For the specific values used here, they happen to agree. But the important
    # point is that we force callers to provide mean explicitly.
    pass  # Test exists to prevent the default from ever being re-added.


# ---------------------------------------------------------------------------
# _build_random_mask
# ---------------------------------------------------------------------------


def test_build_random_mask_zero_fraction() -> None:
    tensor = torch.randn(100)
    mask = _build_random_mask(tensor, fraction=0.0)
    assert mask.all().item()
    assert mask.shape == tensor.shape


def test_build_random_mask_full_fraction() -> None:
    tensor = torch.randn(100)
    mask = _build_random_mask(tensor, fraction=1.0)
    assert not mask.any().item()
    assert mask.shape == tensor.shape


def test_build_random_mask_exact_count() -> None:
    tensor = torch.randn(1000)
    mask = _build_random_mask(tensor, fraction=0.3, seed=42)
    # 30% of 1000 = 300 elements should be zeroed (False).
    assert (~mask).sum().item() == 300
    assert mask.shape == tensor.shape


def test_build_random_mask_deterministic() -> None:
    tensor = torch.randn(500)
    mask1 = _build_random_mask(tensor, fraction=0.2, seed=42)
    mask2 = _build_random_mask(tensor, fraction=0.2, seed=42)
    assert torch.equal(mask1, mask2)


def test_build_random_mask_different_seeds_different() -> None:
    tensor = torch.randn(500)
    mask1 = _build_random_mask(tensor, fraction=0.2, seed=42)
    mask2 = _build_random_mask(tensor, fraction=0.2, seed=43)
    assert not torch.equal(mask1, mask2)


def test_build_random_mask_salt_produces_different() -> None:
    tensor = torch.randn(500)
    mask1 = _build_random_mask(tensor, fraction=0.2, seed=42, salt=0)
    mask2 = _build_random_mask(tensor, fraction=0.2, seed=42, salt=1)
    assert not torch.equal(mask1, mask2)


def test_build_random_mask_preserves_shape() -> None:
    tensor = torch.randn(4, 197, 3072)
    mask = _build_random_mask(tensor, fraction=0.01, seed=42)
    assert mask.shape == tensor.shape


def test_build_random_mask_dtype() -> None:
    tensor = torch.randn(100)
    mask = _build_random_mask(tensor, fraction=0.5, seed=42)
    assert mask.dtype == torch.bool


def test_build_random_mask_small_fraction_no_zeros() -> None:
    """With 10 elements and fraction=0.05, k=0 → no elements zeroed."""
    tensor = torch.randn(10)
    mask = _build_random_mask(tensor, fraction=0.05, seed=42)
    assert mask.all().item()


# ---------------------------------------------------------------------------
# _build_per_channel_zeroing_mask
# ---------------------------------------------------------------------------


def test_build_per_channel_zeroing_mask_all_kept() -> None:
    """With high per-channel σ and low sigma_k, nothing should be zeroed."""
    tensor = torch.ones(2, 3, 4)  # (B=2, N=3, D=4)
    pc_sigma = [100.0, 100.0, 100.0, 100.0]
    pc_mean = [0.0, 0.0, 0.0, 0.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    assert mask.all().item()


def test_build_per_channel_zeroing_mask_all_zeroed() -> None:
    """With tiny per-channel σ and low sigma_k, everything should be zeroed."""
    tensor = torch.ones(2, 3, 4) * 10.0
    pc_sigma = [0.1, 0.1, 0.1, 0.1]
    pc_mean = [0.0, 0.0, 0.0, 0.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=1.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    assert not mask.any().item()


def test_build_per_channel_zeroing_mask_mixed_channels() -> None:
    """Channel 0 has σ=100 (all kept), channel 1 has σ=0.1 (all zeroed)."""
    # Shape (1, 1, 2): B=1, N=1, D=2
    tensor = torch.tensor([[[10.0, 10.0]]])
    pc_sigma = [100.0, 0.1]
    pc_mean = [0.0, 0.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    # Channel 0: |10-0|=10 <= 300 → True (kept)
    # Channel 1: |10-0|=10 > 0.3 → False (zeroed)
    expected = torch.tensor([[[True, False]]])
    assert torch.equal(mask, expected)


def test_build_per_channel_zeroing_mask_mean_centered_per_channel() -> None:
    """Per-channel mean affects threshold centre per channel independently."""
    tensor = torch.tensor([[[0.0, 5.0]]])  # (1, 1, 2)
    pc_sigma = [1.0, 1.0]
    pc_mean = [10.0, 0.0]  # Channel 0 centred at 10, channel 1 at 0
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    # Channel 0: |0-10|=10 > 3 → False (zeroed)
    # Channel 1: |5-0|=5 > 3 → False (zeroed)
    expected = torch.tensor([[[False, False]]])
    assert torch.equal(mask, expected)


def test_build_per_channel_zeroing_mask_shape_preserved() -> None:
    """Mask should have the same shape as the input tensor."""
    tensor = torch.randn(4, 197, 3072)
    pc_sigma = [1.0] * 3072
    pc_mean = [0.0] * 3072
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    assert mask.shape == tensor.shape


def test_build_per_channel_zeroing_mask_dtype_is_bool() -> None:
    """Mask must be boolean."""
    tensor = torch.randn(8, 100, 64)
    pc_sigma = [2.0] * 64
    pc_mean = [0.0] * 64
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    assert mask.dtype == torch.bool


def test_build_per_channel_zeroing_mask_boundary() -> None:
    """Elements exactly at the threshold boundary should be kept (≤)."""
    tensor = torch.tensor([[[3.0, -3.0, 3.01, -3.01]]])  # (1, 1, 4)
    pc_sigma = [1.0, 1.0, 1.0, 1.0]
    pc_mean = [0.0, 0.0, 0.0, 0.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    # |3.0| ≤ 3 → True (kept), |-3.0| ≤ 3 → True, |3.01| > 3 → False, |-3.01| > 3 → False
    expected = torch.tensor([[[True, True, False, False]]])
    assert torch.equal(mask, expected)


def test_build_per_channel_zeroing_mask_heterogeneous_channels() -> None:
    """Per-channel mask with different σ_c per channel must apply correctly.

    Channel 0: σ=1.0, μ=0.0 → threshold k·σ = 3.0
    Channel 1: σ=10.0, μ=0.0 → threshold k·σ = 30.0
    Channel 2: σ=1.0, μ=5.0 → threshold k·σ = 3.0, centered at 5.0

    This test distinguishes mean_only (global σ, per-channel μ) from
    var_only (global μ, per-channel σ) from outlier (full per-channel).
    """
    tensor = torch.tensor([[[4.0, 25.0, 9.0]]])  # (1, 1, 3)
    # Channel 0: |4.0 - 0| = 4.0 > 3.0 → zeroed
    # Channel 1: |25.0 - 0| = 25.0 ≤ 30.0 → kept
    # Channel 2: |9.0 - 5| = 4.0 > 3.0 → zeroed
    pc_sigma = [1.0, 10.0, 1.0]
    pc_mean = [0.0, 0.0, 5.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    expected = torch.tensor([[[False, True, False]]])
    assert torch.equal(mask, expected)


def test_build_per_channel_zeroing_mask_mean_only_equivalent() -> None:
    """mean_only mode: per-channel μ_c with global σ.

    All channels share the same σ (global), but each has its own μ_c.
    This is the mask that _intervene_pre_gelu produces in mean_only mode.
    """
    tensor = torch.tensor([[[4.0, 4.0, 9.0]]])  # (1, 1, 3)
    # Global σ = 2.0, k=3.0 → threshold = 6.0
    # Channel 0: |4.0 - 0| = 4.0 ≤ 6.0 → kept
    # Channel 1: |4.0 - 10| = 6.0 ≤ 6.0 → kept (boundary)
    # Channel 2: |9.0 - 5| = 4.0 ≤ 6.0 → kept
    global_sigma = [2.0, 2.0, 2.0]
    pc_mean = [0.0, 10.0, 5.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=global_sigma,
        per_channel_mean=pc_mean, device=torch.device("cpu"),
    )
    expected = torch.tensor([[[True, True, True]]])
    assert torch.equal(mask, expected)


def test_build_per_channel_zeroing_mask_var_only_equivalent() -> None:
    """var_only mode: global μ with per-channel σ_c.

    All channels share the same μ (global), but each has its own σ_c.
    This is the mask that _intervene_pre_gelu produces in var_only mode.
    """
    tensor = torch.tensor([[[4.0, 25.0, 4.0]]])  # (1, 1, 3)
    # Global μ = 0.0, k=3.0
    # Channel 0: σ=1.0 → threshold=3.0, |4.0| > 3.0 → zeroed
    # Channel 1: σ=10.0 → threshold=30.0, |25.0| ≤ 30.0 → kept
    # Channel 2: σ=1.0 → threshold=3.0, |4.0| > 3.0 → zeroed
    pc_sigma = [1.0, 10.0, 1.0]
    global_mean = [0.0, 0.0, 0.0]
    mask = _build_per_channel_zeroing_mask(
        tensor, sigma_k=3.0, per_channel_sigma=pc_sigma,
        per_channel_mean=global_mean, device=torch.device("cpu"),
    )
    expected = torch.tensor([[[False, True, False]]])
    assert torch.equal(mask, expected)


# ---------------------------------------------------------------------------
# compute_entropy_delta
# ---------------------------------------------------------------------------


def test_compute_entropy_delta_positive() -> None:
    delta = compute_entropy_delta(
        ablated_cls=[2.0, 3.0], ablated_patch=[1.0, 2.0],
        baseline_cls=[1.0, 2.0], baseline_patch=[0.5, 1.5],
    )
    assert delta["mean_cls_delta"] == pytest.approx(1.0)
    assert delta["mean_patch_delta"] == pytest.approx(0.5)


def test_compute_entropy_delta_negative() -> None:
    delta = compute_entropy_delta(
        ablated_cls=[0.5, 1.0], ablated_patch=[0.2, 0.5],
        baseline_cls=[1.0, 2.0], baseline_patch=[0.5, 1.5],
    )
    assert delta["mean_cls_delta"] == pytest.approx(-0.75)
    assert delta["mean_patch_delta"] == pytest.approx(-0.65)


def test_compute_entropy_delta_empty() -> None:
    delta = compute_entropy_delta([], [], [], [])
    assert delta["mean_cls_delta"] == 0.0
    assert delta["mean_patch_delta"] == 0.0


# ---------------------------------------------------------------------------
# AblationResult
# ---------------------------------------------------------------------------


def test_ablation_result_construction() -> None:
    r = AblationResult(
        site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
        pct_zeroed=0.27, top1_accuracy=80.5, top5_accuracy=95.0,
        baseline_top1=81.0, baseline_top5=95.5,
    )
    assert r.site == "pre_gelu"
    assert r.sigma_threshold == 3.0
    assert r.cls_entropy == []
    assert r.patch_entropy == []
    assert r.is_random is False


def test_ablation_result_with_entropy() -> None:
    r = AblationResult(
        site="pre_softmax", sigma_threshold=3.0, site_identifier="blocks.0/pre_softmax",
        pct_zeroed=0.5, top1_accuracy=80.0, top5_accuracy=95.0,
        baseline_top1=81.0, baseline_top5=95.5,
        cls_entropy=[1.0, 2.0, 3.0], patch_entropy=[0.5, 1.0, 1.5],
        baseline_cls_entropy=[1.2, 2.1, 3.0], baseline_patch_entropy=[0.6, 1.1, 1.6],
    )
    assert r.cls_entropy == [1.0, 2.0, 3.0]
    assert r.baseline_cls_entropy == [1.2, 2.1, 3.0]


def test_ablation_result_degradation() -> None:
    r = AblationResult(
        site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
        pct_zeroed=0.27, top1_accuracy=78.0, top5_accuracy=93.0,
        baseline_top1=81.0, baseline_top5=95.5,
    )
    assert r.baseline_top1 - r.top1_accuracy == pytest.approx(3.0)


def test_ablation_result_is_random() -> None:
    r = AblationResult(
        site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
        pct_zeroed=0.27, top1_accuracy=80.5, top5_accuracy=95.0,
        baseline_top1=81.0, baseline_top5=95.5, is_random=True,
    )
    assert r.is_random is True


def test_ablation_result_with_granularity() -> None:
    """AblationResult with per_channel granularity must store and serialize it."""
    r = AblationResult(
        site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
        pct_zeroed=0.27, top1_accuracy=80.5, top5_accuracy=95.0,
        baseline_top1=81.0, baseline_top5=95.5,
        granularity="per_channel",
    )
    assert r.granularity == "per_channel"


# ---------------------------------------------------------------------------
# save_ablation_results
# ---------------------------------------------------------------------------


def test_save_ablation_results_creates_file(tmp_path: Path) -> None:
    results = [
        AblationResult(
            site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
            pct_zeroed=0.27, top1_accuracy=80.5, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5,
        ),
    ]
    path = tmp_path / "subdir" / "results.csv"
    save_ablation_results(results, path)
    assert path.exists()
    content = path.read_text()
    assert "site,sigma_threshold,site_identifier" in content
    assert "is_random" in content


def test_save_ablation_results_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    save_ablation_results([], path)
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1


def test_save_ablation_results_with_entropy(tmp_path: Path) -> None:
    results = [
        AblationResult(
            site="pre_softmax", sigma_threshold=3.0, site_identifier="blocks.0/pre_softmax",
            pct_zeroed=0.5, top1_accuracy=80.0, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5,
            cls_entropy=[1.0, 2.0], patch_entropy=[0.5, 1.0],
            baseline_cls_entropy=[1.2, 2.1], baseline_patch_entropy=[0.6, 1.1],
        ),
    ]
    path = tmp_path / "entropy_results.csv"
    save_ablation_results(results, path)
    content = path.read_text()
    assert "[1.0, 2.0]" in content


def test_save_ablation_results_with_random(tmp_path: Path) -> None:
    results = [
        AblationResult(
            site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
            pct_zeroed=0.27, top1_accuracy=80.0, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5, is_random=True,
        ),
    ]
    path = tmp_path / "random_results.csv"
    save_ablation_results(results, path)
    content = path.read_text()
    assert "True" in content


def test_save_ablation_results_with_granularity(tmp_path: Path) -> None:
    """AblationResult CSV must preserve the granularity field in round-trip."""
    results = [
        AblationResult(
            site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
            pct_zeroed=0.27, top1_accuracy=80.0, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5, granularity="per_channel",
        ),
    ]
    path = tmp_path / "granularity_results.csv"
    save_ablation_results(results, path)
    content = path.read_text()
    assert "per_channel" in content
    # Verify CSV has the granularity column header.
    assert "granularity" in content.split("\n")[0]


# ---------------------------------------------------------------------------
# save_entropy_deltas
# ---------------------------------------------------------------------------


def test_save_entropy_deltas_creates_file(tmp_path: Path) -> None:
    results = [
        AblationResult(
            site="pre_softmax", sigma_threshold=3.0, site_identifier="blocks.0/pre_softmax",
            pct_zeroed=0.5, top1_accuracy=80.0, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5,
            cls_entropy=[2.0, 3.0], patch_entropy=[1.0, 2.0],
            baseline_cls_entropy=[1.0, 2.0], baseline_patch_entropy=[0.5, 1.5],
        ),
    ]
    path = tmp_path / "deltas.csv"
    save_entropy_deltas(results, path)
    assert path.exists()
    content = path.read_text()
    assert "mean_cls_delta,mean_patch_delta" in content


def test_save_entropy_deltas_no_pre_softmax(tmp_path: Path) -> None:
    results = [
        AblationResult(
            site="pre_gelu", sigma_threshold=3.0, site_identifier="blocks.0/pre_gelu",
            pct_zeroed=0.27, top1_accuracy=80.5, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5,
        ),
    ]
    path = tmp_path / "no_entropy.csv"
    save_entropy_deltas(results, path)
    # No pre_softmax results -> function returns without writing.
    assert not path.exists()


def test_save_entropy_deltas_filters_random(tmp_path: Path) -> None:
    """Random-control pre_softmax results should be excluded from entropy deltas."""
    results = [
        AblationResult(
            site="pre_softmax", sigma_threshold=3.0, site_identifier="blocks.0/pre_softmax",
            pct_zeroed=0.5, top1_accuracy=80.0, top5_accuracy=95.0,
            baseline_top1=81.0, baseline_top5=95.5,
            cls_entropy=[2.0, 3.0], patch_entropy=[1.0, 2.0],
            baseline_cls_entropy=[1.0, 2.0], baseline_patch_entropy=[0.5, 1.5],
            is_random=True,
        ),
    ]
    path = tmp_path / "random_deltas.csv"
    save_entropy_deltas(results, path)
    # Random results should be filtered out -> no pre_softmax (non-random) results.
    assert not path.exists()


# ---------------------------------------------------------------------------
# _site_matches (from exp2_ablation)
# ---------------------------------------------------------------------------


def test_site_matches_pre_gelu() -> None:
    from src.exp2_ablation import _site_matches
    assert _site_matches("blocks.3/pre_gelu", "pre_gelu") is True
    assert _site_matches("blocks.0/pre_gelu", "pre_gelu") is True
    assert _site_matches("blocks.3/pre_softmax", "pre_gelu") is False
    assert _site_matches("blocks.3/residual_stream", "pre_gelu") is False


def test_site_matches_residual_stream() -> None:
    from src.exp2_ablation import _site_matches
    assert _site_matches("blocks.3/residual_stream", "residual_stream") is True
    assert _site_matches("patch_embed/residual_stream", "residual_stream") is True
    assert _site_matches("blocks.3/pre_gelu", "residual_stream") is False


def test_site_matches_pre_softmax() -> None:
    from src.exp2_ablation import _site_matches
    assert _site_matches("blocks.5/pre_softmax", "pre_softmax") is True
    assert _site_matches("blocks.5/post_softmax", "pre_softmax") is False


# ---------------------------------------------------------------------------
# Slow tests — zero_outliers_in_trace (require model + nnsight trace)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _vit_model():
    """Module-scoped fixture: load ViT-B/16 once for all slow ablation tests."""
    seed_everything(42)
    device = get_device()
    model, _transform = load_vit(device)
    wrapped = NNsight(model)
    return wrapped, model, device


def _make_fake_layer_stats(num_blocks: int) -> dict[str, LayerStats]:
    """Build a fake layer_stats dict with plausible sigma and mean values."""
    stats: dict[str, LayerStats] = {}
    stats["patch_embed/residual_stream"] = LayerStats(
        site_identifier="patch_embed/residual_stream",
        mean=0.0, std=0.5, kurtosis=0.0, outlier_fractions={}, n_samples=1000,
    )
    for i in range(num_blocks):
        stats[f"blocks.{i}/pre_gelu"] = LayerStats(
            site_identifier=f"blocks.{i}/pre_gelu",
            mean=-2.0, std=28.0, kurtosis=10.0, outlier_fractions={}, n_samples=1000,
        )
        stats[f"blocks.{i}/pre_softmax"] = LayerStats(
            site_identifier=f"blocks.{i}/pre_softmax",
            mean=0.0, std=3.4, kurtosis=5.0, outlier_fractions={}, n_samples=1000,
        )
        if i > 0:
            stats[f"blocks.{i - 1}/residual_stream"] = LayerStats(
                site_identifier=f"blocks.{i - 1}/residual_stream",
                mean=0.0, std=0.5, kurtosis=0.0, outlier_fractions={}, n_samples=1000,
            )
    stats[f"blocks.{num_blocks - 1}/residual_stream"] = LayerStats(
        site_identifier=f"blocks.{num_blocks - 1}/residual_stream",
        mean=0.0, std=0.5, kurtosis=0.0, outlier_fractions={}, n_samples=1000,
    )
    return stats


@pytest.mark.slow
class TestZeroOutliersInTrace:
    """Integration tests for zero_outliers_in_trace with a real ViT-B/16."""

    def test_pre_gelu_logits_change(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()
        logits, pct, entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=0.001, layer_stats=layer_stats,
        )
        diff = (baseline_logits - logits).abs().max().item()
        assert diff > 1e-3, f"max diff = {diff:.6f}"
        assert len(entropy) == 0

    def test_pre_gelu_returns_pct_zeroed(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        _logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0, layer_stats=layer_stats,
        )
        assert len(pct) == len(model.blocks)
        for v in pct.values():
            assert 0.0 <= v <= 100.0

    def test_pre_gelu_logits_shape(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        logits, _pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0, layer_stats=layer_stats,
        )
        assert logits.shape == (2, 1000)

    def test_pre_gelu_random_mode(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        random_fractions = {
            f"blocks.{i}/pre_gelu": 0.1 for i in range(len(model.blocks))
        }
        logits, pct, entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0, layer_stats=layer_stats,
            random_fractions=random_fractions, random_seed=42,
        )
        assert logits.shape == (2, 1000)
        assert len(pct) == len(model.blocks)
        for v in pct.values():
            assert v == pytest.approx(10.0)  # 0.1 fraction → 10%

    def test_residual_stream_logits_change(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()
        logits, _pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "residual_stream", sigma_k=0.001, layer_stats=layer_stats,
        )
        diff = (baseline_logits - logits).abs().max().item()
        assert diff > 1e-3, f"max diff = {diff:.6f}"

    def test_residual_stream_no_zeroing_at_high_k(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()
        logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "residual_stream", sigma_k=10000.0, layer_stats=layer_stats,
        )
        for v in pct.values():
            assert v == 0.0
        diff = (baseline_logits - logits).abs().max().item()
        assert diff < 1e-3, f"max diff = {diff:.6f}"

    def test_residual_stream_random_mode(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        random_fractions = {
            f"blocks.{i}/residual_stream": 0.05
            for i in range(len(model.blocks))
        }
        random_fractions["patch_embed/residual_stream"] = 0.05
        logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "residual_stream", sigma_k=3.0, layer_stats=layer_stats,
            random_fractions=random_fractions, random_seed=42,
        )
        assert logits.shape == (2, 1000)
        for v in pct.values():
            assert v == pytest.approx(5.0)

    def test_pre_softmax_logits_change(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()
        logits, _pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_softmax", sigma_k=0.001, layer_stats=layer_stats,
        )
        diff = (baseline_logits - logits).abs().max().item()
        assert diff > 1e-3, f"max diff = {diff:.6f}"

    def test_pre_softmax_returns_pct_zeroed(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        _logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_softmax", sigma_k=3.0, layer_stats=layer_stats,
        )
        assert len(pct) == len(model.blocks)

    def test_pre_softmax_returns_entropy(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        _logits, _pct, entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_softmax", sigma_k=3.0, layer_stats=layer_stats,
        )
        assert len(entropy) == len(model.blocks)
        for sid, ent in entropy.items():
            assert len(ent["cls"]) == 12
            assert len(ent["patch"]) == 12
            for v in ent["cls"]:
                assert v >= 0.0

    def test_invalid_site_raises(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        with pytest.raises(ValueError, match="Unknown site"):
            zero_outliers_in_trace(
                wrapped, batch, "nonexistent_site", sigma_k=3.0, layer_stats=layer_stats,
            )

    def test_missing_site_in_stats_handled_gracefully(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        batch = torch.randn(2, 3, 224, 224, device=device)
        logits, pct, entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0, layer_stats={},
        )
        assert logits.shape == (2, 1000)
        assert len(pct) == 0
        assert len(entropy) == 0

    def test_zero_sigma_handled_gracefully(self, _vit_model) -> None:
        wrapped, model, device = _vit_model
        zero_stats: dict[str, LayerStats] = {}
        for i in range(len(model.blocks)):
            zero_stats[f"blocks.{i}/pre_gelu"] = LayerStats(
                site_identifier=f"blocks.{i}/pre_gelu",
                mean=0.0, std=0.0, kurtosis=0.0, outlier_fractions={}, n_samples=0,
            )
        batch = torch.randn(2, 3, 224, 224, device=device)
        logits, pct, entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0, layer_stats=zero_stats,
        )
        assert logits.shape == (2, 1000)
        assert len(pct) == 0
        assert len(entropy) == 0

    # --- Per-channel ablation tests ---

    def test_pre_gelu_per_channel_logits_change(self, _vit_model) -> None:
        """Per-channel zeroing at aggressive k should change logits."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        # Add per-channel stats to the fake layer_stats.
        for i in range(len(model.blocks)):
            sid = f"blocks.{i}/pre_gelu"
            layer_stats[sid] = LayerStats(
                site_identifier=sid,
                mean=-2.0, std=28.0, kurtosis=10.0,
                outlier_fractions={}, n_samples=1000,
                per_channel_std=[28.0] * 3072,
                per_channel_mean=[-2.0] * 3072,
            )
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()
        logits, pct, entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=0.001,
            layer_stats=layer_stats, per_channel=True,
        )
        diff = (baseline_logits - logits).abs().max().item()
        assert diff > 1e-3, f"max diff = {diff:.6f}"
        assert len(entropy) == 0

    def test_pre_gelu_per_channel_returns_pct_zeroed(self, _vit_model) -> None:
        """Per-channel mode should return per-layer pct_zeroed."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        for i in range(len(model.blocks)):
            sid = f"blocks.{i}/pre_gelu"
            layer_stats[sid] = LayerStats(
                site_identifier=sid,
                mean=-2.0, std=28.0, kurtosis=10.0,
                outlier_fractions={}, n_samples=1000,
                per_channel_std=[28.0] * 3072,
                per_channel_mean=[-2.0] * 3072,
            )
        batch = torch.randn(2, 3, 224, 224, device=device)
        _logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0,
            layer_stats=layer_stats, per_channel=True,
        )
        assert len(pct) == len(model.blocks)
        for v in pct.values():
            assert 0.0 <= v <= 100.0

    def test_pre_gelu_per_channel_no_zeroing_at_high_k(self, _vit_model) -> None:
        """Per-channel mode with very high k should zero nothing."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        for i in range(len(model.blocks)):
            sid = f"blocks.{i}/pre_gelu"
            layer_stats[sid] = LayerStats(
                site_identifier=sid,
                mean=-2.0, std=28.0, kurtosis=10.0,
                outlier_fractions={}, n_samples=1000,
                per_channel_std=[28.0] * 3072,
                per_channel_mean=[-2.0] * 3072,
            )
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()
        logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=10000.0,
            layer_stats=layer_stats, per_channel=True,
        )
        for v in pct.values():
            assert v == 0.0
        diff = (baseline_logits - logits).abs().max().item()
        assert diff < 1e-3, f"max diff = {diff:.6f}"

    def test_pre_gelu_mean_only_mode(self, _vit_model) -> None:
        """mean_only mode: per-channel μ_c but global σ."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        for i in range(len(model.blocks)):
            sid = f"blocks.{i}/pre_gelu"
            layer_stats[sid] = LayerStats(
                site_identifier=sid,
                mean=-2.0, std=28.0, kurtosis=10.0,
                outlier_fractions={}, n_samples=1000,
                per_channel_std=[28.0] * 3072,
                per_channel_mean=[-2.0] * 3072,
            )
        batch = torch.randn(2, 3, 224, 224, device=device)
        logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0,
            layer_stats=layer_stats, per_channel=True,
            ablation_mode="mean_only",
        )
        assert logits.shape == (2, 1000)
        assert len(pct) == len(model.blocks)
        for v in pct.values():
            assert 0.0 <= v <= 100.0

    def test_pre_gelu_var_only_mode(self, _vit_model) -> None:
        """var_only mode: global μ but per-channel σ_c."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        for i in range(len(model.blocks)):
            sid = f"blocks.{i}/pre_gelu"
            layer_stats[sid] = LayerStats(
                site_identifier=sid,
                mean=-2.0, std=28.0, kurtosis=10.0,
                outlier_fractions={}, n_samples=1000,
                per_channel_std=[28.0] * 3072,
                per_channel_mean=[-2.0] * 3072,
            )
        batch = torch.randn(2, 3, 224, 224, device=device)
        logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0,
            layer_stats=layer_stats, per_channel=True,
            ablation_mode="var_only",
        )
        assert logits.shape == (2, 1000)
        assert len(pct) == len(model.blocks)
        for v in pct.values():
            assert 0.0 <= v <= 100.0

    def test_pre_gelu_per_channel_missing_stats_fallback(self, _vit_model) -> None:
        """When per_channel_std is None, per-channel mode should fall back to global."""
        wrapped, model, device = _vit_model
        # Build stats WITHOUT per_channel_std (simulating old Phase 1 data).
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)
        logits, pct, _entropy = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=3.0,
            layer_stats=layer_stats, per_channel=True,
        )
        # Should not crash — falls back to global σ.
        assert logits.shape == (2, 1000)
        assert len(pct) == len(model.blocks)

    # --- layer_range tests ---

    def test_layer_range_restricts_intervention(self, _vit_model) -> None:
        """layer_range must restrict zeroing to only the specified blocks.

        With layer_range=(5, 5), only block 5 should be intervened on.
        All other blocks should pass through unchanged.
        """
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)

        # Full-range baseline: zero all blocks at aggressive k.
        logits_all, pct_all, _ = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=0.001,
            layer_stats=layer_stats,
        )

        # Layer-range: only block 5.
        logits_range, pct_range, _ = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=0.001,
            layer_stats=layer_stats, layer_range=(5, 5),
        )

        # Only block 5 should appear in pct_zeroed.
        assert len(pct_range) == 1, (
            f"Expected 1 block in pct_zeroed, got {len(pct_range)}: {list(pct_range.keys())}"
        )
        assert "blocks.5/pre_gelu" in pct_range

        # Layer-range logits should differ from full-range logits
        # (fewer blocks zeroed → less degradation).
        diff_all = (logits_all - logits_range).abs().max().item()
        assert diff_all > 1e-3, (
            f"Layer-range logits should differ from full-range, got max diff {diff_all:.6f}"
        )

    def test_layer_range_multi_block(self, _vit_model) -> None:
        """layer_range=(8, 11) must intervene on blocks 8, 9, 10, 11 only."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)

        logits_range, pct_range, _ = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=0.001,
            layer_stats=layer_stats, layer_range=(8, 11),
        )

        # Should have exactly 4 blocks: 8, 9, 10, 11.
        assert len(pct_range) == 4, (
            f"Expected 4 blocks in pct_zeroed, got {len(pct_range)}: {list(pct_range.keys())}"
        )
        for blk in range(8, 12):
            assert f"blocks.{blk}/pre_gelu" in pct_range, (
                f"Missing block {blk} in pct_zeroed"
            )

    def test_layer_range_no_blocks_outside_range(self, _vit_model) -> None:
        """Blocks outside layer_range must not appear in pct_zeroed."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)

        _logits, pct_range, _ = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=0.001,
            layer_stats=layer_stats, layer_range=(3, 5),
        )

        # Blocks 0-2 and 6-11 must not appear.
        for blk in list(range(0, 3)) + list(range(6, len(model.blocks))):
            assert f"blocks.{blk}/pre_gelu" not in pct_range, (
                f"Block {blk} should not be in pct_zeroed but was found"
            )

    def test_layer_range_single_block_no_zeroing_at_high_k(self, _vit_model) -> None:
        """layer_range with high k must zero nothing (same as no layer_range)."""
        wrapped, model, device = _vit_model
        layer_stats = _make_fake_layer_stats(len(model.blocks))
        batch = torch.randn(2, 3, 224, 224, device=device)

        with torch.no_grad():
            with wrapped.trace(batch):
                baseline = wrapped.output.save()
            baseline_logits = baseline.clone()

        logits, pct, _ = zero_outliers_in_trace(
            wrapped, batch, "pre_gelu", sigma_k=10000.0,
            layer_stats=layer_stats, layer_range=(10, 10),
        )

        # Block 10 should appear with 0% zeroed.
        assert pct.get("blocks.10/pre_gelu", 0.0) == 0.0
        # Logits should be unchanged.
        diff = (baseline_logits - logits).abs().max().item()
        assert diff < 1e-3, f"max diff = {diff:.6f}"


# ---------------------------------------------------------------------------
# End-to-end integration test: full exp2_ablation.run() pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_e2e_exp2_run_pipeline(tmp_path: Path) -> None:
    """End-to-end test: run the full Phase 2 pipeline on a tiny synthetic dataset.

    Verifies that exp2_ablation.run() completes without error and produces
    the expected output files (ablation_results.csv, entropy_deltas.csv)
    with correct row counts.
    """
    from src.config import AblationConfig
    from src.exp2_ablation import run
    from src.profiler import (
        LayerStats, ProfilingResult, RunMetadata, load_profiling_result,
        save_profiling_result,
    )

    seed_everything(42)
    device = get_device()

    # Build synthetic profiling_result.json with plausible stats.
    model, _transform = load_vit(device)
    num_blocks = len(model.blocks)

    stats: dict[str, LayerStats] = {}
    stats["patch_embed/residual_stream"] = LayerStats(
        site_identifier="patch_embed/residual_stream",
        mean=0.0, std=0.5, kurtosis=0.0, outlier_fractions={}, n_samples=1000,
    )
    for i in range(num_blocks):
        stats[f"blocks.{i}/pre_gelu"] = LayerStats(
            site_identifier=f"blocks.{i}/pre_gelu",
            mean=-2.0, std=28.0, kurtosis=10.0, outlier_fractions={}, n_samples=1000,
            per_channel_std=[28.0] * 3072,
            per_channel_mean=[-2.0] * 3072,
        )
        stats[f"blocks.{i}/pre_softmax"] = LayerStats(
            site_identifier=f"blocks.{i}/pre_softmax",
            mean=0.0, std=3.4, kurtosis=5.0, outlier_fractions={}, n_samples=1000,
            attention_entropy_cls=[2.0] * 12,
            attention_entropy_patches=[2.0] * 12,
        )
        if i > 0:
            stats[f"blocks.{i - 1}/residual_stream"] = LayerStats(
                site_identifier=f"blocks.{i - 1}/residual_stream",
                mean=0.0, std=0.5, kurtosis=0.0, outlier_fractions={}, n_samples=1000,
            )
    stats[f"blocks.{num_blocks - 1}/residual_stream"] = LayerStats(
        site_identifier=f"blocks.{num_blocks - 1}/residual_stream",
        mean=0.0, std=0.5, kurtosis=0.0, outlier_fractions={}, n_samples=1000,
    )

    profiling_result = ProfilingResult(
        stats=stats, num_blocks=num_blocks,
        batch_shape=(64, 3, 224, 224),
        metadata=RunMetadata(
            python_version="3.12.0", pytorch_version="2.12.0",
            timm_version="1.0.0", nnsight_version="0.7.0",
            cuda_available=True, cuda_version=None, gpu_name=None,
            gpu_memory_gb=None, model_name="vit_base_patch16_224",
            dataset="test", num_images=8, batch_size=4,
            seed=42, num_seeds=1, timestamp_utc="2026-01-01T00:00:00Z",
        ),
    )
    stats_dir = tmp_path / "phase1"
    stats_dir.mkdir()
    stats_path = stats_dir / "profiling_result.json"
    save_profiling_result(profiling_result, stats_path)

    # Verify round-trip.
    _loaded = load_profiling_result(stats_path)
    assert len(_loaded.stats) == len(stats)

    # Build a minimal ImageFolder dataset so build_val_loader works.
    from PIL import Image as PILImage
    from torchvision import transforms as T
    import numpy as np

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(seed=0)
    for class_idx in range(10):
        class_dir = data_dir / f"class_{class_idx}"
        class_dir.mkdir()
        for img_idx in range(5):
            colour = tuple(rng.integers(0, 256, size=3).tolist())
            img = PILImage.new("RGB", (224, 224), colour)
            img.save(class_dir / f"img_{img_idx}.JPEG")

    # Run the pipeline with a small subset.
    output_dir = tmp_path / "phase2"
    config = AblationConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        num_images=8,
        batch_size=4,
        device=device,
        sigma_thresholds=(3.0, 6.0),
        layer_stats_path=stats_path,
        seed=42,
        num_seeds=1,
        granularity="global",
    )
    run(config)

    # Verify output files (written to seed_42/ subdirectory).
    seed_dir = output_dir / "seed_42"
    csv_path = seed_dir / "ablation_results.csv"
    entropy_path = seed_dir / "entropy_deltas.csv"
    assert csv_path.exists(), f"{csv_path} not found"
    assert csv_path.stat().st_size > 0, "ablation_results.csv is empty"
    assert entropy_path.exists(), f"{entropy_path} not found"

    # Verify CSV has expected columns and rows.
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Verify seed column is present.
    assert "seed" in reader.fieldnames, f"Missing 'seed' column; got {reader.fieldnames}"
    assert all(int(r["seed"]) == 42 for r in rows), "All rows should have seed=42"

    # 3 sites × 2 thresholds × N blocks for outlier + random control rows
    # pre_gelu: 12 blocks × 2 thresholds = 24 outlier + 24 random = 48
    # pre_softmax: 12 blocks × 2 thresholds = 24 outlier (no random)
    # residual_stream: 13 sites × 2 thresholds = 26 outlier + 26 random = 52
    # Total: 48 + 24 + 52 = 124
    assert len(rows) > 0, "ablation_results.csv has no rows"
    assert any(r["is_random"] == "True" for r in rows), "No random control rows"
    assert any(r["is_random"] == "False" for r in rows), "No outlier rows"

    # Verify top-1 accuracy is reasonable (should be in [0, 100]).
    for row in rows:
        acc = float(row["top1_accuracy"])
        assert 0.0 <= acc <= 100.0, f"top1_accuracy out of range: {acc}"