"""Tests for the outlier-zeroing ablation utilities in :mod:`src.ablation`.

Covers:
- Pure functions: ``compute_pct_zeroed``, ``_build_zeroing_mask``,
  ``_build_random_mask``, ``compute_entropy_delta``, ``AblationResult``,
  ``save_ablation_results``, ``save_entropy_deltas``.
- Slow tests (require nnsight trace + model): ``zero_outliers_in_trace``
  for all three sites, including entropy capture for pre_softmax and
  random-zeroing control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from nnsight import NNsight

from src.ablation import (
    AblationResult,
    _build_random_mask,
    _build_zeroing_mask,
    compute_entropy_delta,
    compute_pct_zeroed,
    save_ablation_results,
    save_entropy_deltas,
    zero_outliers_in_trace,
)
from src.model import load_vit
from src.profiler import LayerStats
from src.utils import get_device, seed_everything


# ---------------------------------------------------------------------------
# compute_pct_zeroed
# ---------------------------------------------------------------------------


def test_compute_pct_zeroed_all_below_threshold(dummy_tensor: torch.Tensor) -> None:
    tensor = torch.full_like(dummy_tensor, 0.1)
    result = compute_pct_zeroed(tensor, threshold=1.0)
    assert result == pytest.approx(0.0)


def test_compute_pct_zeroed_all_above_threshold(dummy_tensor: torch.Tensor) -> None:
    tensor = torch.full_like(dummy_tensor, 10.0)
    result = compute_pct_zeroed(tensor, threshold=1.0)
    assert result == pytest.approx(100.0)


def test_compute_pct_zeroed_mixed() -> None:
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, -3.0, 4.0, 0.5, 1.5, 2.0, -1.0])
    result = compute_pct_zeroed(tensor, threshold=2.0)
    assert result == pytest.approx(30.0)


def test_compute_pct_zeroed_empty_tensor() -> None:
    tensor = torch.tensor([])
    result = compute_pct_zeroed(tensor, threshold=1.0)
    assert result == 0.0


def test_compute_pct_zeroed_exact_boundary() -> None:
    tensor = torch.tensor([-5.0, -3.0, 0.0, 3.0, 5.0])
    result = compute_pct_zeroed(tensor, threshold=3.0)
    assert result == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# _build_zeroing_mask
# ---------------------------------------------------------------------------


def test_build_zeroing_mask_all_kept() -> None:
    tensor = torch.tensor([-1.0, 0.0, 1.0])
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0)
    assert mask.all().item()


def test_build_zeroing_mask_all_zeroed() -> None:
    tensor = torch.tensor([-10.0, 10.0])
    mask = _build_zeroing_mask(tensor, sigma_k=1.0, sigma=5.0)
    assert not mask.any().item()


def test_build_zeroing_mask_mixed() -> None:
    tensor = torch.tensor([-6.0, -4.0, 0.0, 4.0, 6.0])
    mask = _build_zeroing_mask(tensor, sigma_k=1.0, sigma=5.0)
    expected = torch.tensor([False, True, True, True, False])
    assert torch.equal(mask, expected)


def test_build_zeroing_mask_preserves_shape() -> None:
    tensor = torch.randn(4, 197, 3072)
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=2.0)
    assert mask.shape == tensor.shape


def test_build_zeroing_mask_dtype() -> None:
    tensor = torch.randn(10)
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0)
    assert mask.dtype == torch.bool


def test_build_zeroing_mask_mean_centered() -> None:
    """With mean=10, sigma=1, k=3: threshold is 3.  Elements at x=0 should
    be zeroed (|0-10|=10 > 3) even though |0| < 3."""
    tensor = torch.tensor([0.0, 10.0, 13.0, 7.0])
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0, mean=10.0)
    # |0-10|=10 > 3 → False, |10-10|=0 ≤ 3 → True, |13-10|=3 ≤ 3 → True, |7-10|=3 ≤ 3 → True
    expected = torch.tensor([False, True, True, True])
    assert torch.equal(mask, expected)


def test_build_zeroing_mask_mean_centered_default_zero() -> None:
    """Without mean, defaults to zero-centered (backward compatible)."""
    tensor = torch.tensor([0.0, 10.0, 13.0, 7.0])
    mask = _build_zeroing_mask(tensor, sigma_k=3.0, sigma=1.0)
    # |0|≤3 → True, |10|>3 → False, |13|>3 → False, |7|>3 → False
    expected = torch.tensor([True, False, False, False])
    assert torch.equal(mask, expected)


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