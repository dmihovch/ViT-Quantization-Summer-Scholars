"""Tests for the hook registration and statistics machinery in :mod:`src.hooks`.

All tests operate on small synthetic modules — no real ViT weights are loaded.
The test model is a minimal transformer-like structure with nn.GELU and
nn.LayerNorm layers, mirroring the module types that register_profiling_hooks
targets in the real ViT.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.exceptions import HookRegistrationError
from src.hooks import (
    HookHandle,
    LayerStats,
    SITE_POST_LAYERNORM,
    SITE_PRE_GELU,
    SITE_RESIDUAL_STREAM,
    _finalize_accumulator,
    _SiteAccumulator,
    _update_accumulator,
    load_stats,
    register_profiling_hooks,
    remove_hooks,
    save_stats,
)


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic models
# ---------------------------------------------------------------------------


class _GELUBlock(nn.Module):
    """Minimal block with nn.LayerNorm → nn.Linear → nn.GELU."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(self.norm(x)))


class _TinyViT(nn.Module):
    """Two stacked _GELUBlocks — enough to test multi-layer hook coverage."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.block0 = _GELUBlock(dim)
        self.block1 = _GELUBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block1(self.block0(x))


def _run_forward(model: nn.Module, n_batches: int = 2, dim: int = 16) -> None:
    """Push ``n_batches`` random tensors through ``model``.

    Intentionally not wrapped in ``torch.no_grad()`` — the hook callbacks
    themselves use ``torch.no_grad()`` internally, and omitting it at the
    call site tests that the full stack (hooks + module forward) works
    correctly with autograd enabled.
    """
    generator = torch.Generator()
    generator.manual_seed(42)
    for _ in range(n_batches):
        # Shape (B=4, N=8, D=dim) — mimics (batch, tokens, channels)
        x = torch.randn(4, 8, dim, generator=generator)
        model(x)


# ---------------------------------------------------------------------------
# Registration guard
# ---------------------------------------------------------------------------


def test_register_hooks_raises_on_model_with_no_gelu() -> None:
    """register_profiling_hooks must raise HookRegistrationError for GELU-free models."""
    model = nn.Linear(4, 4)
    with pytest.raises(HookRegistrationError):
        register_profiling_hooks(model)


def test_register_hooks_raises_on_layernorm_only_model() -> None:
    """A model with LayerNorm but no GELU must still raise HookRegistrationError."""
    model = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8))
    with pytest.raises(HookRegistrationError):
        register_profiling_hooks(model)


# ---------------------------------------------------------------------------
# Hook registration and handle structure
# ---------------------------------------------------------------------------


def test_register_hooks_returns_hook_handle() -> None:
    """register_profiling_hooks should return a HookHandle with non-empty handles."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    assert isinstance(handle, HookHandle)
    assert len(handle.handles) > 0
    remove_hooks(handle)


def test_stats_empty_before_remove_hooks() -> None:
    """handle.stats must be empty until remove_hooks is called."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    assert handle.stats == {}
    _run_forward(model)
    # Still empty — not finalised yet
    assert handle.stats == {}
    remove_hooks(handle)


# ---------------------------------------------------------------------------
# Stats are populated correctly after forward passes
# ---------------------------------------------------------------------------


def test_all_three_sites_present_after_removal() -> None:
    """After remove_hooks, stats keys must cover pre_gelu, post_layernorm, and residual_stream."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    sites_found = {key.split("/")[-1] for key in handle.stats}
    assert SITE_PRE_GELU in sites_found
    assert SITE_POST_LAYERNORM in sites_found
    assert SITE_RESIDUAL_STREAM in sites_found


def test_stats_keys_use_layer_slash_site_format() -> None:
    """Stats dict keys must follow the '{layer_name}/{site}' format."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key in handle.stats:
        assert "/" in key, f"Key '{key}' does not contain '/'"
        layer_name, site = key.rsplit("/", 1)
        assert site in (SITE_PRE_GELU, SITE_POST_LAYERNORM, SITE_RESIDUAL_STREAM)
        assert layer_name != ""


def test_stats_fields_have_correct_types() -> None:
    """Each LayerStats instance must have correctly typed fields."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        assert isinstance(stats.site, str), key
        assert isinstance(stats.layer_name, str), key
        assert isinstance(stats.max, float), key
        assert isinstance(stats.min, float), key
        assert isinstance(stats.mean, float), key
        assert isinstance(stats.std, float), key
        assert isinstance(stats.kurtosis, float), key
        assert isinstance(stats.outlier_frac, dict), key
        assert set(stats.outlier_frac.keys()) == {"3", "4", "6"}, key
        assert isinstance(stats.n_samples, int), key
        assert stats.n_samples > 0, key


def test_scalar_stats_are_finite() -> None:
    """max, min, mean, std, kurtosis must all be finite floats."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        assert math.isfinite(stats.max), f"max not finite for {key}"
        assert math.isfinite(stats.min), f"min not finite for {key}"
        assert math.isfinite(stats.mean), f"mean not finite for {key}"
        assert math.isfinite(stats.std), f"std not finite for {key}"
        assert math.isfinite(stats.kurtosis), f"kurtosis not finite for {key}"


def test_max_geq_min() -> None:
    """max must be >= min for all stats entries."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        assert stats.max >= stats.min, f"max < min for {key}"


def test_std_nonnegative() -> None:
    """std must be non-negative for all stats entries."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        assert stats.std >= 0.0, f"negative std for {key}"


def test_outlier_fracs_between_zero_and_one() -> None:
    """All outlier fractions must lie in [0, 1]."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        for sigma_key, frac in stats.outlier_frac.items():
            assert 0.0 <= frac <= 1.0, (
                f"outlier_frac[{sigma_key!r}]={frac} out of [0,1] for {key}"
            )


def test_outlier_fracs_decrease_with_sigma() -> None:
    """Fraction of elements beyond k·σ must decrease as k increases."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        f3 = stats.outlier_frac["3"]
        f4 = stats.outlier_frac["4"]
        f6 = stats.outlier_frac["6"]
        assert f3 >= f4 >= f6, (
            f"Outlier fracs not monotone decreasing for {key}: "
            f"3σ={f3}, 4σ={f4}, 6σ={f6}"
        )


# ---------------------------------------------------------------------------
# Per-channel std
# ---------------------------------------------------------------------------


def test_per_channel_std_present_for_gelu_and_layernorm() -> None:
    """pre_gelu and post_layernorm sites must have non-None per_channel_std."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        if stats.site in (SITE_PRE_GELU, SITE_POST_LAYERNORM):
            assert stats.per_channel_std is not None, (
                f"per_channel_std should not be None for site={stats.site} ({key})"
            )
            assert len(stats.per_channel_std) > 0


def test_per_channel_std_none_for_residual_stream() -> None:
    """residual_stream site must have per_channel_std=None."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        if stats.site == SITE_RESIDUAL_STREAM:
            assert stats.per_channel_std is None, (
                f"per_channel_std should be None for residual_stream ({key})"
            )


def test_per_channel_std_all_nonnegative() -> None:
    """Every element of per_channel_std must be non-negative."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    for key, stats in handle.stats.items():
        if stats.per_channel_std is not None:
            for i, v in enumerate(stats.per_channel_std):
                assert v >= 0.0, f"per_channel_std[{i}]={v} < 0 for {key}"


# ---------------------------------------------------------------------------
# Hook removal and idempotency
# ---------------------------------------------------------------------------


def test_remove_hooks_clears_handles() -> None:
    """After remove_hooks, handle.handles must be empty."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)
    assert handle.handles == []


def test_remove_hooks_idempotent() -> None:
    """Calling remove_hooks twice must not raise."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)
    remove_hooks(handle)  # second call must be a no-op


def test_hooks_do_not_affect_output() -> None:
    """Hooks must not modify the model's output tensor."""
    model = _TinyViT()
    generator = torch.Generator()
    generator.manual_seed(7)

    x = torch.randn(2, 8, 16, generator=generator)

    # Baseline without hooks
    expected = model(x.clone()).detach().clone()

    handle = register_profiling_hooks(model)
    actual = model(x.clone()).detach()
    remove_hooks(handle)

    assert torch.allclose(expected, actual), "Hook registration altered model output"


# ---------------------------------------------------------------------------
# Accumulator unit tests
# ---------------------------------------------------------------------------


def test_accumulator_mean_matches_torch() -> None:
    """After accumulating a known tensor, welford_mean must match torch.mean."""
    acc = _SiteAccumulator(site="pre_gelu", layer_name="test")
    generator = torch.Generator()
    generator.manual_seed(0)
    t = torch.randn(100, generator=generator)
    _update_accumulator(acc, t)
    assert abs(acc.welford_mean - t.mean().item()) < 1e-4


def test_accumulator_std_matches_torch() -> None:
    """After accumulating a known tensor, finalized std must match torch.std."""
    acc = _SiteAccumulator(site="pre_gelu", layer_name="test")
    generator = torch.Generator()
    generator.manual_seed(1)
    t = torch.randn(1000, generator=generator)
    _update_accumulator(acc, t)
    stats = _finalize_accumulator(acc)
    expected_std = t.std(unbiased=False).item()
    assert abs(stats.std - expected_std) < 1e-3, (
        f"std mismatch: got {stats.std}, expected {expected_std}"
    )


def test_accumulator_multi_batch_mean_matches_full() -> None:
    """Accumulating in two batches must give the same mean as a single full batch."""
    generator = torch.Generator()
    generator.manual_seed(2)
    full = torch.randn(200, generator=generator)
    batch_a = full[:100]
    batch_b = full[100:]

    acc_single = _SiteAccumulator(site="pre_gelu", layer_name="test")
    _update_accumulator(acc_single, full)

    acc_multi = _SiteAccumulator(site="pre_gelu", layer_name="test")
    _update_accumulator(acc_multi, batch_a)
    _update_accumulator(acc_multi, batch_b)

    assert abs(acc_single.welford_mean - acc_multi.welford_mean) < 1e-5


def test_accumulator_finalize_raises_on_zero_elements() -> None:
    """_finalize_accumulator must raise RuntimeError for an empty accumulator."""
    acc = _SiteAccumulator(site="pre_gelu", layer_name="empty_layer")
    with pytest.raises(RuntimeError, match="zero elements"):
        _finalize_accumulator(acc)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save_stats → load_stats must reproduce identical LayerStats values."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    out_path = tmp_path / "stats.json"
    save_stats(handle.stats, out_path)
    assert out_path.exists()

    loaded = load_stats(out_path)
    assert set(loaded.keys()) == set(handle.stats.keys())

    for key in handle.stats:
        orig = handle.stats[key]
        reco = loaded[key]
        assert orig.site == reco.site
        assert orig.layer_name == reco.layer_name
        assert abs(orig.mean - reco.mean) < 1e-6
        assert abs(orig.std - reco.std) < 1e-6
        assert abs(orig.kurtosis - reco.kurtosis) < 1e-6
        assert orig.n_samples == reco.n_samples
        assert orig.outlier_frac == reco.outlier_frac


def test_load_stats_raises_on_missing_file(tmp_path: Path) -> None:
    """load_stats must raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_stats(tmp_path / "nonexistent.json")


def test_save_stats_creates_parent_dirs(tmp_path: Path) -> None:
    """save_stats must create nested parent directories if they do not exist."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    deep_path = tmp_path / "a" / "b" / "c" / "stats.json"
    save_stats(handle.stats, deep_path)
    assert deep_path.exists()


def test_saved_json_is_valid(tmp_path: Path) -> None:
    """The JSON written by save_stats must be parseable and contain expected keys."""
    model = _TinyViT()
    handle = register_profiling_hooks(model)
    _run_forward(model)
    remove_hooks(handle)

    out_path = tmp_path / "stats.json"
    save_stats(handle.stats, out_path)

    with out_path.open() as f:
        raw = json.load(f)

    assert isinstance(raw, dict)
    for key, val in raw.items():
        assert "site" in val, f"Missing 'site' in JSON entry for {key}"
        assert "layer_name" in val
        assert "max" in val
        assert "min" in val
        assert "mean" in val
        assert "std" in val
        assert "kurtosis" in val
        assert "outlier_frac" in val
        assert "n_samples" in val
