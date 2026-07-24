"""Tests for the nnsight-based activation profiler in :mod:`src.profiler`.

Test organisation
-----------------
Fast tests (no PyTorch forward pass, no nnsight trace)
    Run on every platform including macOS with PyTorch 2.2:

    - LayerStats dataclass construction and defaults
    - ProfilingResult serialisation round-trip (uses hand-crafted data)
    - Input-validation guards (ValueError / ProfilingError without a trace)

Slow tests  ``@pytest.mark.slow``
    Require a forward pass through a timm ViT or nnsight trace context.
    These pass reliably on Linux.  On macOS with PyTorch 2.2.x they abort
    due to a C-level signal-handler conflict between nnsight and pytest
    (observed from ``layer_norm`` / ``conv2d`` C++ kernels).

    Run with::

        pytest -m slow tests/test_profiler.py

    Skip with::

        pytest -m "not slow" tests/test_profiler.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.exceptions import ProfilingError
from src.profiler import (
    OUTLIER_SIGMAS,
    SITE_POST_LAYERNORM_1,
    SITE_POST_LAYERNORM_2,
    SITE_POST_SOFTMAX,
    SITE_PRE_GELU,
    SITE_PRE_SOFTMAX,
    SITE_RESIDUAL_STREAM,
    LayerStats,
    ProfilingResult,
    _finalize_stats,
    _register_stat_saves,
    load_profiling_result,
    profile_vit,
    save_profiling_result,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 12
_SITES_PER_BLOCK = 6


# ---------------------------------------------------------------------------
# Helpers — hand-crafted data, no ViT or nnsight trace required
# ---------------------------------------------------------------------------


def _canned_result() -> ProfilingResult:
    """Build a ProfilingResult from hand-crafted LayerStats (no trace needed)."""
    stats: dict[str, LayerStats] = {}
    for i in range(_NUM_BLOCKS):
        for site in (
            SITE_POST_LAYERNORM_1,
            SITE_POST_LAYERNORM_2,
            SITE_PRE_GELU,
            SITE_PRE_SOFTMAX,
            SITE_POST_SOFTMAX,
        ):
            key = f"blocks.{i}/{site}"
            stats[key] = LayerStats(
                    site_identifier=key,
                    mean=float(i) * 0.01,
                    std=1.0 + float(i) * 0.05,
                    kurtosis=0.5,
                    m3=0.0,
                    outlier_fractions={f"{s}_sigma": 0.001 for s in OUTLIER_SIGMAS},
                    n_samples=0,
                )
            res_key = (
                "patch_embed/residual_stream"
                if i == 0
                else f"blocks.{i - 1}/residual_stream"
            )
            stats[res_key] = LayerStats(
                site_identifier=res_key,
                mean=0.0,
                std=0.5,
                kurtosis=0.1,
                m3=0.0,
                outlier_fractions={f"{s}_sigma": 0.002 for s in OUTLIER_SIGMAS},
                n_samples=0,
            )
    return ProfilingResult(
        stats=stats,
        num_blocks=_NUM_BLOCKS,
        batch_shape=(1, 3, 224, 224),
    )


# ---------------------------------------------------------------------------
# LayerStats dataclass — no trace needed
# ---------------------------------------------------------------------------


def test_layer_stats_construction() -> None:
    """LayerStats must be constructible with the required fields."""
    stats = LayerStats(
        site_identifier="blocks.0/pre_gelu",
        mean=0.1,
        std=1.5,
        kurtosis=2.3,
        outlier_fractions={"3.0_sigma": 0.003, "5.0_sigma": 0.0, "8.0_sigma": 0.0},
    )
    assert stats.site_identifier == "blocks.0/pre_gelu"
    assert stats.mean == pytest.approx(0.1)
    assert stats.kurtosis == pytest.approx(2.3)


def test_layer_stats_default_outlier_fractions() -> None:
    """outlier_fractions defaults to an empty dict when not supplied."""
    stats = LayerStats(site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0)
    assert stats.outlier_fractions == {}


def test_layer_stats_invariant_outlier_keys() -> None:
    """A LayerStats built with correct keys must have exactly the expected keys."""
    fracs = {f"{s}_sigma": 0.0 for s in OUTLIER_SIGMAS}
    stats = LayerStats(
        site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0,
        outlier_fractions=fracs,
    )
    assert set(stats.outlier_fractions.keys()) == {f"{s}_sigma" for s in OUTLIER_SIGMAS}


# ---------------------------------------------------------------------------
# Input validation — fires before any nnsight trace
# ---------------------------------------------------------------------------


def test_profile_vit_raises_on_non_4d_input() -> None:
    """profile_vit must raise ValueError for any non-4-D input tensor."""
    from nnsight import NNsight
    # The shape check runs before the trace context opens, so no kernel dispatch.
    wrapped = NNsight(nn.Linear(8, 4))
    with pytest.raises(ValueError, match="4-D"):
        profile_vit(wrapped, torch.randn(3, 224, 224))


def test_profile_vit_raises_on_model_without_blocks() -> None:
    """profile_vit must raise ProfilingError for a model without .blocks."""
    from nnsight import NNsight
    # Pass a 4-D tensor so the shape guard doesn't fire first.
    # The .blocks check runs before the trace context opens too.
    wrapped = NNsight(nn.Sequential(nn.Linear(8, 4)))
    with pytest.raises((ProfilingError, ValueError)):
        profile_vit(wrapped, torch.randn(1, 8, 1, 1))


# ---------------------------------------------------------------------------
# ProfilingResult serialisation — uses hand-crafted data, no trace
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save_profiling_result / load_profiling_result must be lossless."""
    result = _canned_result()
    out_path = tmp_path / "result.json"
    save_profiling_result(result, out_path)
    assert out_path.exists()

    loaded = load_profiling_result(out_path)
    assert loaded.num_blocks == result.num_blocks
    assert loaded.batch_shape == result.batch_shape
    assert set(loaded.stats.keys()) == set(result.stats.keys())

    sample_key = "blocks.3/pre_gelu"
    orig = result.stats[sample_key]
    reco = loaded.stats[sample_key]
    assert reco.mean == pytest.approx(orig.mean)
    assert reco.std == pytest.approx(orig.std)
    assert reco.kurtosis == pytest.approx(orig.kurtosis)
    assert reco.outlier_fractions == orig.outlier_fractions


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """save_profiling_result must create deeply nested parent directories."""
    deep = tmp_path / "a" / "b" / "c" / "result.json"
    save_profiling_result(_canned_result(), deep)
    assert deep.exists()


def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    """load_profiling_result must raise FileNotFoundError for a missing path."""
    with pytest.raises(FileNotFoundError):
        load_profiling_result(tmp_path / "nonexistent.json")


def test_saved_json_schema(tmp_path: Path) -> None:
    """JSON written by save_profiling_result must have the expected schema."""
    out_path = tmp_path / "result.json"
    save_profiling_result(_canned_result(), out_path)

    with out_path.open() as f:
        raw = json.load(f)

    assert "stats" in raw
    assert "num_blocks" in raw
    assert "batch_shape" in raw
    assert raw["num_blocks"] == _NUM_BLOCKS
    assert raw["batch_shape"] == [1, 3, 224, 224]

    sample = next(iter(raw["stats"].values()))
    for field in ("site_identifier", "mean", "std", "kurtosis", "outlier_fractions"):
        assert field in sample, f"Missing JSON field: {field!r}"


def test_site_identifier_survives_serialisation(tmp_path: Path) -> None:
    """site_identifier must match its dict key after save → load."""
    path = tmp_path / "r.json"
    save_profiling_result(_canned_result(), path)
    loaded = load_profiling_result(path)
    for key, stats in loaded.stats.items():
        assert stats.site_identifier == key


def test_profiling_result_canned_site_keys() -> None:
    """Hand-crafted result must contain all expected site keys for every block."""
    result = _canned_result()
    keys = set(result.stats.keys())
    assert "patch_embed/residual_stream" in keys
    for i in range(_NUM_BLOCKS):
        assert f"blocks.{i}/{SITE_POST_LAYERNORM_1}" in keys
        assert f"blocks.{i}/{SITE_POST_LAYERNORM_2}" in keys
        assert f"blocks.{i}/{SITE_PRE_GELU}" in keys
        assert f"blocks.{i}/{SITE_PRE_SOFTMAX}" in keys
        assert f"blocks.{i}/{SITE_POST_SOFTMAX}" in keys
    for i in range(1, _NUM_BLOCKS):
        assert f"blocks.{i - 1}/residual_stream" in keys


# ---------------------------------------------------------------------------
# Slow tests — require nnsight trace context (fail on macOS + PyTorch 2.2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _vit_wrapped():
    """NNsight-wrapped vit_base_patch16_224 with random weights; module-scoped."""
    import timm
    from nnsight import NNsight

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    return NNsight(model)


@pytest.fixture(scope="module")
def _vit_result(_vit_wrapped) -> ProfilingResult:
    """Full profiling result from a single forward pass; module-scoped."""
    x = torch.randn(1, 3, 224, 224)
    return profile_vit(_vit_wrapped, x)


@pytest.mark.slow
def test_slow_num_blocks(_vit_result: ProfilingResult) -> None:
    assert _vit_result.num_blocks == _NUM_BLOCKS


@pytest.mark.slow
def test_slow_total_site_count(_vit_result: ProfilingResult) -> None:
    assert len(_vit_result.stats) == _NUM_BLOCKS * _SITES_PER_BLOCK


@pytest.mark.slow
def test_slow_all_expected_sites_present(_vit_result: ProfilingResult) -> None:
    keys = set(_vit_result.stats.keys())
    assert "patch_embed/residual_stream" in keys
    for i in range(_NUM_BLOCKS):
        assert f"blocks.{i}/{SITE_POST_LAYERNORM_1}" in keys
        assert f"blocks.{i}/{SITE_POST_LAYERNORM_2}" in keys
        assert f"blocks.{i}/{SITE_PRE_GELU}" in keys
        assert f"blocks.{i}/{SITE_PRE_SOFTMAX}" in keys
        assert f"blocks.{i}/{SITE_POST_SOFTMAX}" in keys
    for i in range(1, _NUM_BLOCKS):
        assert f"blocks.{i - 1}/residual_stream" in keys


@pytest.mark.slow
def test_slow_all_stats_finite(_vit_result: ProfilingResult) -> None:
    for key, stats in _vit_result.stats.items():
        assert math.isfinite(stats.mean), f"non-finite mean at {key}"
        assert math.isfinite(stats.std), f"non-finite std at {key}"
        assert math.isfinite(stats.kurtosis), f"non-finite kurtosis at {key}"


@pytest.mark.slow
def test_slow_std_nonnegative(_vit_result: ProfilingResult) -> None:
    for key, stats in _vit_result.stats.items():
        assert stats.std >= 0.0, f"negative std at {key}"


@pytest.mark.slow
def test_slow_outlier_fracs_monotone(_vit_result: ProfilingResult) -> None:
    for key, stats in _vit_result.stats.items():
        fracs = [stats.outlier_fractions[f"{s}_sigma"] for s in OUTLIER_SIGMAS]
        for lo, hi in zip(fracs, fracs[1:]):
            assert lo >= hi - 1e-8, f"Non-monotone fracs at {key}: {fracs}"


@pytest.mark.slow
def test_slow_outlier_fracs_in_unit_interval(_vit_result: ProfilingResult) -> None:
    for key, stats in _vit_result.stats.items():
        for sigma_key, frac in stats.outlier_fractions.items():
            assert 0.0 <= frac <= 1.0, f"outlier_fractions[{sigma_key!r}]={frac} at {key}"


@pytest.mark.slow
def test_slow_post_softmax_mean(_vit_result: ProfilingResult) -> None:
    """Post-softmax mean ≈ 1/N_tokens = 1/197 for ViT-B/16."""
    expected = 1.0 / 197
    for i in range(_NUM_BLOCKS):
        key = f"blocks.{i}/{SITE_POST_SOFTMAX}"
        actual = _vit_result.stats[key].mean
        assert abs(actual - expected) < 1e-3, (
            f"Post-softmax mean={actual:.6f} at {key}, expected ~{expected:.6f}"
        )


@pytest.mark.slow
def test_slow_post_layernorm_std_near_one(_vit_result: ProfilingResult) -> None:
    for i in range(_NUM_BLOCKS):
        for site in (SITE_POST_LAYERNORM_1, SITE_POST_LAYERNORM_2):
            key = f"blocks.{i}/{site}"
            std = _vit_result.stats[key].std
            assert 0.5 <= std <= 2.0, f"Post-LN std={std:.4f} at {key}"


@pytest.mark.slow
def test_slow_site_identifier_matches_key(_vit_result: ProfilingResult) -> None:
    for key, stats in _vit_result.stats.items():
        assert stats.site_identifier == key


@pytest.mark.slow
def test_slow_round_trip(tmp_path: Path, _vit_result: ProfilingResult) -> None:
    """save → load must be lossless on a real profiling result."""
    path = tmp_path / "slow_result.json"
    save_profiling_result(_vit_result, path)
    loaded = load_profiling_result(path)
    assert loaded.num_blocks == _vit_result.num_blocks
    sample_key = "blocks.0/pre_gelu"
    assert loaded.stats[sample_key].mean == pytest.approx(
        _vit_result.stats[sample_key].mean, abs=1e-6
    )


@pytest.mark.slow
def test_slow_register_saves_finalize_layernorm() -> None:
    """_register_stat_saves + _finalize_stats on a LayerNorm give correct stats."""
    from nnsight import NNsight

    torch.manual_seed(0)
    wrapped = NNsight(nn.LayerNorm(16))
    x = torch.randn(4, 32, 16)
    n_samples = 4 * 32 * 16  # B * seq * D
    with wrapped.trace(x):
        savers = _register_stat_saves(wrapped.output, "test/ln", n_samples)
    stats = _finalize_stats(savers)

    assert stats.n_samples == n_samples
    assert math.isfinite(stats.mean)
    assert abs(stats.mean) < 0.05, f"LN mean={stats.mean:.6f} expected ~0"
    # Population std of a LayerNorm output should be close to 1.
    assert 0.8 <= stats.std <= 1.2, f"LN std={stats.std:.4f} expected ~1"
    assert stats.std >= 0.0
    expected_keys = {f"{s}_sigma" for s in OUTLIER_SIGMAS}
    assert set(stats.outlier_fractions.keys()) == expected_keys
    for k, v in stats.outlier_fractions.items():
        assert 0.0 <= v <= 1.0, f"outlier_fractions[{k!r}]={v} out of [0,1]"


@pytest.mark.slow
def test_slow_kurtosis_gaussian() -> None:
    """Excess kurtosis of a large Gaussian sample should be close to 0."""
    from nnsight import NNsight

    torch.manual_seed(42)
    wrapped = NNsight(nn.Identity())
    t = torch.randn(1, 1, 10000)
    n_samples = 1 * 1 * 10000
    with wrapped.trace(t):
        savers = _register_stat_saves(wrapped.output, "test/gauss", n_samples)
    stats = _finalize_stats(savers)
    assert abs(stats.kurtosis) < 0.5, f"kurtosis={stats.kurtosis:.4f} expected ~0"
    assert stats.n_samples == n_samples
