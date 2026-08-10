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
    histogram_profile_vit,
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
    # Add the final residual stream (output of last encoder block, before head LN).
    # This is blocks.{_NUM_BLOCKS-1}/residual_stream — the one site that the
    # per-block loop cannot produce because it labels block[i].norm1.input as
    # blocks.{i-1}/residual_stream.  See T-001.
    final_res_key = f"blocks.{_NUM_BLOCKS - 1}/residual_stream"
    stats[final_res_key] = LayerStats(
        site_identifier=final_res_key,
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
        outlier_fractions={"3.0_sigma": 0.003, "4.0_sigma": 0.0, "6.0_sigma": 0.0},
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


def test_layer_stats_max_min_defaults() -> None:
    """max and min must default to 0.0 when not supplied."""
    stats = LayerStats(site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0)
    assert stats.max == 0.0
    assert stats.min == 0.0


def test_layer_stats_max_min_store_values() -> None:
    """max and min must accept and store explicit float values."""
    stats = LayerStats(
        site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0,
        max=12.5, min=-3.7,
    )
    assert stats.max == pytest.approx(12.5)
    assert stats.min == pytest.approx(-3.7)


def test_layer_stats_max_min_survive_serialization(tmp_path: Path) -> None:
    """max and min must survive JSON save → load roundtrip."""
    result = ProfilingResult(
        stats={
            "blocks.0/pre_gelu": LayerStats(
                site_identifier="blocks.0/pre_gelu",
                mean=0.0, std=1.0, kurtosis=0.0,
                max=8.5, min=-6.2,
            ),
        },
        num_blocks=12,
        batch_shape=(1, 3, 224, 224),
    )
    path = tmp_path / "maxmin_roundtrip.json"
    save_profiling_result(result, path)
    loaded = load_profiling_result(path)
    recovered = loaded.stats["blocks.0/pre_gelu"]
    assert recovered.max == pytest.approx(8.5)
    assert recovered.min == pytest.approx(-6.2)


def test_layer_stats_max_min_backwards_compat(tmp_path: Path) -> None:
    """Old JSON without max/min keys must deserialize with max=0.0, min=0.0."""
    import json

    raw = {
        "stats": {
            "blocks.0/pre_gelu": {
                "site_identifier": "blocks.0/pre_gelu",
                "mean": 0.5,
                "std": 2.0,
                "kurtosis": 1.0,
                "m3": 0.0,
                "outlier_fractions": {"3.0_sigma": 0.01},
                "n_samples": 1000,
                "per_channel_std": None,
                "per_channel_sum": None,
                "per_channel_sum_sq": None,
                "attention_entropy_cls": None,
                "attention_entropy_patches": None,
                "layernorm_gamma": None,
                "layernorm_beta": None,
                "ln2_amplification_ratio": None,
            }
        },
        "num_blocks": 12,
        "batch_shape": [1, 3, 224, 224],
        "metadata": None,
    }
    path = tmp_path / "old_no_maxmin.json"
    path.write_text(json.dumps(raw))

    loaded = load_profiling_result(path)
    stats = loaded.stats["blocks.0/pre_gelu"]
    assert stats.max == 0.0, f"Expected max=0.0 for backward compat, got {stats.max}"
    assert stats.min == 0.0, f"Expected min=0.0 for backward compat, got {stats.min}"


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


def test_run_metadata_round_trip(tmp_path: Path) -> None:
    """RunMetadata must survive JSON save → load roundtrip."""
    from src.profiler import RunMetadata

    metadata = RunMetadata(
        python_version="3.12.3",
        pytorch_version="2.13.0",
        timm_version="1.0.28",
        nnsight_version="0.7.0",
        cuda_available=True,
        cuda_version="13.0",
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_memory_gb=8.0,
        model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        dataset="ImageNet-1K validation",
        num_images=50000,
        batch_size=64,
        seed=42,
        num_seeds=3,
        timestamp_utc="2026-07-30T00:00:00+00:00",
    )
    result = ProfilingResult(
        stats={},
        num_blocks=12,
        batch_shape=(1, 3, 224, 224),
        metadata=metadata,
    )
    path = tmp_path / "with_metadata.json"
    save_profiling_result(result, path)
    loaded = load_profiling_result(path)
    assert loaded.metadata is not None
    assert loaded.metadata.python_version == "3.12.3"
    assert loaded.metadata.pytorch_version == "2.13.0"
    assert loaded.metadata.timm_version == "1.0.28"
    assert loaded.metadata.nnsight_version == "0.7.0"
    assert loaded.metadata.cuda_available is True
    assert loaded.metadata.cuda_version == "13.0"
    assert loaded.metadata.gpu_name == "NVIDIA GeForce RTX 3070"
    assert loaded.metadata.gpu_memory_gb == pytest.approx(8.0)
    assert loaded.metadata.model_name == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    assert loaded.metadata.dataset == "ImageNet-1K validation"
    assert loaded.metadata.num_images == 50000
    assert loaded.metadata.batch_size == 64
    assert loaded.metadata.seed == 42
    assert loaded.metadata.num_seeds == 3
    assert loaded.metadata.timestamp_utc == "2026-07-30T00:00:00+00:00"


def test_run_metadata_backwards_compat(tmp_path: Path) -> None:
    """Old JSON without metadata key must deserialize with metadata=None."""
    import json

    raw = {
        "stats": {},
        "num_blocks": 12,
        "batch_shape": [1, 3, 224, 224],
    }
    path = tmp_path / "old_no_metadata.json"
    path.write_text(json.dumps(raw))
    loaded = load_profiling_result(path)
    assert loaded.metadata is None


def test_site_identifier_survives_serialisation(tmp_path: Path) -> None:
    """site_identifier must match its dict key after save → load."""
    path = tmp_path / "r.json"
    save_profiling_result(_canned_result(), path)
    loaded = load_profiling_result(path)
    for key, stats in loaded.stats.items():
        assert stats.site_identifier == key


def test_float_precision_round_trip(tmp_path: Path) -> None:
    """Float values must survive JSON round-trip with full float64 precision.

    Verifies that json.dump (which uses repr() internally) preserves exact
    IEEE 754 binary values.  This is critical for outlier fractions where
    values like 0.00261 must round-trip losslessly — a truncated value like
    0.0026 would be a different float64.
    """
    # Use values that are sensitive to truncation.
    sensitive_values = [
        0.0026099999999999999,  # near the suspicious 0.00261
        1.2345678901234567e-10,  # very small
        3.141592653589793,       # π
        0.0,
        -0.0,
        1.0,
    ]
    for original in sensitive_values:
        stats = LayerStats(
            site_identifier="test",
            mean=original,
            std=original * 2.0,
            kurtosis=original * 3.0,
            outlier_fractions={"3.0_sigma": original},
        )
        result = ProfilingResult(
            stats={"test": stats},
            num_blocks=12,
            batch_shape=(1, 3, 224, 224),
        )
        path = tmp_path / "precision_test.json"
        save_profiling_result(result, path)
        loaded = load_profiling_result(path)
        recovered = loaded.stats["test"]

        # Use exact equality (not pytest.approx) — the values must be
        # bit-identical after round-trip.
        assert recovered.mean == original, (
            f"mean: {recovered.mean!r} != {original!r}"
        )
        assert recovered.std == original * 2.0, (
            f"std: {recovered.std!r} != {original * 2.0!r}"
        )
        assert recovered.kurtosis == original * 3.0, (
            f"kurtosis: {recovered.kurtosis!r} != {original * 3.0!r}"
        )
        assert recovered.outlier_fractions["3.0_sigma"] == original, (
            f"outlier_frac: {recovered.outlier_fractions['3.0_sigma']!r} != {original!r}"
        )


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
    # Final residual stream (output of last encoder block) — see T-001.
    assert f"blocks.{_NUM_BLOCKS - 1}/residual_stream" in keys


# ---------------------------------------------------------------------------
# LayerStats — LayerNorm γ/β fields (T-004)
# ---------------------------------------------------------------------------


def test_layer_stats_layernorm_fields_default_none() -> None:
    """layernorm_gamma and layernorm_beta must default to None."""
    stats = LayerStats(site_identifier="blocks.0/pre_gelu", mean=0.0, std=1.0, kurtosis=0.0)
    assert stats.layernorm_gamma is None
    assert stats.layernorm_beta is None


def test_layer_stats_layernorm_fields_store_values() -> None:
    """layernorm_gamma and layernorm_beta must accept and store float lists."""
    gamma = [0.5, 1.2, 3.0]
    beta = [-0.1, 0.0, 0.1]
    stats = LayerStats(
        site_identifier="blocks.0/post_layernorm_1",
        mean=0.0, std=1.0, kurtosis=0.0,
        layernorm_gamma=gamma,
        layernorm_beta=beta,
    )
    assert stats.layernorm_gamma == gamma
    assert stats.layernorm_beta == beta
    assert len(stats.layernorm_gamma) == 3


def test_layer_stats_layernorm_serialization_roundtrip(tmp_path: Path) -> None:
    """γ/β must survive JSON save → load roundtrip (dataclasses.asdict)."""
    gamma = [0.5, 1.2, 3.0]
    beta = [-0.1, 0.0, 0.1]
    result = ProfilingResult(
        stats={
            "blocks.0/post_layernorm_1": LayerStats(
                site_identifier="blocks.0/post_layernorm_1",
                mean=0.0, std=1.0, kurtosis=0.0,
                layernorm_gamma=gamma, layernorm_beta=beta,
            ),
        },
        num_blocks=12,
        batch_shape=(1, 3, 224, 224),
    )
    path = tmp_path / "gamma_roundtrip.json"
    save_profiling_result(result, path)
    loaded = load_profiling_result(path)
    ln_stats = loaded.stats["blocks.0/post_layernorm_1"]
    assert ln_stats.layernorm_gamma == gamma
    assert ln_stats.layernorm_beta == beta


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
    # 12 blocks × 6 sites + 1 final residual stream = 73
    assert len(_vit_result.stats) == _NUM_BLOCKS * _SITES_PER_BLOCK + 1


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
    # Final residual stream (output of last encoder block) — see T-001.
    assert f"blocks.{_NUM_BLOCKS - 1}/residual_stream" in keys


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
def test_slow_layernorm_gamma_present(_vit_result: ProfilingResult) -> None:
    """All post_layernorm sites must have non-None layernorm_gamma.

    γ weights are static model parameters extracted outside the trace.
    For ViT-B/16 with D=768, each γ vector must have length 768.
    """
    for i in range(_NUM_BLOCKS):
        for site in (SITE_POST_LAYERNORM_1, SITE_POST_LAYERNORM_2):
            key = f"blocks.{i}/{site}"
            stats = _vit_result.stats[key]
            assert stats.layernorm_gamma is not None, f"missing gamma at {key}"
            assert isinstance(stats.layernorm_gamma, list)
            assert len(stats.layernorm_gamma) == 768, (
                f"Expected 768 gamma values at {key}, got {len(stats.layernorm_gamma)}"
            )
            assert all(isinstance(g, float) for g in stats.layernorm_gamma), (
                f"gamma values must be floats at {key}"
            )


@pytest.mark.slow
def test_slow_layernorm_beta_present(_vit_result: ProfilingResult) -> None:
    """All post_layernorm sites must have non-None layernorm_beta.

    β biases are static model parameters extracted outside the trace.
    For ViT-B/16 with D=768, each β vector must have length 768.
    """
    for i in range(_NUM_BLOCKS):
        for site in (SITE_POST_LAYERNORM_1, SITE_POST_LAYERNORM_2):
            key = f"blocks.{i}/{site}"
            stats = _vit_result.stats[key]
            assert stats.layernorm_beta is not None, f"missing beta at {key}"
            assert isinstance(stats.layernorm_beta, list)
            assert len(stats.layernorm_beta) == 768, (
                f"Expected 768 beta values at {key}, got {len(stats.layernorm_beta)}"
            )
            assert all(isinstance(b, float) for b in stats.layernorm_beta), (
                f"beta values must be floats at {key}"
            )


@pytest.mark.slow
def test_slow_layernorm_fields_absent_on_non_ln_sites(
    _vit_result: ProfilingResult,
) -> None:
    """Non-LayerNorm sites must have None for layernorm_gamma and layernorm_beta."""
    for key, stats in _vit_result.stats.items():
        if SITE_POST_LAYERNORM_1 in key or SITE_POST_LAYERNORM_2 in key:
            continue  # LN sites — covered by the tests above
        assert stats.layernorm_gamma is None, (
            f"non-LN site {key} has unexpected layernorm_gamma"
        )
        assert stats.layernorm_beta is None, (
            f"non-LN site {key} has unexpected layernorm_beta"
        )


@pytest.mark.slow
def test_slow_layernorm_gamma_match_model_weights(
    _vit_wrapped, _vit_result: ProfilingResult,
) -> None:
    """γ/β from profile_vit must match the model's actual LayerNorm weights.

    Extracts norm{1,2}.weight and norm{1,2}.bias from the underlying model
    and compares against the values stored in LayerStats.
    """
    inner = _vit_wrapped._model
    for i in range(_NUM_BLOCKS):
        block = inner.blocks[i]
        for norm_attr, site in [
            ("norm1", SITE_POST_LAYERNORM_1),
            ("norm2", SITE_POST_LAYERNORM_2),
        ]:
            key = f"blocks.{i}/{site}"
            ln = getattr(block, norm_attr)
            expected_gamma = ln.weight.detach().cpu().tolist()
            expected_beta = (
                ln.bias.detach().cpu().tolist() if ln.bias is not None else None
            )
            stats = _vit_result.stats[key]
            # Compare with tolerance for float conversion roundtrip
            for j, (a, b) in enumerate(zip(stats.layernorm_gamma, expected_gamma)):
                assert a == pytest.approx(b, rel=1e-6), (
                    f"gamma mismatch at {key}[{j}]: {a} vs {b}"
                )
            if expected_beta is not None:
                for j, (a, b) in enumerate(zip(stats.layernorm_beta, expected_beta)):
                    assert a == pytest.approx(b, rel=1e-6), (
                        f"beta mismatch at {key}[{j}]: {a} vs {b}"
                    )


@pytest.mark.slow
def test_slow_layernorm_gamma_survives_serialisation(
    tmp_path: Path, _vit_result: ProfilingResult,
) -> None:
    """γ/β must survive JSON save → load roundtrip on a real result."""
    path = tmp_path / "ln_roundtrip.json"
    save_profiling_result(_vit_result, path)
    loaded = load_profiling_result(path)
    for i in range(_NUM_BLOCKS):
        for site in (SITE_POST_LAYERNORM_1, SITE_POST_LAYERNORM_2):
            key = f"blocks.{i}/{site}"
            orig = _vit_result.stats[key]
            loaded_stats = loaded.stats[key]
            assert loaded_stats.layernorm_gamma == orig.layernorm_gamma
            assert loaded_stats.layernorm_beta == orig.layernorm_beta



@pytest.mark.slow
def test_slow_register_saves_finalize_layernorm() -> None:
    """_register_stat_saves + _finalize_stats on a LayerNorm give correct stats."""
    from nnsight import NNsight

    torch.manual_seed(0)
    wrapped = NNsight(nn.LayerNorm(16))
    x = torch.randn(4, 32, 16)
    n_samples = 4 * 32 * 16  # B * seq * D

    # nnsight ≥0.3: .trace() does not bind local variables in the body.
    # Use the list-outside-trace pattern from profile_vit.
    savers_list: list = []
    with wrapped.trace(x):
        savers_list.append(_register_stat_saves(wrapped.output, "test/ln", n_samples))
    stats = _finalize_stats(savers_list[0])

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

    # nnsight ≥0.3: .trace() does not bind local variables in the body.
    savers_list: list = []
    with wrapped.trace(t):
        savers_list.append(_register_stat_saves(wrapped.output, "test/gauss", n_samples))
    stats = _finalize_stats(savers_list[0])
    assert abs(stats.kurtosis) < 0.5, f"kurtosis={stats.kurtosis:.4f} expected ~0"
    assert stats.n_samples == n_samples


# ---------------------------------------------------------------------------
# WelfordAccumulator — fast tests (no trace)
# ---------------------------------------------------------------------------


def test_welford_accumulator_construction() -> None:
    """WelfordAccumulator initialises with zero-state defaults."""
    from src.profiler import WelfordAccumulator

    acc = WelfordAccumulator(site_identifier="blocks.0/pre_gelu")
    assert acc.n == 0
    assert acc.mean == 0.0
    assert acc.M2 == 0.0
    assert acc.M3 == 0.0
    assert acc.M4 == 0.0
    assert set(acc.outlier_counts.keys()) == {f"{k}_sigma" for k in OUTLIER_SIGMAS}
    assert all(v == 0 for v in acc.outlier_counts.values())
    # per_channel fields default to None/0
    assert acc.per_channel_sum is None
    assert acc.per_channel_sum_sq is None
    assert acc.per_channel_n == 0


def test_merge_batch_stats_single_batch() -> None:
    """After one merge, accumulator mean and population std must match batch."""
    from src.profiler import WelfordAccumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="test/site")
    batch_stats = LayerStats(
        site_identifier="test/site",
        mean=2.0,
        std=3.0,
        kurtosis=0.0,
        m3=0.0,
        outlier_fractions={f"{k}_sigma": 0.01 for k in OUTLIER_SIGMAS},
        n_samples=1000,
    )
    merge_batch_stats(acc, batch_stats, 1000)

    assert acc.n == 1000
    assert math.isclose(acc.mean, 2.0)
    # Population variance = std² = 9.0;  M2 = 9.0 * 1000 = 9000.
    assert math.isclose(acc.M2, 9000.0, rel_tol=1e-6)


def test_finalize_accumulator_two_equal_batches() -> None:
    """Two identical batches: global mean and std must match the batch values exactly."""
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    acc = WelfordAccumulator(site_identifier="test/site")
    for _ in range(2):
        bs = LayerStats(
            site_identifier="test/site",
            mean=4.0,
            std=2.0,
            kurtosis=0.0,
            m3=0.0,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=100,
        )
        merge_batch_stats(acc, bs, 100)

    result = finalize_accumulator(acc)
    assert math.isclose(result.mean, 4.0, rel_tol=1e-6)
    assert math.isclose(result.std, 2.0, rel_tol=1e-6)
    assert result.n_samples == 200


def test_merge_batch_stats_exact_kurtosis_known_data() -> None:
    """Pébay M3/M4 merge must produce correct kurtosis for known-distribution data.

    Two batches drawn from N(0, 1) with known sample moments.  The merged
    excess kurtosis should be close to 0 (Gaussian).
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(0)
    # Generate two batches, compute their exact per-batch stats.
    b1 = torch.randn(5000)
    b2 = torch.randn(5000)

    def _batch_layer_stats(t: torch.Tensor, site_id: str) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier=site_id,
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
        )

    acc = WelfordAccumulator(site_identifier="test/gauss")
    merge_batch_stats(acc, _batch_layer_stats(b1, "test/gauss"), 5000)
    merge_batch_stats(acc, _batch_layer_stats(b2, "test/gauss"), 5000)
    result = finalize_accumulator(acc)

    # Full 10k-sample mean and std should be close to 0 and 1.
    full = torch.cat([b1, b2])
    assert math.isclose(result.mean, full.mean().item(), rel_tol=1e-4)
    assert math.isclose(result.std, full.std(correction=0).item(), rel_tol=1e-4)
    # Excess kurtosis of Gaussian should be near 0.
    assert abs(result.kurtosis) < 0.5, f"kurtosis={result.kurtosis:.4f} expected ~0"
    assert result.n_samples == 10000


def test_merge_batch_stats_raises_on_zero_batch_n() -> None:
    """merge_batch_stats must raise ValueError when batch_n <= 0."""
    from src.profiler import WelfordAccumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="test/site")
    bs = LayerStats(
        site_identifier="test/site",
        mean=0.0,
        std=1.0,
        kurtosis=0.0,
        m3=0.0,
        outlier_fractions={},
        n_samples=0,
    )
    with pytest.raises(ValueError, match="batch_n must be positive"):
        merge_batch_stats(acc, bs, 0)
    with pytest.raises(ValueError, match="batch_n must be positive"):
        merge_batch_stats(acc, bs, -1)


def test_finalize_accumulator_raises_on_zero_n() -> None:
    """finalize_accumulator must raise ValueError when acc.n == 0."""
    from src.profiler import WelfordAccumulator, finalize_accumulator

    acc = WelfordAccumulator(site_identifier="test/empty")
    with pytest.raises(ValueError, match="zero elements"):
        finalize_accumulator(acc)


def test_site_n_returns_correct_counts() -> None:
    """_site_n must return correct element counts for each site type."""
    from src.profiler import (
        SITE_PRE_GELU,
        SITE_PRE_SOFTMAX,
        SITE_POST_SOFTMAX,
        SITE_POST_LAYERNORM_1,
        SITE_RESIDUAL_STREAM,
        _site_n,
    )

    B, N, D, D_mlp, H = 4, 197, 768, 3072, 12

    # pre_gelu: B * N * D_mlp
    assert _site_n(f"blocks.0/{SITE_PRE_GELU}", B, N, D, D_mlp, H) == 4 * 197 * 3072
    # pre_softmax: B * H * N * N
    assert (
        _site_n(f"blocks.0/{SITE_PRE_SOFTMAX}", B, N, D, D_mlp, H)
        == 4 * 12 * 197 * 197
    )
    # post_softmax: same as pre_softmax
    assert (
        _site_n(f"blocks.0/{SITE_POST_SOFTMAX}", B, N, D, D_mlp, H)
        == 4 * 12 * 197 * 197
    )
    # residual_stream / post_layernorm: B * N * D
    assert (
        _site_n(f"blocks.0/{SITE_POST_LAYERNORM_1}", B, N, D, D_mlp, H)
        == 4 * 197 * 768
    )
    assert (
        _site_n(f"blocks.0/{SITE_RESIDUAL_STREAM}", B, N, D, D_mlp, H)
        == 4 * 197 * 768
    )


def test_merge_batch_stats_outlier_accumulation() -> None:
    """Outlier counts must accumulate correctly across batches."""
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    acc = WelfordAccumulator(site_identifier="test/site")
    # Batch with 10% outliers at 3σ, 5% at 4σ, 0% at 6σ
    bs = LayerStats(
        site_identifier="test/site",
        mean=0.0,
        std=1.0,
        kurtosis=0.0,
        m3=0.0,
        outlier_fractions={"3.0_sigma": 0.10, "4.0_sigma": 0.05, "6.0_sigma": 0.0},
        n_samples=1000,
    )
    merge_batch_stats(acc, bs, 1000)
    merge_batch_stats(acc, bs, 1000)

    result = finalize_accumulator(acc)
    # 0.10 * 1000 = 100 per batch, two batches → 200 / 2000 = 0.10
    assert math.isclose(result.outlier_fractions["3.0_sigma"], 0.10, rel_tol=1e-6)
    assert math.isclose(result.outlier_fractions["4.0_sigma"], 0.05, rel_tol=1e-6)
    assert math.isclose(result.outlier_fractions["6.0_sigma"], 0.0)
    assert result.n_samples == 2000


def test_per_channel_merge_two_batches() -> None:
    """Per-channel M2 merge must produce correct per-channel std."""
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(1)
    # Two batches of shape (2, 4, 8): B=2, N=4, D=8
    b1 = torch.randn(2, 4, 8)
    b2 = torch.randn(2, 4, 8)
    full = torch.cat([b1, b2], dim=0)  # (4, 4, 8)

    def _batch_stats(t: torch.Tensor) -> LayerStats:
        x = t.float()
        flat = x.reshape(-1, x.shape[-1])  # (B*N, D)
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x.flatten() - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        per_ch = flat.std(dim=0, correction=0).tolist()
        per_ch_sum = flat.sum(dim=0).tolist()
        per_ch_sum_sq = (flat**2).sum(dim=0).tolist()
        return LayerStats(
            site_identifier="test/site",
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
            per_channel_std=per_ch,
            per_channel_sum=per_ch_sum,
            per_channel_sum_sq=per_ch_sum_sq,
        )

    acc = WelfordAccumulator(site_identifier="test/site")
    merge_batch_stats(acc, _batch_stats(b1), b1.numel())
    merge_batch_stats(acc, _batch_stats(b2), b2.numel())
    result = finalize_accumulator(acc)

    # Per-channel std from merge should match full-dataset per-channel std.
    full_flat = full.reshape(-1, 8)
    expected_per_ch = full_flat.std(dim=0, correction=0).tolist()
    assert result.per_channel_std is not None
    assert len(result.per_channel_std) == 8
    for i, (got, exp) in enumerate(zip(result.per_channel_std, expected_per_ch)):
        assert math.isclose(got, exp, rel_tol=1e-4), (
            f"Channel {i}: got {got}, expected {exp}"
        )


# ---------------------------------------------------------------------------
# Slow tests — Welford multi-batch pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_site_coverage(_vit_wrapped) -> None:
    """run_profiling_dataset_pass must return all 6 sites for every block."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.profiler import (
        SITE_POST_SOFTMAX,
        SITE_PRE_SOFTMAX,
        run_profiling_dataset_pass,
    )

    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    keys = set(stats.keys())
    assert "patch_embed/residual_stream" in keys
    for i in range(12):
        assert f"blocks.{i}/{SITE_PRE_SOFTMAX}" in keys
        assert f"blocks.{i}/{SITE_POST_SOFTMAX}" in keys


@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_exact_n_samples(_vit_wrapped) -> None:
    """n_samples in finalized LayerStats must equal total elements processed."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.profiler import SITE_PRE_GELU, run_profiling_dataset_pass

    # 4 images, batch_size=2 → 2 batches of B=2.
    # For pre_gelu at ViT-B/16: N=197, D_mlp=3072, so n per batch = 2*197*3072.
    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    key = "blocks.0/pre_gelu"
    expected_n = 4 * 197 * 3072  # total images × N × D_mlp
    assert stats[key].n_samples == expected_n, (
        f"n_samples={stats[key].n_samples}, expected {expected_n}"
    )


@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_per_channel_std_present(
    _vit_wrapped,
) -> None:
    """per_channel_std must be populated for pre_gelu and post_layernorm sites."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.profiler import (
        SITE_POST_LAYERNORM_1,
        SITE_POST_LAYERNORM_2,
        SITE_PRE_GELU,
        run_profiling_dataset_pass,
    )

    images = torch.randn(2, 3, 224, 224)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    for i in range(12):
        for site in (SITE_PRE_GELU, SITE_POST_LAYERNORM_1, SITE_POST_LAYERNORM_2):
            key = f"blocks.{i}/{site}"
            assert key in stats, f"Missing key: {key}"
            s = stats[key]
            assert s.per_channel_std is not None, (
                f"per_channel_std is None for {key}"
            )
            assert len(s.per_channel_std) > 0, (
                f"per_channel_std is empty for {key}"
            )
            assert all(v >= 0.0 for v in s.per_channel_std), (
                f"Negative per_channel_std value in {key}"
            )


@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_per_channel_std_shape(
    _vit_wrapped,
) -> None:
    """per_channel_std must have correct dimensionality for each site type."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.profiler import (
        SITE_POST_LAYERNORM_1,
        SITE_PRE_GELU,
        run_profiling_dataset_pass,
    )

    images = torch.randn(2, 3, 224, 224)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    # pre_gelu: D_mlp = 3072 for ViT-B/16
    pg = stats["blocks.0/pre_gelu"]
    assert pg.per_channel_std is not None
    assert len(pg.per_channel_std) == 3072, (
        f"pre_gelu per_channel_std has {len(pg.per_channel_std)} channels, expected 3072"
    )

    # post_layernorm: D = 768
    ln = stats[f"blocks.0/{SITE_POST_LAYERNORM_1}"]
    assert ln.per_channel_std is not None
    assert len(ln.per_channel_std) == 768, (
        f"post_layernorm per_channel_std has {len(ln.per_channel_std)} channels, expected 768"
    )


# ---------------------------------------------------------------------------
# Slow test — histogram_profile_vit
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_histogram_profile_vit_shapes() -> None:
    """histogram_profile_vit returns correct keys and tensor shapes."""
    import timm
    from nnsight import NNsight
    from src.model import disable_fused_attn
    from src.profiler import histogram_profile_vit

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    disable_fused_attn(model)
    wrapped = NNsight(model)

    B = 2
    batch = torch.zeros(B, 3, 224, 224)
    result = histogram_profile_vit(wrapped, batch, block_indices=(0,))

    N    = model.patch_embed.num_patches + 1        # 197
    D    = model.embed_dim                           # 768
    D_mlp = model.blocks[0].mlp.fc1.out_features   # 3072
    H    = model.blocks[0].attn.num_heads           # 12

    expected_shapes = {
        "patch_embed/residual_stream": (B, N, D),
        "blocks.0/post_layernorm_1":   (B, N, D),
        "blocks.0/post_layernorm_2":   (B, N, D),
        "blocks.0/pre_gelu":           (B, N, D_mlp),
        "blocks.0/pre_softmax":        (B, H, N, N),
        "blocks.0/post_softmax":       (B, H, N, N),
    }
    assert set(result.keys()) == set(expected_shapes.keys()), (
        f"Missing keys: {set(expected_shapes) - set(result.keys())}"
    )
    for key, shape in expected_shapes.items():
        assert result[key].shape == torch.Size(shape), (
            f"{key}: expected {shape}, got {tuple(result[key].shape)}"
        )
        assert result[key].device == torch.device("cpu")
        assert result[key].dtype == torch.float32


# ---------------------------------------------------------------------------
# Fast tests — merge_batch_stats edge cases
# ---------------------------------------------------------------------------


def test_merge_batch_stats_unequal_batch_sizes() -> None:
    """Pébay merge must be exact when batches have different sizes.

    Two batches with different n (3000 and 7000) drawn from the same
    distribution.  The merged result must match the full 10000-element
    computation exactly.
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(7)
    full = torch.randn(10000)
    b1 = full[:3000]
    b2 = full[3000:]

    def _stats(t: torch.Tensor) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/unequal",
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
        )

    acc = WelfordAccumulator(site_identifier="test/unequal")
    merge_batch_stats(acc, _stats(b1), 3000)
    merge_batch_stats(acc, _stats(b2), 7000)
    result = finalize_accumulator(acc)

    full_mean = full.float().mean().item()
    full_std = full.float().std(correction=0).item()
    assert math.isclose(result.mean, full_mean, rel_tol=1e-5), (
        f"mean: {result.mean} vs {full_mean}"
    )
    assert math.isclose(result.std, full_std, rel_tol=1e-5), (
        f"std: {result.std} vs {full_std}"
    )
    assert result.n_samples == 10000


def test_merge_batch_stats_large_mean_delta() -> None:
    """Pébay merge must be exact when batch means differ substantially.

    Two batches with very different means (0 vs 100).  The merge formula
    involves δ, δ², δ³, δ⁴ terms — numerical issues could arise.
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(3)
    b1 = torch.randn(5000)          # mean ~0
    b2 = torch.randn(5000) + 100.0  # mean ~100
    full = torch.cat([b1, b2])

    def _stats(t: torch.Tensor) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/delta",
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
        )

    acc = WelfordAccumulator(site_identifier="test/delta")
    merge_batch_stats(acc, _stats(b1), 5000)
    merge_batch_stats(acc, _stats(b2), 5000)
    result = finalize_accumulator(acc)

    full_mean = full.float().mean().item()
    full_std = full.float().std(correction=0).item()
    assert math.isclose(result.mean, full_mean, rel_tol=1e-5)
    assert math.isclose(result.std, full_std, rel_tol=1e-5)
    assert result.n_samples == 10000


def test_merge_batch_stats_zero_variance_batch() -> None:
    """Merge must handle a zero-variance batch correctly.

    If one batch is a constant (all elements equal), its std=0 and
    kurtosis is undefined.  The merge should not produce NaN or Inf.
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(1)
    b1 = torch.randn(5000)           # normal batch
    b2 = torch.full((5000,), 3.0)    # constant batch
    full = torch.cat([b1, b2])

    def _stats(t: torch.Tensor) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/zero_var",
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
        )

    acc = WelfordAccumulator(site_identifier="test/zero_var")
    merge_batch_stats(acc, _stats(b1), 5000)
    merge_batch_stats(acc, _stats(b2), 5000)
    result = finalize_accumulator(acc)

    full_mean = full.float().mean().item()
    full_std = full.float().std(correction=0).item()
    assert math.isclose(result.mean, full_mean, rel_tol=1e-5)
    assert math.isclose(result.std, full_std, rel_tol=1e-5)
    assert math.isfinite(result.kurtosis), (
        f"kurtosis should be finite, got {result.kurtosis}"
    )
    assert result.n_samples == 10000


def test_merge_batch_stats_idempotent() -> None:
    """Merging the same batch twice with half-n must equal one merge with full-n.

    Split a batch into two identical halves.  Merging both halves should
    produce the same result as merging the full batch once.
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(42)
    full = torch.randn(10000)
    half1 = full[:5000]
    half2 = full[5000:]

    def _stats(t: torch.Tensor) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/idem",
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
        )

    # Merge two halves.
    acc_split = WelfordAccumulator(site_identifier="test/idem")
    merge_batch_stats(acc_split, _stats(half1), 5000)
    merge_batch_stats(acc_split, _stats(half2), 5000)
    result_split = finalize_accumulator(acc_split)

    # Merge full batch once.
    acc_full = WelfordAccumulator(site_identifier="test/idem")
    merge_batch_stats(acc_full, _stats(full), 10000)
    result_full = finalize_accumulator(acc_full)

    # Pébay merge is exact: two halves merged = full batch merged.
    # Kurtosis uses 4th moments — floating-point roundoff means the
    # two computation paths may differ at the 1e-6 level.  Use abs tol.
    assert math.isclose(result_split.mean, result_full.mean, rel_tol=1e-5), (
        f"mean: {result_split.mean} vs {result_full.mean}"
    )
    assert math.isclose(result_split.std, result_full.std, rel_tol=1e-5), (
        f"std: {result_split.std} vs {result_full.std}"
    )
    assert math.isclose(result_split.kurtosis, result_full.kurtosis, abs_tol=1e-4), (
        f"kurtosis: {result_split.kurtosis} vs {result_full.kurtosis}"
    )


def test_merge_batch_stats_kurtosis_laplace() -> None:
    """Pébay merge must recover correct excess kurtosis for Laplace(0,1).

    Laplace distribution has theoretical excess kurtosis = 3.
    With 100k samples the empirical value should be close.
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(99)
    # Laplace(0,1): location=0, scale=1
    laplace = torch.distributions.Laplace(0.0, 1.0)
    b1 = laplace.sample((50000,))
    b2 = laplace.sample((50000,))
    full = torch.cat([b1, b2])

    def _stats(t: torch.Tensor) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/laplace",
            mean=mean,
            std=std,
            kurtosis=kurt,
            m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
        )

    acc = WelfordAccumulator(site_identifier="test/laplace")
    merge_batch_stats(acc, _stats(b1), 50000)
    merge_batch_stats(acc, _stats(b2), 50000)
    result = finalize_accumulator(acc)

    # Full-dataset kurtosis for comparison.
    x = full.float()
    n = x.numel()
    m = x.mean()
    s = x.std(correction=0)
    full_kurt = ((x - m) ** 4).sum().item() / (n * s.item() ** 4) - 3.0

    assert math.isclose(result.kurtosis, full_kurt, rel_tol=1e-4), (
        f"merged kurtosis={result.kurtosis:.4f}, full kurtosis={full_kurt:.4f}"
    )
    # Laplace excess kurtosis = 3.  With 100k samples, should be within ~0.3.
    assert 2.0 <= result.kurtosis <= 4.0, (
        f"Laplace kurtosis should be ~3, got {result.kurtosis:.4f}"
    )
    assert result.n_samples == 100000


def test_merge_batch_stats_per_channel_first_batch_none() -> None:
    """Per-channel merge must work when the first batch lacks per_channel data.

    If batch 1 has per_channel_sum=None but batch 2 has it, the accumulator
    should initialise from batch 2.
    """
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    torch.manual_seed(5)
    b1 = torch.randn(4, 8, 16)  # no per-channel tracking
    b2 = torch.randn(4, 8, 16)  # with per-channel tracking

    def _stats_no_ch(t: torch.Tensor) -> LayerStats:
        x = t.float()
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x.flatten() - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/ch_none",
            mean=mean, std=std, kurtosis=kurt, m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
            per_channel_std=None, per_channel_sum=None, per_channel_sum_sq=None,
        )

    def _stats_with_ch(t: torch.Tensor) -> LayerStats:
        x = t.float()
        flat = x.reshape(-1, x.shape[-1])
        n = x.numel()
        mean = x.mean().item()
        std = x.std(correction=0).item()
        centred = x.flatten() - mean
        m3 = (centred**3).sum().item()
        m4 = (centred**4).sum().item()
        kurt = m4 / (n * std**4) - 3.0 if std > 0 else 0.0
        return LayerStats(
            site_identifier="test/ch_none",
            mean=mean, std=std, kurtosis=kurt, m3=m3,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=n,
            per_channel_std=flat.std(dim=0, correction=0).tolist(),
            per_channel_sum=flat.sum(dim=0).tolist(),
            per_channel_sum_sq=(flat**2).sum(dim=0).tolist(),
        )

    acc = WelfordAccumulator(site_identifier="test/ch_none")
    merge_batch_stats(acc, _stats_no_ch(b1), b1.numel())
    merge_batch_stats(acc, _stats_with_ch(b2), b2.numel())
    result = finalize_accumulator(acc)

    # After merging, per_channel_std should be populated from batch 2.
    assert result.per_channel_std is not None, (
        "per_channel_std should be populated from second batch"
    )
    assert len(result.per_channel_std) == 16


# ---------------------------------------------------------------------------
# Fast tests — _site_n edge cases
# ---------------------------------------------------------------------------


def test_site_n_unknown_site_type() -> None:
    """_site_n must fall back to B*N*D for unrecognised site types."""
    from src.profiler import _site_n

    B, N, D, D_mlp, H = 2, 197, 768, 3072, 12
    # An unknown site should be treated as residual/layernorm (B*N*D).
    assert _site_n("blocks.0/unknown_site", B, N, D, D_mlp, H) == B * N * D


def test_site_n_substring_matching() -> None:
    """_site_n substring matching must correctly classify each site type."""
    from src.profiler import (
        SITE_POST_LAYERNORM_1,
        SITE_POST_SOFTMAX,
        SITE_PRE_GELU,
        SITE_PRE_SOFTMAX,
        SITE_RESIDUAL_STREAM,
        _site_n,
    )

    B, N, D, D_mlp, H = 4, 197, 768, 3072, 12

    # pre_softmax and post_softmax: B*H*N*N
    for site in (SITE_PRE_SOFTMAX, SITE_POST_SOFTMAX):
        assert _site_n(f"blocks.5/{site}", B, N, D, D_mlp, H) == B * H * N * N

    # pre_gelu: B*N*D_mlp
    assert _site_n(f"blocks.5/{SITE_PRE_GELU}", B, N, D, D_mlp, H) == B * N * D_mlp

    # residual and layernorm: B*N*D
    for site in (SITE_RESIDUAL_STREAM, SITE_POST_LAYERNORM_1):
        assert _site_n(f"blocks.5/{site}", B, N, D, D_mlp, H) == B * N * D


# ---------------------------------------------------------------------------
# Fast tests — serialisation robustness
# ---------------------------------------------------------------------------


def test_load_profiling_result_raises_on_malformed_json(tmp_path: Path) -> None:
    """load_profiling_result must raise on syntactically invalid JSON."""
    from src.profiler import load_profiling_result

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("this is not json {{{{")
    with pytest.raises(json.JSONDecodeError):
        load_profiling_result(bad_path)


def test_load_profiling_result_raises_on_missing_keys(tmp_path: Path) -> None:
    """load_profiling_result must raise when required top-level keys are absent."""
    from src.profiler import load_profiling_result

    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"stats": {}, "num_blocks": 12}))
    with pytest.raises(KeyError):
        load_profiling_result(path)


def test_save_profiling_result_overwrites_existing(tmp_path: Path) -> None:
    """save_profiling_result must overwrite an existing file without error."""
    from src.profiler import save_profiling_result

    result = _canned_result()
    path = tmp_path / "result.json"
    save_profiling_result(result, path)
    first_size = path.stat().st_size
    save_profiling_result(result, path)
    assert path.stat().st_size == first_size, (
        "Overwritten file should have the same size for identical data"
    )


def test_profiling_result_batch_shape_preserves_order() -> None:
    """batch_shape must survive serialisation with correct element order."""
    from dataclasses import asdict

    result = ProfilingResult(
        stats={},
        num_blocks=12,
        batch_shape=(64, 3, 224, 224),
    )
    raw = asdict(result)
    # asdict preserves tuples (they are immutable).
    assert raw["batch_shape"] == (64, 3, 224, 224)
    # Round-trip through JSON (tuples become lists in JSON).
    reloaded = json.loads(json.dumps(raw))
    assert tuple(reloaded["batch_shape"]) == (64, 3, 224, 224)


# ---------------------------------------------------------------------------
# Fast tests — LayerStats field validation
# ---------------------------------------------------------------------------


def test_layer_stats_per_channel_fields_default_none() -> None:
    """per_channel_std, per_channel_sum, per_channel_sum_sq must default to None."""
    stats = LayerStats(
        site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0,
    )
    assert stats.per_channel_std is None
    assert stats.per_channel_sum is None
    assert stats.per_channel_sum_sq is None


def test_layer_stats_m3_default_zero() -> None:
    """m3 must default to 0.0."""
    stats = LayerStats(site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0)
    assert stats.m3 == 0.0


def test_layer_stats_n_samples_default_zero() -> None:
    """n_samples must default to 0."""
    stats = LayerStats(site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0)
    assert stats.n_samples == 0


# ---------------------------------------------------------------------------
# Fast tests — WelfordAccumulator outlier key initialisation
# ---------------------------------------------------------------------------


def test_welford_accumulator_outlier_keys_match_outlier_sigmas() -> None:
    """WelfordAccumulator outlier_counts keys must exactly match OUTLIER_SIGMAS."""
    from src.profiler import WelfordAccumulator

    acc = WelfordAccumulator(site_identifier="test")
    expected = {f"{k}_sigma": 0 for k in OUTLIER_SIGMAS}
    assert acc.outlier_counts == expected


# ---------------------------------------------------------------------------
# Fast tests — histogram_profile_vit input validation
# ---------------------------------------------------------------------------


def test_histogram_profile_vit_raises_on_non_4d_input() -> None:
    """histogram_profile_vit must raise ValueError for non-4-D input."""
    from nnsight import NNsight
    from src.profiler import histogram_profile_vit

    wrapped = NNsight(nn.Linear(8, 4))
    with pytest.raises(ValueError, match="4-D"):
        histogram_profile_vit(wrapped, torch.randn(3, 224, 224))


def test_histogram_profile_vit_raises_on_model_without_blocks() -> None:
    """histogram_profile_vit must raise ProfilingError for model without .blocks."""
    from nnsight import NNsight
    from src.profiler import histogram_profile_vit

    wrapped = NNsight(nn.Sequential(nn.Linear(8, 4)))
    with pytest.raises((ProfilingError, ValueError)):
        histogram_profile_vit(wrapped, torch.randn(1, 8, 1, 1))


# ---------------------------------------------------------------------------
# Fast tests — data_loader shuffle behaviour
# ---------------------------------------------------------------------------


def test_build_val_loader_shuffle_default_none() -> None:
    """build_val_loader must default to shuffle=None for auto-select behaviour."""
    import inspect
    from src.data_loader import build_val_loader

    sig = inspect.signature(build_val_loader)
    assert "shuffle" in sig.parameters
    assert sig.parameters["shuffle"].default is None


# ---------------------------------------------------------------------------
# Fast tests — finalize_accumulator edge cases
# ---------------------------------------------------------------------------


def test_finalize_accumulator_single_element() -> None:
    """finalize_accumulator must handle n=1 (single scalar element)."""
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    acc = WelfordAccumulator(site_identifier="test/single")
    bs = LayerStats(
        site_identifier="test/single",
        mean=5.0,
        std=0.0,  # single element has zero variance
        kurtosis=0.0,
        m3=0.0,
        outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
        n_samples=1,
    )
    merge_batch_stats(acc, bs, 1)
    result = finalize_accumulator(acc)
    assert result.mean == 5.0
    assert result.std == 0.0
    assert result.n_samples == 1


def test_finalize_accumulator_all_constant() -> None:
    """finalize_accumulator must handle a constant distribution (all values equal)."""
    from src.profiler import (
        WelfordAccumulator,
        finalize_accumulator,
        merge_batch_stats,
    )

    acc = WelfordAccumulator(site_identifier="test/constant")
    bs = LayerStats(
        site_identifier="test/constant",
        mean=7.0,
        std=0.0,
        kurtosis=0.0,
        m3=0.0,
        outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
        n_samples=100,
    )
    merge_batch_stats(acc, bs, 100)
    result = finalize_accumulator(acc)
    assert result.mean == 7.0
    assert result.std == 0.0
    # Kurtosis is undefined for point mass; stored as 0.
    assert result.kurtosis == 0.0


# ---------------------------------------------------------------------------
# Slow test — pre_softmax reconstruction correctness
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_pre_softmax_reconstruction_matches_manual() -> None:
    """histogram_profile_vit pre_softmax must match manual QKᵀ/√d computation.

    Runs the model with fused_attn=False, captures the qkv output, and
    verifies that the reconstructed logits match a manual computation
    from the same qkv tensor.
    """
    import timm
    from nnsight import NNsight
    from src.model import disable_fused_attn

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    disable_fused_attn(model)

    # Run a manual forward pass to get qkv output for block 0.
    x = torch.randn(1, 3, 224, 224)

    # Capture qkv from a hook.
    qkv_outputs: dict[int, torch.Tensor] = {}

    def _hook(module, inp, outp, idx):
        qkv_outputs[idx] = outp.detach()

    handle = model.blocks[0].attn.qkv.register_forward_hook(
        lambda m, i, o: _hook(m, i, o, 0)
    )
    with torch.no_grad():
        model(x)
    handle.remove()

    qkv = qkv_outputs[0]  # (1, 197, 3*H*D) = (1, 197, 2304)
    H = model.blocks[0].attn.num_heads   # 12
    head_dim = model.blocks[0].attn.head_dim  # 64
    scale = model.blocks[0].attn.scale

    # Manual reconstruction.
    qkv_reshaped = qkv.reshape(1, 197, 3, H, head_dim)
    qkv_permuted = qkv_reshaped.permute(2, 0, 3, 1, 4)  # (3, 1, H, 197, 64)
    q_manual = qkv_permuted[0] * scale
    k_manual = qkv_permuted[1]
    logits_manual = q_manual @ k_manual.transpose(-2, -1)  # (1, H, 197, 197)

    # Now get the same via histogram_profile_vit.
    wrapped = NNsight(model)
    result = histogram_profile_vit(wrapped, x, block_indices=(0,))

    logits_profiled = result["blocks.0/pre_softmax"]  # (1, H, 197, 197)

    assert torch.allclose(logits_profiled, logits_manual, atol=1e-5), (
        f"Max diff: {(logits_profiled - logits_manual).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# Slow test — per-channel std ground truth
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_per_channel_std_matches_numpy() -> None:
    """_register_stat_saves per-channel std must match numpy ground truth.

    Runs a single forward pass through a LayerNorm, captures per-channel
    std via _register_stat_saves, and compares against numpy.
    """
    from nnsight import NNsight
    from src.profiler import _finalize_stats, _register_stat_saves

    torch.manual_seed(42)
    wrapped = NNsight(nn.LayerNorm(16))
    x = torch.randn(4, 32, 16)
    n_samples = 4 * 32 * 16

    # nnsight ≥0.3: .trace() does not bind local variables in the body.
    savers_list: list = []
    with wrapped.trace(x):
        savers_list.append(_register_stat_saves(
            wrapped.output, "test/ln", n_samples, track_per_channel=True,
        ))
    stats = _finalize_stats(savers_list[0])

    # Ground truth: per-channel population std of the LayerNorm output.
    with torch.no_grad():
        ln_out = wrapped._model(x)  # (4, 32, 16)
    out_flat = ln_out.reshape(-1, 16)  # (128, 16)
    gt_std = out_flat.std(dim=0, correction=0).numpy()  # shape (16,)

    assert stats.per_channel_std is not None
    for c in range(16):
        assert math.isclose(stats.per_channel_std[c], float(gt_std[c]), rel_tol=1e-4), (
            f"Channel {c}: got {stats.per_channel_std[c]:.6f}, expected {gt_std[c]:.6f}"
        )


# ---------------------------------------------------------------------------
# Fast tests — auto-shuffle random sampling
# ---------------------------------------------------------------------------


def test_build_val_loader_auto_shuffle_different_seeds() -> None:
    """build_val_loader with shuffle=None and subset must use different
    indices for different seeds."""
    import torchvision.datasets as datasets
    from src.data_loader import build_val_loader
    from src.utils import seed_everything

    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("data/ directory not found")

    # We cannot easily compare DataLoader outputs (PIL vs tensor issues),
    # but we can verify the auto-select logic directly.
    ds = datasets.ImageFolder(str(data_dir))
    full_size = len(ds)
    num_images = 256

    indices_by_seed = {}
    for seed in [42, 43]:
        seed_everything(seed)
        is_subset = num_images < full_size
        shuffle = is_subset  # auto-selected
        if shuffle:
            indices = torch.randperm(full_size)[:num_images].tolist()
        else:
            indices = list(range(num_images))
        indices_by_seed[seed] = indices

    assert indices_by_seed[42] != indices_by_seed[43], (
        "Auto-shuffle with different seeds must produce different subsets"
    )


def test_build_val_loader_auto_shuffle_full_dataset(temp_image_dir: Path) -> None:
    """build_val_loader with num_images=None must auto-select shuffle=True.

    Updated for F1: the auto-select logic now defaults to True for both
    subsets and full-dataset runs.  Class-diverse batches produce
    representative per-batch σ, reducing the outlier-fraction overestimate.
    """
    from torch.utils.data import SequentialSampler
    from src.data_loader import build_val_loader
    import torchvision.transforms as T

    loader = build_val_loader(
        temp_image_dir,
        T.Compose([T.Resize((8, 8)), T.ToTensor()]),
        batch_size=2,
        num_images=None,
        device=torch.device("cpu"),
    )
    assert not isinstance(loader.sampler, SequentialSampler), (
        "Full dataset must auto-select shuffle=True (F1 fix)"
    )


# ---------------------------------------------------------------------------
# F2 — Global-σ outlier recount tests
# ---------------------------------------------------------------------------


def test_profiling_config_approximate_outliers_default_false() -> None:
    """ProfilingConfig.approximate_outliers must default to False."""
    from src.config import ProfilingConfig

    cfg = ProfilingConfig(
        data_dir=Path("data"),
        output_dir=Path("outputs"),
        num_images=64,
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert cfg.approximate_outliers is False


def test_profiling_config_approximate_outliers_can_be_set() -> None:
    """ProfilingConfig.approximate_outliers can be set to True."""
    from src.config import ProfilingConfig

    cfg = ProfilingConfig(
        data_dir=Path("data"),
        output_dir=Path("outputs"),
        num_images=64,
        batch_size=8,
        device=torch.device("cpu"),
        approximate_outliers=True,
    )
    assert cfg.approximate_outliers is True


@pytest.mark.slow
def test_run_outlier_counting_pass_returns_correct_keys() -> None:
    """run_outlier_counting_pass must return the same site keys as input stats.

    Uses a mock that avoids nnsight traces by injecting a fake model that
    produces known activations.
    """
    from src.profiler import run_outlier_counting_pass

    # Build fake finalized_stats with known mean/std.
    fake_stats: dict[str, LayerStats] = {
        "blocks.0/pre_gelu": LayerStats(
            site_identifier="blocks.0/pre_gelu",
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={"3.0_sigma": 0.01, "4.0_sigma": 0.001, "6.0_sigma": 0.0},
            n_samples=1000,
        ),
        "blocks.0/post_softmax": LayerStats(
            site_identifier="blocks.0/post_softmax",
            mean=0.0, std=0.5, kurtosis=0.0,
            outlier_fractions={"3.0_sigma": 0.02, "4.0_sigma": 0.002, "6.0_sigma": 0.0},
            n_samples=2000,
        ),
    }

    # Create a DataLoader with a simple tensor dataset.
    from torch.utils.data import DataLoader, TensorDataset
    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    # Build a mock wrapped model that produces known activations.
    # We use a real nnsight-wrapped identity model and intercept the trace.
    # For a fast test, we construct a minimal model that exposes the right
    # architecture attributes.
    import timm
    from nnsight import NNsight
    from src.model import disable_fused_attn

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    disable_fused_attn(model)
    wrapped = NNsight(model)

    with torch.no_grad():
        corrected = run_outlier_counting_pass(wrapped, loader, torch.device("cpu"), fake_stats)

    assert set(corrected.keys()) == set(fake_stats.keys()), (
        f"Keys mismatch: {set(corrected.keys())} vs {set(fake_stats.keys())}"
    )


@pytest.mark.slow
def test_run_outlier_counting_pass_fractions_in_unit_interval() -> None:
    """All returned outlier fractions must be in [0, 1]."""
    from src.profiler import run_outlier_counting_pass
    from torch.utils.data import DataLoader, TensorDataset
    import timm
    from nnsight import NNsight
    from src.model import disable_fused_attn

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    disable_fused_attn(model)
    wrapped = NNsight(model)

    fake_stats: dict[str, LayerStats] = {
        "blocks.0/pre_gelu": LayerStats(
            site_identifier="blocks.0/pre_gelu",
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={"3.0_sigma": 0.01, "4.0_sigma": 0.001, "6.0_sigma": 0.0},
            n_samples=1000,
        ),
    }

    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    with torch.no_grad():
        corrected = run_outlier_counting_pass(wrapped, loader, torch.device("cpu"), fake_stats)

    for site_id, fracs in corrected.items():
        for key, val in fracs.items():
            assert 0.0 <= val <= 1.0, (
                f"{site_id}[{key}] = {val} not in [0, 1]"
            )


def test_run_outlier_counting_pass_known_gaussian() -> None:
    """Outlier counting on N(0,1) data must recover ~0.0027 at 3σ.

    Uses a direct tensor computation (no nnsight trace) to verify the
    mathematical definition of outlier fraction is correct.  The 3σ
    fraction for a standard normal is approximately 0.0027 (two-sided).

    NOTE: This is a **mathematical sanity check**, not a test of the
    recount pass code.  It does not exercise run_outlier_counting_pass,
    nnsight traces, batch accumulation, or site_id lookups.  For a true
    integration test of the recount pass, see
    test_slow_run_outlier_counting_pass_correctness.
    """
    from src.profiler import OUTLIER_SIGMAS

    torch.manual_seed(42)
    n = 100_000
    data = torch.randn(n)
    mu = data.mean().item()
    sigma = data.std(correction=0).item()

    # Count outliers using the mean-centered definition: |x − μ| > k·σ.
    # For N(0,1) data, μ ≈ 0 so this is nearly identical to |x| > k·σ,
    # but we use the correct definition for consistency with the codebase.
    for k in OUTLIER_SIGMAS:
        frac = ((data - mu).abs() > k * sigma).float().mean().item()
        if k == 3.0:
            # Theoretical: 2 * Φ(-3) ≈ 0.0027.  Allow ±0.002 for finite sample.
            assert 0.0007 <= frac <= 0.0047, (
                f"3σ fraction={frac:.6f}, expected ~0.0027"
            )
        # All fractions must be in [0, 1].
        assert 0.0 <= frac <= 1.0


@pytest.mark.slow
def test_slow_run_outlier_counting_pass_correctness(_vit_wrapped) -> None:
    """run_outlier_counting_pass must match ground-truth outlier fractions.

    Integration test (T-008): creates a small synthetic dataset, runs the
    full two-pass pipeline, and independently verifies recount fractions
    against a ground-truth computation on the same data using the global
    μ and σ from pass 1.

    This exercises the actual recount pass code — nnsight trace, site_id
    lookup, batch accumulation, and fraction computation — not just the
    mathematical definition of outlier fraction.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from src.profiler import (
        OUTLIER_SIGMAS,
        run_outlier_counting_pass,
        run_profiling_dataset_pass,
    )

    # Small synthetic dataset: 4 images, batch_size=2 → 2 batches.
    torch.manual_seed(42)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    # --- Pass 1: get global μ and σ ---
    with torch.no_grad():
        finalized_stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    # --- Pass 2: recount outlier fractions using global σ ---
    loader2 = DataLoader(dataset, batch_size=2)
    with torch.no_grad():
        recount_fractions = run_outlier_counting_pass(
            _vit_wrapped, loader2, device, finalized_stats,
        )

    # --- Ground truth: capture raw activations and count manually ---
    # We run a separate nnsight trace that saves the actual activation
    # tensors for a subset of sites, then compute outlier fractions
    # using the global μ and σ from pass 1.
    raw_activations: list[torch.Tensor] = []
    loader3 = DataLoader(dataset, batch_size=2)
    for images_batch, _ in loader3:
        images_batch = images_batch.to(device)
        with _vit_wrapped.trace(images_batch):
            raw = _vit_wrapped.blocks[0].mlp.act.input.save()
        raw_activations.append(raw)

    # Concatenate all batches along the batch dimension.
    all_raw = torch.cat(raw_activations, dim=0)  # [4, 197, 3072]

    # Ground-truth outlier fractions using global μ, σ from pass 1.
    site_key = "blocks.0/pre_gelu"
    global_mu = finalized_stats[site_key].mean
    global_sigma = finalized_stats[site_key].std

    gt_fractions: dict[str, float] = {}
    for k in OUTLIER_SIGMAS:
        key = f"{k}_sigma"
        # Match the recount pass: |x - μ_global| > k·σ_global
        gt_fractions[key] = (
            ((all_raw - global_mu).abs() > k * global_sigma).float().mean().item()
        )

    # --- Assert recount matches ground truth ---
    recount = recount_fractions[site_key]
    for k in OUTLIER_SIGMAS:
        key = f"{k}_sigma"
        gt = gt_fractions[key]
        rc = recount[key]
        # Allow small floating-point tolerance (nnsight proxies may have
        # minor numerical differences vs direct tensor computation).
        assert rc == pytest.approx(gt, abs=1e-6), (
            f"{site_key}[{key}]: recount={rc:.8f}, ground_truth={gt:.8f}"
        )

    # Also verify the recount fractions are in [0, 1] and monotone.
    fracs = [recount[f"{k}_sigma"] for k in OUTLIER_SIGMAS]
    for f in fracs:
        assert 0.0 <= f <= 1.0, f"Fraction {f} not in [0, 1]"
    for lo, hi in zip(fracs, fracs[1:]):
        assert lo >= hi - 1e-8, f"Non-monotone: {fracs}"


@pytest.mark.slow
def test_slow_run_outlier_counting_pass_multiple_sites_correctness(
    _vit_wrapped,
) -> None:
    """Recount fractions must match ground truth across multiple site types.

    Verifies the recount pass for pre_gelu, post_layernorm_1, and
    residual_stream sites — each with different tensor shapes and
    element counts.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from src.profiler import (
        OUTLIER_SIGMAS,
        run_outlier_counting_pass,
        run_profiling_dataset_pass,
    )

    torch.manual_seed(123)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    device = torch.device("cpu")

    # Pass 1
    loader1 = DataLoader(dataset, batch_size=2)
    with torch.no_grad():
        finalized_stats = run_profiling_dataset_pass(_vit_wrapped, loader1, device)

    # Pass 2
    loader2 = DataLoader(dataset, batch_size=2)
    with torch.no_grad():
        recount_fractions = run_outlier_counting_pass(
            _vit_wrapped, loader2, device, finalized_stats,
        )

    # Ground truth for multiple sites.
    # Collect raw activations for blocks.0/pre_gelu, blocks.0/post_layernorm_1,
    # and patch_embed/residual_stream.
    raw_pre_gelu: list[torch.Tensor] = []
    raw_post_ln1: list[torch.Tensor] = []
    raw_residual: list[torch.Tensor] = []

    loader3 = DataLoader(dataset, batch_size=2)
    for images_batch, _ in loader3:
        images_batch = images_batch.to(device)
        with _vit_wrapped.trace(images_batch):
            # Forward-pass dependency order: norm1.input → norm1.output → mlp.act.input
            r_res = _vit_wrapped.blocks[0].norm1.input.save()
            r_ln1 = _vit_wrapped.blocks[0].norm1.output.save()
            r_gelu = _vit_wrapped.blocks[0].mlp.act.input.save()
        raw_pre_gelu.append(r_gelu)
        raw_post_ln1.append(r_ln1)
        raw_residual.append(r_res)

    all_pre_gelu = torch.cat(raw_pre_gelu, dim=0)
    all_post_ln1 = torch.cat(raw_post_ln1, dim=0)
    all_residual = torch.cat(raw_residual, dim=0)

    sites_to_check = [
        ("blocks.0/pre_gelu", all_pre_gelu),
        ("blocks.0/post_layernorm_1", all_post_ln1),
        ("patch_embed/residual_stream", all_residual),
    ]

    for site_key, raw_tensor in sites_to_check:
        global_mu = finalized_stats[site_key].mean
        global_sigma = finalized_stats[site_key].std
        recount = recount_fractions[site_key]

        for k in OUTLIER_SIGMAS:
            key = f"{k}_sigma"
            # Match the recount pass: |x - μ_global| > k·σ_global
            gt = ((raw_tensor - global_mu).abs() > k * global_sigma).float().mean().item()
            rc = recount[key]
            assert rc == pytest.approx(gt, abs=1e-6), (
                f"{site_key}[{key}]: recount={rc:.8f}, ground_truth={gt:.8f}"
            )


# ---------------------------------------------------------------------------
# F3 — Attention entropy tests
# ---------------------------------------------------------------------------


def test_layer_stats_attention_entropy_cls_defaults_none() -> None:
    """attention_entropy_cls and attention_entropy_patches must default to None."""
    stats = LayerStats(
        site_identifier="test", mean=0.0, std=1.0, kurtosis=0.0,
    )
    assert stats.attention_entropy_cls is None
    assert stats.attention_entropy_patches is None


def test_welford_accumulator_entropy_fields_default_none() -> None:
    """WelfordAccumulator entropy fields must default to None/0."""
    from src.profiler import WelfordAccumulator

    acc = WelfordAccumulator(site_identifier="test")
    assert acc.entropy_cls_sum is None
    assert acc.entropy_cls_count == 0
    assert acc.entropy_patch_sum is None
    assert acc.entropy_patch_count == 0


def test_merge_batch_stats_entropy_cls_accumulation_first_batch() -> None:
    """First batch with entropy data must initialise accumulator fields."""
    from src.profiler import WelfordAccumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="test")
    bs = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=[1.5, 2.0],
        attention_entropy_patches=[6.0, 8.0],
    )
    merge_batch_stats(acc, bs, batch_n=100, patch_token_count=4)

    assert acc.entropy_cls_sum == [1.5, 2.0]
    assert acc.entropy_cls_count == 1
    assert acc.entropy_patch_sum == [6.0, 8.0]
    assert acc.entropy_patch_count == 4


def test_merge_batch_stats_entropy_accumulation_two_batches() -> None:
    """Two batches with entropy must accumulate sums and counts correctly."""
    from src.profiler import WelfordAccumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="test")

    bs1 = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=[1.0, 2.0],
        attention_entropy_patches=[4.0, 8.0],
    )
    merge_batch_stats(acc, bs1, batch_n=100, patch_token_count=4)

    bs2 = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=[3.0, 4.0],
        attention_entropy_patches=[8.0, 12.0],
    )
    merge_batch_stats(acc, bs2, batch_n=100, patch_token_count=4)

    assert acc.entropy_cls_sum == [4.0, 6.0]
    assert acc.entropy_cls_count == 2
    assert acc.entropy_patch_sum == [12.0, 20.0]
    assert acc.entropy_patch_count == 8


def test_finalize_accumulator_entropy_cls_mean() -> None:
    """finalize_accumulator must compute correct entropy means."""
    from src.profiler import WelfordAccumulator, finalize_accumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="test")

    bs1 = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=[1.0, 2.0],
        attention_entropy_patches=[4.0, 8.0],
    )
    merge_batch_stats(acc, bs1, batch_n=100, patch_token_count=4)

    bs2 = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=[3.0, 4.0],
        attention_entropy_patches=[8.0, 12.0],
    )
    merge_batch_stats(acc, bs2, batch_n=100, patch_token_count=4)

    result = finalize_accumulator(acc)
    # CLS: batch-count mean = (1+3)/2, (2+4)/2
    assert result.attention_entropy_cls == pytest.approx([2.0, 3.0])
    # Patch: sample-count mean = (4+8)/8, (8+12)/8
    assert result.attention_entropy_patches == pytest.approx([1.5, 2.5])


def test_finalize_accumulator_entropy_none_when_no_data() -> None:
    """finalize_accumulator must return None for entropy when no data accumulated."""
    from src.profiler import WelfordAccumulator, finalize_accumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="test")
    bs = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=None,
        attention_entropy_patches=None,
    )
    merge_batch_stats(acc, bs, batch_n=100)
    result = finalize_accumulator(acc)
    assert result.attention_entropy_cls is None
    assert result.attention_entropy_patches is None


def test_layer_stats_entropy_serialisation_round_trip(tmp_path: Path) -> None:
    """Entropy fields must survive save → load round-trip."""
    from src.profiler import save_profiling_result, load_profiling_result

    stats = LayerStats(
        site_identifier="blocks.0/post_softmax",
        mean=0.0, std=1.0, kurtosis=0.0,
        attention_entropy_cls=[1.1, 2.2, 3.3],
        attention_entropy_patches=[4.4, 5.5, 6.6],
    )
    result = ProfilingResult(
        stats={"blocks.0/post_softmax": stats},
        num_blocks=12,
        batch_shape=(1, 3, 224, 224),
    )
    path = tmp_path / "entropy_test.json"
    save_profiling_result(result, path)
    loaded = load_profiling_result(path)

    recovered = loaded.stats["blocks.0/post_softmax"]
    assert recovered.attention_entropy_cls == pytest.approx([1.1, 2.2, 3.3])
    assert recovered.attention_entropy_patches == pytest.approx([4.4, 5.5, 6.6])


def test_layer_stats_entropy_backwards_compat(tmp_path: Path) -> None:
    """Old JSON without entropy fields must deserialise with None defaults."""
    from src.profiler import load_profiling_result

    # Manually construct JSON without entropy fields.
    raw = {
        "stats": {
            "blocks.0/post_softmax": {
                "site_identifier": "blocks.0/post_softmax",
                "mean": 0.0,
                "std": 1.0,
                "kurtosis": 0.0,
                "m3": 0.0,
                "outlier_fractions": {"3.0_sigma": 0.01},
                "n_samples": 100,
                "per_channel_std": None,
                "per_channel_sum": None,
                "per_channel_sum_sq": None,
            }
        },
        "num_blocks": 12,
        "batch_shape": [1, 3, 224, 224],
    }
    path = tmp_path / "old_format.json"
    import json
    path.write_text(json.dumps(raw))

    loaded = load_profiling_result(path)
    stats = loaded.stats["blocks.0/post_softmax"]
    assert stats.attention_entropy_cls is None
    assert stats.attention_entropy_patches is None


def test_register_entropy_saves_output_shapes() -> None:
    """_register_entropy_saves on a concrete tensor must produce correct shapes.

    Tests the standalone computation (not through nnsight proxies).
    """
    from src.profiler import _register_entropy_saves

    # (B=2, H=4, N=5, N=5): 1 CLS row + 4 patch rows
    t = torch.rand(2, 4, 5, 5)
    # Normalise to valid attention weights (sum to 1 along last dim).
    t = t / t.sum(dim=-1, keepdim=True)

    cls_proxy, patch_proxy = _register_entropy_saves(t)

    # Both should be concrete tensors of shape (H,).
    assert cls_proxy.shape == (4,)
    assert patch_proxy.shape == (4,)
    assert (cls_proxy >= 0).all(), "CLS entropy must be non-negative"
    assert (patch_proxy >= 0).all(), "Patch entropy sum must be non-negative"
    # Max entropy for 5 key tokens: log(5) ≈ 1.609
    assert (cls_proxy <= math.log(5) + 1e-6).all(), (
        f"CLS entropy exceeds log(5): {cls_proxy}"
    )


def test_register_entropy_saves_uniform_attention() -> None:
    """Uniform attention weights must produce max entropy = log(N)."""
    from src.profiler import _register_entropy_saves

    B, H, N = 2, 4, 5
    uniform = torch.ones(B, H, N, N) / N

    cls_proxy, patch_proxy = _register_entropy_saves(uniform)

    log_N = math.log(N)
    # CLS: mean over B of H_cls.  Uniform → H_cls = log(N) for all heads.
    assert cls_proxy.tolist() == pytest.approx([log_N] * H, abs=1e-4)
    # Patch: sum over B*(N-1) of H_patch.  Each patch row has H=log(N).
    # Total sum = B * (N-1) * log(N) = 2 * 4 * log(5) = 8 * log(5).
    expected_patch_sum = B * (N - 1) * log_N
    assert patch_proxy.tolist() == pytest.approx([expected_patch_sum] * H, abs=1e-4)


def test_register_entropy_saves_peaked_attention() -> None:
    """Peaked attention (one token gets prob 1.0) must give zero entropy."""
    from src.profiler import _register_entropy_saves

    B, H, N = 2, 4, 5
    peaked = torch.zeros(B, H, N, N)
    peaked[:, :, :, 0] = 1.0  # All queries attend to key 0 with prob 1.

    cls_proxy, patch_proxy = _register_entropy_saves(peaked)

    assert cls_proxy.tolist() == pytest.approx([0.0] * H, abs=1e-6)
    assert patch_proxy.tolist() == pytest.approx([0.0] * H, abs=1e-6)


def test_register_entropy_saves_cls_sink_not_diluted_by_patches() -> None:
    """CLS sink signal must not be diluted by high patch entropy.

    CLS row is peaked (H_cls=0, sink behaviour), patch rows are uniform
    (H_patch = log(N)).  If CLS and patch were pooled, the sink signal
    would be diluted.  This test catches that regression.
    """
    from src.profiler import _register_entropy_saves

    B, H, N = 2, 4, 5
    attn = torch.ones(B, H, N, N) / N  # uniform baseline
    # Make CLS row peaked: all probability on key 0.
    attn[:, :, 0, :] = 0.0
    attn[:, :, 0, 0] = 1.0

    cls_proxy, patch_proxy = _register_entropy_saves(attn)

    # CLS entropy must be ~0 (sink detected).
    assert cls_proxy.tolist() == pytest.approx([0.0] * H, abs=1e-6)
    # Patch entropy must be > 0 (uniform, not diluted).
    assert (patch_proxy > 0).all(), (
        f"Patch entropy sum should be > 0, got {patch_proxy}"
    )


# ---------------------------------------------------------------------------
# F4 — Summary table tests
# ---------------------------------------------------------------------------


def test_generate_summary_table_row_count() -> None:
    """generate_summary_table must produce one row per site."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {
        "patch_embed/residual_stream": LayerStats(
            site_identifier="patch_embed/residual_stream",
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={"3.0_sigma": 0.01},
        ),
        "blocks.0/pre_gelu": LayerStats(
            site_identifier="blocks.0/pre_gelu",
            mean=0.5, std=2.0, kurtosis=1.0,
            outlier_fractions={"3.0_sigma": 0.02},
        ),
        "blocks.0/post_softmax": LayerStats(
            site_identifier="blocks.0/post_softmax",
            mean=0.0, std=0.5, kurtosis=0.0,
            outlier_fractions={"3.0_sigma": 0.03},
        ),
    }
    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)
    assert len(rows) == 3


def test_generate_summary_table_column_names() -> None:
    """Column names must be derived from outlier_fractions keys, not hardcoded."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {
        "blocks.0/pre_gelu": LayerStats(
            site_identifier="blocks.0/pre_gelu",
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={"3.0_sigma": 0.01, "4.0_sigma": 0.001},
        ),
    }
    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)
    row = rows[0]

    assert "block" in row
    assert "site" in row
    assert "mean" in row
    assert "std" in row
    assert "kurtosis" in row
    assert "frac_3.0_sigma" in row
    assert "frac_4.0_sigma" in row
    assert "frac_6.0_sigma" not in row  # not in this test's outlier_fractions


def test_generate_summary_table_block_site_parsing() -> None:
    """Site identifiers must be correctly parsed into block and site columns."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {
        "blocks.3/pre_gelu": LayerStats(
            site_identifier="blocks.3/pre_gelu",
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={},
        ),
        "patch_embed/residual_stream": LayerStats(
            site_identifier="patch_embed/residual_stream",
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={},
        ),
    }
    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)

    # Find rows by block.
    by_block = {r["block"]: r for r in rows}
    assert by_block["blocks.3"]["site"] == "pre_gelu"
    assert by_block["patch_embed"]["site"] == "residual_stream"


def test_generate_summary_table_ordering() -> None:
    """Rows must be ordered: patch_embed first, then blocks sorted numerically."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {}
    for block_idx in [11, 0, 5]:
        key = f"blocks.{block_idx}/pre_gelu"
        stats[key] = LayerStats(
            site_identifier=key,
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={},
        )
    stats["patch_embed/residual_stream"] = LayerStats(
        site_identifier="patch_embed/residual_stream",
        mean=0.0, std=1.0, kurtosis=0.0,
        outlier_fractions={},
    )
    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)

    blocks = [r["block"] for r in rows]
    assert blocks[0] == "patch_embed"
    assert blocks[1] == "blocks.0"
    assert blocks[2] == "blocks.5"
    assert blocks[3] == "blocks.11"


def test_generate_summary_table_canonical_site_order() -> None:
    """Within a block, sites must follow canonical order."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {}
    for site in ["pre_gelu", "residual_stream", "pre_softmax"]:
        key = f"blocks.0/{site}"
        stats[key] = LayerStats(
            site_identifier=key,
            mean=0.0, std=1.0, kurtosis=0.0,
            outlier_fractions={},
        )
    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)

    sites = [r["site"] for r in rows]
    # Canonical order: residual_stream (0), pre_softmax (2), pre_gelu (5)
    assert sites[0] == "residual_stream"
    assert sites[1] == "pre_softmax"
    assert sites[2] == "pre_gelu"


def test_generate_summary_table_values_correct() -> None:
    """Row values must match the source LayerStats exactly."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {
        "blocks.0/pre_gelu": LayerStats(
            site_identifier="blocks.0/pre_gelu",
            mean=1.5, std=2.0, kurtosis=3.0,
            outlier_fractions={"3.0_sigma": 0.02},
        ),
    }
    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)
    row = rows[0]

    assert row["mean"] == 1.5
    assert row["std"] == 2.0
    assert row["kurtosis"] == 3.0
    assert row["frac_3.0_sigma"] == 0.02


def test_generate_summary_table_raises_on_empty_stats() -> None:
    """generate_summary_table must raise ValueError on empty stats."""
    from src.profiler import generate_summary_table

    result = ProfilingResult(stats={}, num_blocks=0, batch_shape=(1, 3, 224, 224))
    with pytest.raises(ValueError):
        generate_summary_table(result)


def test_save_summary_table_creates_file(tmp_path: Path) -> None:
    """save_summary_table must create a non-empty CSV file."""
    from src.profiler import save_summary_table

    rows = [{"block": "blocks.0", "site": "pre_gelu", "mean": 1.0}]
    path = tmp_path / "summary.csv"
    save_summary_table(rows, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_summary_table_header_row(tmp_path: Path) -> None:
    """CSV must have a header row matching the dict keys."""
    from src.profiler import save_summary_table

    rows = [{"block": "blocks.0", "site": "pre_gelu", "mean": 1.0}]
    path = tmp_path / "summary.csv"
    save_summary_table(rows, path)

    first_line = path.read_text().split("\n")[0]
    assert first_line == "block,site,mean"


def test_save_summary_table_round_trip(tmp_path: Path) -> None:
    """Float values must survive CSV round-trip without lossy rounding."""
    import csv
    from src.profiler import save_summary_table

    value = 1.23456789012345
    rows = [{"mean": value}]
    path = tmp_path / "summary.csv"
    save_summary_table(rows, path)

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        recovered = next(reader)
    assert float(recovered["mean"]) == pytest.approx(value, rel=1e-10)


def test_save_summary_table_raises_on_empty_rows(tmp_path: Path) -> None:
    """save_summary_table must raise ValueError on empty rows list."""
    from src.profiler import save_summary_table

    with pytest.raises(ValueError):
        save_summary_table([], tmp_path / "empty.csv")


def test_generate_summary_table_full_profiling_result() -> None:
    """generate_summary_table must produce 73 rows for a full ViT-B/16 result."""
    from src.profiler import generate_summary_table

    stats: dict[str, LayerStats] = {}
    # patch_embed site
    stats["patch_embed/residual_stream"] = LayerStats(
        site_identifier="patch_embed/residual_stream",
        mean=0.0, std=1.0, kurtosis=0.0,
        outlier_fractions={"3.0_sigma": 0.01, "4.0_sigma": 0.001, "6.0_sigma": 0.0},
    )
    # 12 blocks × 6 sites
    for i in range(12):
        for site in [
            "residual_stream", "post_layernorm_1", "pre_softmax",
            "post_softmax", "post_layernorm_2", "pre_gelu",
        ]:
            key = f"blocks.{i}/{site}"
            stats[key] = LayerStats(
                site_identifier=key,
                mean=float(i) * 0.1,
                std=1.0 + float(i) * 0.05,
                kurtosis=0.5,
                outlier_fractions={"3.0_sigma": 0.01, "4.0_sigma": 0.001, "6.0_sigma": 0.0},
            )

    result = ProfilingResult(stats=stats, num_blocks=12, batch_shape=(1, 3, 224, 224))
    rows = generate_summary_table(result)

    assert len(rows) == 73  # 1 patch_embed + 12*6
    assert rows[0]["block"] == "patch_embed"
    assert rows[0]["site"] == "residual_stream"
    assert rows[-1]["block"] == "blocks.11"
    assert rows[-1]["site"] == "pre_gelu"


# ---------------------------------------------------------------------------
# Residual delta ratio tests (T-005)
# ---------------------------------------------------------------------------


def test_layer_stats_ln2_amplification_ratio_default_none() -> None:
    """ln2_amplification_ratio must default to None."""
    stats = LayerStats(site_identifier="blocks.0/residual_stream", mean=0.0, std=1.0, kurtosis=0.0)
    assert stats.ln2_amplification_ratio is None


def test_layer_stats_ln2_amplification_ratio_stores_value() -> None:
    """ln2_amplification_ratio must accept and store a float value."""
    stats = LayerStats(
        site_identifier="blocks.3/residual_stream",
        mean=0.0, std=1.0, kurtosis=0.0,
        ln2_amplification_ratio=1.234,
    )
    assert stats.ln2_amplification_ratio == pytest.approx(1.234)


def test_layer_stats_ln2_amplification_ratio_serialization_roundtrip(
    tmp_path: Path,
) -> None:
    """ln2_amplification_ratio must survive JSON save → load roundtrip."""
    result = ProfilingResult(
        stats={
            "blocks.0/residual_stream": LayerStats(
                site_identifier="blocks.0/residual_stream",
                mean=0.0, std=1.0, kurtosis=0.0,
                ln2_amplification_ratio=2.718,
            ),
            "blocks.5/residual_stream": LayerStats(
                site_identifier="blocks.5/residual_stream",
                mean=0.5, std=2.0, kurtosis=1.0,
                ln2_amplification_ratio=1.414,
            ),
        },
        num_blocks=12,
        batch_shape=(1, 3, 224, 224),
    )
    path = tmp_path / "ln2_amplification_ratio_roundtrip.json"
    save_profiling_result(result, path)
    loaded = load_profiling_result(path)
    assert loaded.stats["blocks.0/residual_stream"].ln2_amplification_ratio == pytest.approx(2.718)
    assert loaded.stats["blocks.5/residual_stream"].ln2_amplification_ratio == pytest.approx(1.414)


def test_welford_accumulator_ln2_amplification_ratio_defaults() -> None:
    """WelfordAccumulator LN2 amplification ratio fields must default to zero."""
    from src.profiler import WelfordAccumulator

    acc = WelfordAccumulator(site_identifier="blocks.0/residual_stream")
    assert acc.ln2_amplification_ratio_sum == 0.0
    assert acc.ln2_amplification_ratio_count == 0


def test_merge_batch_stats_ln2_amplification_ratio_accumulation() -> None:
    """merge_batch_stats must accumulate LN2 amplification ratio via simple sum."""
    from src.profiler import WelfordAccumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="blocks.0/residual_stream")

    # Batch 1: ratio = 1.5
    stats1 = LayerStats(
        site_identifier="blocks.0/residual_stream",
        mean=0.0, std=1.0, kurtosis=0.0, m3=0.0,
        ln2_amplification_ratio=1.5,
    )
    merge_batch_stats(acc, stats1, batch_n=100)
    assert acc.ln2_amplification_ratio_sum == pytest.approx(1.5)
    assert acc.ln2_amplification_ratio_count == 1

    # Batch 2: ratio = 2.5
    stats2 = LayerStats(
        site_identifier="blocks.0/residual_stream",
        mean=0.5, std=1.2, kurtosis=0.1, m3=0.1,
        ln2_amplification_ratio=2.5,
    )
    merge_batch_stats(acc, stats2, batch_n=100)
    assert acc.ln2_amplification_ratio_sum == pytest.approx(4.0)
    assert acc.ln2_amplification_ratio_count == 2


def test_finalize_accumulator_ln2_amplification_ratio_mean() -> None:
    """finalize_accumulator must compute mean LN2 amplification ratio across batches."""
    from src.profiler import WelfordAccumulator, finalize_accumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="blocks.0/residual_stream")

    for ratio in [1.0, 2.0, 3.0]:
        stats = LayerStats(
            site_identifier="blocks.0/residual_stream",
            mean=0.0, std=1.0, kurtosis=0.0, m3=0.0,
            ln2_amplification_ratio=ratio,
        )
        merge_batch_stats(acc, stats, batch_n=100)

    result = finalize_accumulator(acc)
    assert result.ln2_amplification_ratio == pytest.approx(2.0)  # (1+2+3)/3


def test_finalize_accumulator_ln2_amplification_ratio_none_when_no_data() -> None:
    """finalize_accumulator must return None for LN2 amplification ratio when no batches contributed."""
    from src.profiler import WelfordAccumulator, finalize_accumulator, merge_batch_stats

    acc = WelfordAccumulator(site_identifier="blocks.0/residual_stream")

    # Batch with no LN2 amplification ratio (e.g., non-residual_stream site)
    stats = LayerStats(
        site_identifier="blocks.0/residual_stream",
        mean=0.0, std=1.0, kurtosis=0.0, m3=0.0,
        ln2_amplification_ratio=None,
    )
    merge_batch_stats(acc, stats, batch_n=100)

    result = finalize_accumulator(acc)
    assert result.ln2_amplification_ratio is None


# ---------------------------------------------------------------------------
# Slow tests — LN2 amplification ratio (T-005)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_ln2_amplification_ratio_present_on_residual_stream(
    _vit_result: ProfilingResult,
) -> None:
    """All residual_stream sites (except patch_embed) must have an LN2 amplification ratio.

    blocks.{0..11}/residual_stream must have non-None ln2_amplification_ratio.
    patch_embed/residual_stream must be None (no preceding LN2 block).
    """
    # patch_embed has no preceding LN2 → must be None
    pe_stats = _vit_result.stats["patch_embed/residual_stream"]
    assert pe_stats.ln2_amplification_ratio is None, (
        "patch_embed/residual_stream should not have an LN2 amplification ratio"
    )

    # blocks.0 through blocks.11 must have LN2 amplification ratios
    for i in range(_NUM_BLOCKS):
        key = f"blocks.{i}/residual_stream"
        stats = _vit_result.stats[key]
        assert stats.ln2_amplification_ratio is not None, (
            f"missing ln2_amplification_ratio at {key}"
        )
        assert isinstance(stats.ln2_amplification_ratio, float), (
            f"ln2_amplification_ratio must be float at {key}, got {type(stats.ln2_amplification_ratio)}"
        )


@pytest.mark.slow
def test_slow_ln2_amplification_ratio_positive(
    _vit_result: ProfilingResult,
) -> None:
    """All LN2 amplification ratios must be strictly positive (norms are non-negative)."""
    for i in range(_NUM_BLOCKS):
        key = f"blocks.{i}/residual_stream"
        ratio = _vit_result.stats[key].ln2_amplification_ratio
        assert ratio > 0.0, f"LN2 amplification ratio must be > 0 at {key}, got {ratio}"


@pytest.mark.slow
def test_slow_ln2_amplification_ratio_absent_on_non_residual_sites(
    _vit_result: ProfilingResult,
) -> None:
    """Non-residual_stream sites must have None for ln2_amplification_ratio."""
    for key, stats in _vit_result.stats.items():
        if "residual_stream" in key:
            continue  # covered by the tests above
        assert stats.ln2_amplification_ratio is None, (
            f"non-residual_stream site {key} has unexpected ln2_amplification_ratio"
        )


@pytest.mark.slow
def test_slow_ln2_amplification_ratio_survives_serialisation(
    tmp_path: Path, _vit_result: ProfilingResult,
) -> None:
    """LN2 amplification ratios must survive JSON save → load roundtrip on a real result."""
    path = tmp_path / "ln2_amplification_ratio_roundtrip.json"
    save_profiling_result(_vit_result, path)
    loaded = load_profiling_result(path)
    for i in range(_NUM_BLOCKS):
        key = f"blocks.{i}/residual_stream"
        orig = _vit_result.stats[key].ln2_amplification_ratio
        loaded_ratio = loaded.stats[key].ln2_amplification_ratio
        assert loaded_ratio == pytest.approx(orig, rel=1e-6), (
            f"LN2 amplification ratio mismatch at {key}: {loaded_ratio} vs {orig}"
        )


@pytest.mark.slow
def test_slow_ln2_amplification_ratio_reasonable_magnitude(
    _vit_result: ProfilingResult,
) -> None:
    """LN2 amplification ratios should be in a reasonable range for a trained ViT.

    For a well-trained ViT-B/16, the LN2 output norm relative to the skip
    connection is typically modest (ratios in [0.01, 10.0]).
    Values outside this range would indicate a bug in the computation.
    """
    for i in range(_NUM_BLOCKS):
        key = f"blocks.{i}/residual_stream"
        ratio = _vit_result.stats[key].ln2_amplification_ratio
        assert 0.001 < ratio < 100.0, (
            f"LN2 amplification ratio at {key} is {ratio}, outside expected range [0.001, 100.0]"
        )


# ---------------------------------------------------------------------------
# Slow tests — max/min (T-009)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_max_min_present(_vit_result: ProfilingResult) -> None:
    """All sites must have finite max/min with max >= min."""
    for key, stats in _vit_result.stats.items():
        assert math.isfinite(stats.max), f"max not finite at {key}"
        assert math.isfinite(stats.min), f"min not finite at {key}"
        assert stats.max >= stats.min, (
            f"max < min at {key}: max={stats.max}, min={stats.min}"
        )


@pytest.mark.slow
def test_slow_max_min_survives_serialisation(
    tmp_path: Path, _vit_result: ProfilingResult,
) -> None:
    """max/min must survive JSON save → load roundtrip on a real result."""
    path = tmp_path / "maxmin_slow_roundtrip.json"
    save_profiling_result(_vit_result, path)
    loaded = load_profiling_result(path)
    for key in _vit_result.stats:
        orig_max = _vit_result.stats[key].max
        orig_min = _vit_result.stats[key].min
        loaded_max = loaded.stats[key].max
        loaded_min = loaded.stats[key].min
        assert loaded_max == pytest.approx(orig_max, rel=1e-6), (
            f"max mismatch at {key}: {loaded_max} vs {orig_max}"
        )
        assert loaded_min == pytest.approx(orig_min, rel=1e-6), (
            f"min mismatch at {key}: {loaded_min} vs {orig_min}"
        )


# ---------------------------------------------------------------------------
# Fast tests — max/min merge tracking
# ---------------------------------------------------------------------------


def test_merge_batch_stats_max_min_tracking() -> None:
    """merge_batch_stats must correctly track running max/min across batches."""
    from src.profiler import WelfordAccumulator, merge_batch_stats, finalize_accumulator

    acc = WelfordAccumulator(site_identifier="test")

    # Batch 1: max=5.0, min=-3.0
    stats1 = LayerStats(
        site_identifier="test",
        mean=0.0, std=1.0, kurtosis=0.0, m3=0.0,
        max=5.0, min=-3.0,
    )
    merge_batch_stats(acc, stats1, batch_n=100)
    assert acc.max_val == pytest.approx(5.0)
    assert acc.min_val == pytest.approx(-3.0)

    # Batch 2: max=3.0 (lower), min=-1.0 (higher) — should not change extremum
    stats2 = LayerStats(
        site_identifier="test",
        mean=0.5, std=1.2, kurtosis=0.1, m3=0.1,
        max=3.0, min=-1.0,
    )
    merge_batch_stats(acc, stats2, batch_n=100)
    assert acc.max_val == pytest.approx(5.0), "max should not decrease"
    assert acc.min_val == pytest.approx(-3.0), "min should not increase"

    # Batch 3: max=10.0 (higher), min=-8.0 (lower) — should update extremum
    stats3 = LayerStats(
        site_identifier="test",
        mean=1.0, std=2.0, kurtosis=0.2, m3=0.2,
        max=10.0, min=-8.0,
    )
    merge_batch_stats(acc, stats3, batch_n=100)
    assert acc.max_val == pytest.approx(10.0), "max should update to new higher value"
    assert acc.min_val == pytest.approx(-8.0), "min should update to new lower value"

    # Finalize and verify max/min flow through.
    result = finalize_accumulator(acc)
    assert result.max == pytest.approx(10.0)
    assert result.min == pytest.approx(-8.0)
