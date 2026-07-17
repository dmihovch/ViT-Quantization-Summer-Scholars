"""Tests for the integer GELU LUT utilities in :mod:`src.integer_gelu`."""

from __future__ import annotations

import pytest

from src.integer_gelu import GELULut, build_lut, compare_lut_vs_fp32

_SCALE_IN: float = 0.1
_SCALE_OUT: float = 0.1


def test_build_lut_returns_256_entries() -> None:
    """build_lut must return a list with exactly 256 entries."""
    lut = build_lut(scale_in=_SCALE_IN, scale_out=_SCALE_OUT)
    assert len(lut) == 256


def test_build_lut_all_entries_in_int8_range() -> None:
    """Every entry produced by build_lut must lie within the INT8 range [-128, 127]."""
    lut = build_lut(scale_in=_SCALE_IN, scale_out=_SCALE_OUT)
    assert all(-128 <= value <= 127 for value in lut)


def test_compare_lut_vs_fp32_returns_expected_keys() -> None:
    """compare_lut_vs_fp32 must return a dict with the three expected error keys."""
    lut_values = build_lut(scale_in=_SCALE_IN, scale_out=_SCALE_OUT)
    gelu_lut = GELULut(
        layer_name="blocks.0.mlp.act",
        scale_in=_SCALE_IN,
        scale_out=_SCALE_OUT,
        lut=lut_values,
    )
    metrics = compare_lut_vs_fp32(gelu_lut, scale_in=_SCALE_IN)
    assert set(metrics.keys()) == {"max_abs_error", "mean_abs_error", "rmse"}


def test_compare_lut_vs_fp32_errors_are_nonnegative() -> None:
    """All error metrics returned by compare_lut_vs_fp32 must be >= 0."""
    lut_values = build_lut(scale_in=_SCALE_IN, scale_out=_SCALE_OUT)
    gelu_lut = GELULut(
        layer_name="blocks.0.mlp.act",
        scale_in=_SCALE_IN,
        scale_out=_SCALE_OUT,
        lut=lut_values,
    )
    metrics = compare_lut_vs_fp32(gelu_lut, scale_in=_SCALE_IN)
    assert all(value >= 0.0 for value in metrics.values())
