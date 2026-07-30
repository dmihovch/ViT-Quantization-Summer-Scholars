"""Tests for the outlier-zeroing ablation utilities in :mod:`src.ablation`."""

from __future__ import annotations

import torch
import pytest

from src.ablation import build_zeroing_hook, compute_pct_zeroed
from src.profiler import LayerStats


def test_compute_pct_zeroed_all_below_threshold(dummy_tensor: torch.Tensor) -> None:
    """When all absolute values are below the threshold, 0% should be zeroed.

    We pass a tensor filled entirely with 0.1 so every element satisfies
    |x| <= 1.0, and the expected result is exactly 0.0.
    """
    tensor = torch.full_like(dummy_tensor, 0.1)
    result = compute_pct_zeroed(tensor, threshold=1.0)
    assert result == pytest.approx(0.0)


def test_compute_pct_zeroed_all_above_threshold(dummy_tensor: torch.Tensor) -> None:
    """When all absolute values exceed the threshold, 100% should be zeroed.

    We fill the tensor with 10.0, which is well above the threshold of 1.0,
    so every element satisfies |x| > 1.0.
    """
    tensor = torch.full_like(dummy_tensor, 10.0)
    result = compute_pct_zeroed(tensor, threshold=1.0)
    assert result == pytest.approx(100.0)


def test_compute_pct_zeroed_mixed() -> None:
    """A hand-constructed tensor with a known fraction above the threshold.

    We build a 1-D tensor of 10 elements where exactly 3 have |x| > 2.0:
    values [0, 1, 2, 3, -3, 4, 0.5, 1.5, 2.0, -1].
    Elements > 2.0 in absolute value: 3, -3, 4  →  3 / 10 = 30%.
    Note: the threshold comparison is strict (|x| > threshold), so 2.0 itself
    is NOT counted.
    """
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, -3.0, 4.0, 0.5, 1.5, 2.0, -1.0])
    result = compute_pct_zeroed(tensor, threshold=2.0)
    assert result == pytest.approx(30.0)


def test_build_zeroing_hook_returns_callable(
    tiny_layer_stats: dict[str, LayerStats],
) -> None:
    """build_zeroing_hook must return a callable (forward pre-hook)."""
    layer_name = "blocks.0.mlp.act"
    stats = tiny_layer_stats[layer_name]
    hook = build_zeroing_hook(layer_name=layer_name, threshold=2.0, stats=stats)
    assert callable(hook)
