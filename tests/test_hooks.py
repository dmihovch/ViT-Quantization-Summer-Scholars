"""
test_hooks.py
=============

Tests for the measurement core: the on-the-fly outlier math, the activation
extraction helper, and the hook/collector wiring.

These use small hand-built tensors with known answers, so every metric can be
asserted exactly - no model download required.
"""

import json

import pytest
import torch
from torch import Tensor

from src.hooks import (
    LayerActivationStats,
    OutlierStatsCollector,
    extract_activation_tensor,
    make_outlier_hook,
)
from src.model_utils import LayerType

# --- the three metrics, on a tensor with known answers -----------------------


def test_max_magnitude_picks_the_extreme_value(synthetic_activations: Tensor) -> None:
    stats = LayerActivationStats("demo", LayerType.FEEDFORWARD)
    stats.update(synthetic_activations)
    assert stats.finalize().max_magnitude == 100.0


def test_fixed_outlier_density_is_exact(synthetic_activations: Tensor) -> None:
    # 8 persistent-channel outliers + 1 spike = 9, out of 2*4*5 = 40 values.
    stats = LayerActivationStats("demo", LayerType.FEEDFORWARD)
    stats.update(synthetic_activations)
    summary = stats.finalize()
    assert summary.total_values_seen == 40
    assert summary.fixed_outlier_density == pytest.approx(9 / 40)


def test_channel_persistence_is_higher_when_outliers_concentrate() -> None:
    """
    Concentrating all outliers in one channel must yield a HIGHER persistence
    variance than spreading the same number of outliers across many channels.
    This is the core signal Experiment 1 exists to measure.
    """
    concentrated = torch.zeros(2, 4, 5)
    concentrated[..., 2] = 10.0  # every outlier lands in channel 2

    scattered = torch.zeros(2, 4, 5)
    scattered[0, 0, 0] = 10.0  # one outlier in each of four different channels
    scattered[0, 1, 1] = 10.0
    scattered[0, 2, 2] = 10.0
    scattered[0, 3, 3] = 10.0

    concentrated_stats = LayerActivationStats("c", LayerType.FEEDFORWARD)
    concentrated_stats.update(concentrated)
    scattered_stats = LayerActivationStats("s", LayerType.FEEDFORWARD)
    scattered_stats.update(scattered)

    concentrated_variance = concentrated_stats.finalize().channel_persistence_variance
    scattered_variance = scattered_stats.finalize().channel_persistence_variance
    assert concentrated_variance > scattered_variance


def test_max_magnitude_is_a_running_maximum_across_batches() -> None:
    stats = LayerActivationStats("demo", LayerType.OTHER)
    stats.update(torch.full((1, 2, 3), 5.0))
    stats.update(torch.full((1, 2, 3), 20.0))  # the peak arrives in batch 2
    stats.update(torch.full((1, 2, 3), 8.0))
    assert stats.finalize().max_magnitude == 20.0


def test_finalize_with_no_data_returns_zeros() -> None:
    """A layer that never fired must not crash with a divide-by-zero."""
    summary = LayerActivationStats("demo", LayerType.OTHER).finalize()
    assert summary.total_values_seen == 0
    assert summary.max_magnitude == 0.0
    assert summary.fixed_outlier_density == 0.0
    assert summary.channel_persistence_variance == 0.0


# --- activation extraction (handles Tensor vs attention's tuple output) ------


def test_extract_returns_a_plain_tensor_unchanged() -> None:
    tensor = torch.randn(2, 3)
    assert extract_activation_tensor(tensor) is tensor


def test_extract_pulls_the_first_tensor_from_an_attention_tuple() -> None:
    # nn.MultiheadAttention returns (attn_output, attn_weights); weights is None.
    attention_output = torch.randn(2, 3)
    assert extract_activation_tensor((attention_output, None)) is attention_output


def test_extract_returns_none_when_there_is_no_tensor() -> None:
    assert extract_activation_tensor(None) is None
    assert extract_activation_tensor("not a tensor") is None


# --- collector + hook wiring -------------------------------------------------


def test_hook_feeds_activations_into_the_collector() -> None:
    collector = OutlierStatsCollector()
    collector.register_layer("demo", LayerType.FEEDFORWARD)

    linear = torch.nn.Linear(5, 5)
    handle = linear.register_forward_hook(make_outlier_hook(collector, "demo"))
    _ = linear(torch.randn(2, 4, 5))  # firing the layer triggers the hook
    handle.remove()

    summaries = collector.build_summaries()
    assert len(summaries) == 1
    assert summaries[0].layer_name == "demo"
    assert summaries[0].total_values_seen == 40  # 2 * 4 * 5


def test_summary_to_dict_is_json_serializable(synthetic_activations: Tensor) -> None:
    stats = LayerActivationStats("demo", LayerType.ATTENTION)
    stats.update(synthetic_activations)
    payload = stats.finalize().to_dict()

    # The layer type must be a plain string, not an enum, for json.dump.
    assert payload["layer_type"] == "Attention_QKV"

    # The whole payload round-trips through JSON without error.
    restored = json.loads(json.dumps(payload))
    assert restored["layer_name"] == "demo"
