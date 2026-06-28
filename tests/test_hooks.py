"""
test_hooks.py
=============

Tests for the two-pass measurement core: the exact global mean/std math (pass 1),
the threshold-based outlier and routing-fraction math (pass 2), the activation
extraction helper, and the hook/collector wiring.

These use small hand-built tensors with known answers, so every metric can be
asserted exactly - no model download required.
"""

import json
import math

import pytest
import torch
from torch import Tensor

from src.hooks import (
    LayerMomentAccumulator,
    LayerOutlierAccumulator,
    LayerThreshold,
    MomentCollector,
    OutlierStatsCollector,
    extract_activation_tensor,
    make_measurement_hook,
)
from src.model_utils import LayerType

# A threshold that never flags anything (cutoff = +inf via huge std) so tests
# that only care about the FIXED 6.0 threshold are unaffected by the statistical
# path. Individual statistical tests build their own thresholds explicitly.
NEUTRAL_THRESHOLD = LayerThreshold(global_mean=0.0, global_std=1.0e9)


# --- Pass 1: exact global mean and standard deviation ------------------------


def test_global_mean_and_std_are_exact() -> None:
    # Values 1, 2, 3, 4: mean = 2.5; population variance = 1.25; std = sqrt(1.25).
    stats = LayerMomentAccumulator("demo", LayerType.FEEDFORWARD)
    stats.update(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    threshold = stats.finalize()
    assert threshold.global_mean == pytest.approx(2.5)
    assert threshold.global_std == pytest.approx(math.sqrt(1.25))
    assert threshold.statistical_cutoff == pytest.approx(3.0 * math.sqrt(1.25))


def test_moment_merge_is_order_and_batching_invariant() -> None:
    """
    The Chan/Welford merge must give the SAME global mean/std whether the data
    arrives in one batch or split across several. This is what makes the global
    statistics exact regardless of batch size.
    """
    all_at_once = LayerMomentAccumulator("a", LayerType.FEEDFORWARD)
    all_at_once.update(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))

    in_pieces = LayerMomentAccumulator("b", LayerType.FEEDFORWARD)
    in_pieces.update(torch.tensor([[1.0, 2.0]]))
    in_pieces.update(torch.tensor([[3.0, 4.0, 5.0]]))
    in_pieces.update(torch.tensor([[6.0]]))

    single = all_at_once.finalize()
    split = in_pieces.finalize()
    assert single.global_mean == pytest.approx(split.global_mean)
    assert single.global_std == pytest.approx(split.global_std)


def test_moments_with_no_data_are_zero() -> None:
    threshold = LayerMomentAccumulator("demo", LayerType.OTHER).finalize()
    assert threshold.global_mean == 0.0
    assert threshold.global_std == 0.0


# --- Pass 2: max magnitude, densities, routing fractions ---------------------


def test_max_magnitude_picks_the_extreme_value(synthetic_activations: Tensor) -> None:
    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD)
    stats.update(synthetic_activations)
    assert stats.finalize().max_magnitude == 100.0


def test_max_magnitude_is_a_running_maximum_across_batches() -> None:
    stats = LayerOutlierAccumulator("demo", LayerType.OTHER, NEUTRAL_THRESHOLD)
    stats.update(torch.full((1, 2, 3), 5.0))
    stats.update(torch.full((1, 2, 3), 20.0))  # the peak arrives in batch 2
    stats.update(torch.full((1, 2, 3), 8.0))
    assert stats.finalize().max_magnitude == 20.0


def test_value_outlier_density_fixed_is_exact(synthetic_activations: Tensor) -> None:
    # The per-value (unstructured) density: 8 persistent-channel outliers + 1
    # spike = 9, out of 2*4*5 = 40 values.
    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD)
    stats.update(synthetic_activations)
    summary = stats.finalize()
    assert summary.total_values_seen == 40
    assert summary.total_tokens_seen == 8  # 2 * 4 tokens
    assert summary.value_outlier_density_fixed == pytest.approx(9 / 40)


def test_routing_fraction_counts_only_persistent_columns() -> None:
    """
    The per-column routing fraction must count only columns that are outliers in
    at least 25% of tokens (LLM.int8's participation criterion), NOT any column
    with a single stray spike. This is what keeps the metric from saturating.
    """
    activations = torch.zeros(8, 5)  # [8 tokens, 5 feature columns]
    activations[:, 0] = 10.0  # column 0: outlier in 8/8 tokens (100%)  -> routed
    activations[0:2, 1] = 10.0  # column 1: outlier in 2/8 tokens (25%)  -> routed
    activations[0, 2] = 10.0  # column 2: outlier in 1/8 tokens (12.5%) -> NOT routed

    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD)
    stats.update(activations)
    result = stats.finalize()

    # Two of five columns clear the 25% participation bar -> 0.4. A naive
    # "any token over threshold" rule would wrongly flag column 2 too (0.6).
    assert result.routing_fraction_fixed == pytest.approx(2 / 5)
    # The unstructured baseline counts every outlier value: (8 + 2 + 1) / 40.
    assert result.value_outlier_density_fixed == pytest.approx(11 / 40)


def test_routing_fraction_separates_persistent_from_scattered_outliers() -> None:
    """
    The headline comparison: with the SAME per-value outlier density, outliers
    concentrated in one persistent column are routable (non-zero routing
    fraction), while the same number scattered one-per-column are not. The gap
    between value density and routing fraction is exactly the signal that tells
    us whether LLM.int8's structured routing applies to a layer.
    """
    concentrated = torch.zeros(8, 8)
    concentrated[:, 0] = 10.0  # all 8 outliers in column 0 (8/8 of its tokens)

    scattered = torch.zeros(8, 8)
    for token_index in range(8):
        scattered[token_index, token_index] = 10.0  # 1 outlier per column (1/8)

    concentrated_stats = LayerOutlierAccumulator(
        "c", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD
    )
    concentrated_stats.update(concentrated)
    concentrated_summary = concentrated_stats.finalize()

    scattered_stats = LayerOutlierAccumulator(
        "s", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD
    )
    scattered_stats.update(scattered)
    scattered_summary = scattered_stats.finalize()

    # Identical unstructured density (8 outliers out of 64 values each).
    assert concentrated_summary.value_outlier_density_fixed == pytest.approx(8 / 64)
    assert scattered_summary.value_outlier_density_fixed == pytest.approx(8 / 64)

    # But only the concentrated case yields a routable (persistent) column.
    assert concentrated_summary.routing_fraction_fixed == pytest.approx(1 / 8)
    assert scattered_summary.routing_fraction_fixed == pytest.approx(0.0)


def test_statistical_threshold_uses_the_frozen_global_cutoff() -> None:
    """
    Pass 2 must threshold against the EXACT global mean/std from pass 1, not
    per-batch statistics. With mean=0, std=1 the cutoff is 3.0, so only the
    values above 3.0 count - independent of this batch's own spread.
    """
    threshold = LayerThreshold(global_mean=0.0, global_std=1.0)  # cutoff = 3.0
    activations = torch.zeros(8, 4)  # [8 tokens, 4 columns]
    activations[:, 0] = 5.0  # column 0: |5| > 3 in 8/8 tokens -> outlier column
    activations[0, 1] = 5.0  # column 1: |5| > 3 in 1/8 tokens (12.5%) -> not
    activations[:, 2] = 2.0  # column 2: |2| < 3 -> never an outlier

    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, threshold)
    stats.update(activations)
    summary = stats.finalize()

    # Outlier values above the 3.0 cutoff: 8 (col 0) + 1 (col 1) = 9 of 32.
    assert summary.value_outlier_density_statistical == pytest.approx(9 / 32)
    # Only column 0 clears the 25% participation bar -> 1/4.
    assert summary.routing_fraction_statistical == pytest.approx(1 / 4)
    # The exact threshold is surfaced for auditing.
    assert summary.global_std == pytest.approx(1.0)


def test_channel_persistence_is_higher_when_outliers_concentrate() -> None:
    """
    Concentrating all outliers in one channel must yield a HIGHER persistence
    variance than spreading the same number of outliers across many channels.
    """
    concentrated = torch.zeros(2, 4, 5)
    concentrated[..., 2] = 10.0  # every outlier lands in channel 2

    scattered = torch.zeros(2, 4, 5)
    scattered[0, 0, 0] = 10.0  # one outlier in each of four different channels
    scattered[0, 1, 1] = 10.0
    scattered[0, 2, 2] = 10.0
    scattered[0, 3, 3] = 10.0

    concentrated_stats = LayerOutlierAccumulator(
        "c", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD
    )
    concentrated_stats.update(concentrated)
    scattered_stats = LayerOutlierAccumulator(
        "s", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD
    )
    scattered_stats.update(scattered)

    concentrated_variance = concentrated_stats.finalize().channel_persistence_variance
    scattered_variance = scattered_stats.finalize().channel_persistence_variance
    assert concentrated_variance > scattered_variance


def test_finalize_with_no_data_returns_zeros() -> None:
    """A layer that never fired must not crash with a divide-by-zero."""
    summary = LayerOutlierAccumulator(
        "demo", LayerType.OTHER, NEUTRAL_THRESHOLD
    ).finalize()
    assert summary.total_values_seen == 0
    assert summary.total_tokens_seen == 0
    assert summary.max_magnitude == 0.0
    assert summary.routing_fraction_fixed == 0.0
    assert summary.routing_fraction_statistical == 0.0
    assert summary.value_outlier_density_fixed == 0.0
    assert summary.value_outlier_density_statistical == 0.0
    assert summary.channel_persistence_variance == 0.0


# --- activation extraction (pulls the input tensor from a pre-hook's args) ----


def test_extract_returns_a_plain_tensor_unchanged() -> None:
    tensor = torch.randn(2, 3)
    assert extract_activation_tensor(tensor) is tensor


def test_extract_pulls_the_first_tensor_from_a_tuple() -> None:
    # A forward pre-hook hands us the module's positional inputs as a tuple; for
    # an nn.Linear that is (input_activation,). We pull out that first tensor.
    input_activation = torch.randn(2, 3)
    assert extract_activation_tensor((input_activation, None)) is input_activation


def test_extract_returns_none_when_there_is_no_tensor() -> None:
    assert extract_activation_tensor(None) is None
    assert extract_activation_tensor("not a tensor") is None


# --- two-pass collector + hook wiring ----------------------------------------


def test_two_pass_collectors_feed_activations_through_hooks() -> None:
    """End-to-end wiring: pass 1 computes thresholds, pass 2 consumes them."""
    linear = torch.nn.Linear(5, 5)
    sample = torch.randn(2, 4, 5)  # [Batch, Sequence, Features] input

    # Pass 1: moments.
    moments = MomentCollector()
    moments.register_layer("demo", LayerType.FEEDFORWARD)
    handle = linear.register_forward_pre_hook(make_measurement_hook(moments, "demo"))
    _ = linear(sample)
    handle.remove()
    thresholds = moments.build_thresholds()
    # The threshold for our layer is retrievable.
    assert thresholds.for_layer("demo").global_std >= 0.0

    # Pass 2: outliers, seeded with the pass-1 thresholds.
    collector = OutlierStatsCollector(thresholds)
    collector.register_layer("demo", LayerType.FEEDFORWARD)
    handle = linear.register_forward_pre_hook(make_measurement_hook(collector, "demo"))
    _ = linear(sample)
    handle.remove()

    summaries = collector.build_summaries()
    assert len(summaries) == 1
    assert summaries[0].layer_name == "demo"
    assert summaries[0].total_values_seen == 40  # 2 * 4 * 5 input values


def test_summary_to_dict_is_json_serializable(synthetic_activations: Tensor) -> None:
    stats = LayerOutlierAccumulator("demo", LayerType.ATTENTION, NEUTRAL_THRESHOLD)
    stats.update(synthetic_activations)
    payload = stats.finalize().to_dict()

    # The layer type must be a plain string, not an enum, for json.dump.
    assert payload["layer_type"] == "Attention_QKV"
    # The headline routing-fraction metric is present in the serialized payload.
    assert "routing_fraction_fixed" in payload
    assert "value_outlier_density_statistical" in payload

    # The whole payload round-trips through JSON without error.
    restored = json.loads(json.dumps(payload))
    assert restored["layer_name"] == "demo"
