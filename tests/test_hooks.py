"""
test_hooks.py
=============

Tests for the two-pass measurement core: the exact per-channel mean/std math
(pass 1), the threshold-based outlier and routing-fraction math (pass 2), the
activation extraction helper, and the hook/collector wiring.

These use small hand-built tensors with known answers, so every metric can be
asserted exactly - no model download required.
"""

import json
import math

import pytest
import torch
from torch import Tensor

from src.hooks import (
    FIXED_PARTICIPATION_FRACTION,
    STATISTICAL_PARTICIPATION_FRACTION,
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
# We use 5 channels to match the synthetic_activations fixture.
NEUTRAL_THRESHOLD = LayerThreshold(
    channel_means=torch.zeros(5, dtype=torch.float64),
    channel_stds=torch.full((5,), 1.0e9, dtype=torch.float64),
)


# --- Pass 1: exact per-channel mean and standard deviation --------------------


def test_per_channel_mean_and_std_are_exact() -> None:
    """
    With two channels carrying different distributions, the per-channel
    statistics must capture each channel independently.
    Channel 0: [1, 2, 3, 4] -> mean=2.5, var=1.25, std=sqrt(1.25)
    Channel 1: [10, 20, 30, 40] -> mean=25.0, var=125.0, std=sqrt(125)
    """
    activations = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )  # [4 tokens, 2 features]
    stats = LayerMomentAccumulator("demo", LayerType.FEEDFORWARD)
    stats.update(activations)
    threshold = stats.finalize()

    # Per-channel means.
    assert threshold.channel_means[0].item() == pytest.approx(2.5)
    assert threshold.channel_means[1].item() == pytest.approx(25.0)

    # Per-channel stds.
    assert threshold.channel_stds[0].item() == pytest.approx(math.sqrt(1.25))
    assert threshold.channel_stds[1].item() == pytest.approx(math.sqrt(125.0))

    # Per-channel cutoffs.
    assert threshold.statistical_cutoff[0].item() == pytest.approx(
        3.0 * math.sqrt(1.25)
    )
    assert threshold.statistical_cutoff[1].item() == pytest.approx(
        3.0 * math.sqrt(125.0)
    )

    # Derived aggregates.
    assert threshold.global_mean == pytest.approx((2.5 + 25.0) / 2)
    assert threshold.global_std == pytest.approx(math.sqrt((1.25 + 125.0) / 2))


def test_moment_merge_is_order_and_batching_invariant() -> None:
    """
    The Chan/Welford merge must give the SAME per-channel mean/std whether the
    data arrives in one batch or split across several. This is what makes the
    statistics exact regardless of batch size.
    """
    all_at_once = LayerMomentAccumulator("a", LayerType.FEEDFORWARD)
    all_at_once.update(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    in_pieces = LayerMomentAccumulator("b", LayerType.FEEDFORWARD)
    in_pieces.update(torch.tensor([[1.0, 2.0]]))
    in_pieces.update(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))

    single = all_at_once.finalize()
    split = in_pieces.finalize()

    assert torch.allclose(single.channel_means, split.channel_means)
    assert torch.allclose(single.channel_stds, split.channel_stds)


def test_moments_with_no_data_are_zero() -> None:
    threshold = LayerMomentAccumulator("demo", LayerType.OTHER).finalize()
    assert threshold.channel_means.numel() == 1
    assert threshold.channel_stds.numel() == 1
    assert threshold.channel_means[0].item() == 0.0
    assert threshold.channel_stds[0].item() == 0.0


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
    The per-column routing fraction for the FIXED threshold must count only
    columns that are outliers in at least 25% of tokens (LLM.int8's actual
    participation criterion), NOT any column with a single stray spike.
    """
    activations = torch.zeros(8, 5)  # [8 tokens, 5 feature columns]
    activations[:, 0] = 10.0  # column 0: outlier in 8/8 tokens (100%)  -> routed
    activations[0:2, 1] = 10.0  # column 1: outlier in 2/8 tokens (25%)  -> routed
    activations[0, 2] = 10.0  # column 2: outlier in 1/8 tokens (12.5%) -> NOT routed

    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD)
    stats.update(activations)
    result = stats.finalize()

    # Two of five columns clear the 25% participation bar -> 0.4.
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


def test_statistical_threshold_uses_per_channel_cutoff() -> None:
    """
    Pass 2 must threshold against the EXACT per-channel statistics from pass 1.
    Two channels with different distributions get different cutoffs:
      Channel 0: mean=0, std=1  -> cutoff = 3.0
      Channel 1: mean=10, std=2 -> cutoff = 6.0
    A value of 5.0 in channel 0 is an outlier (|5-0| > 3.0).
    A value of 5.0 in channel 1 is NOT an outlier (|5-10| = 5 <= 6.0).

    The statistical routing fraction uses a 5% participation bar (calibrated
    to ViT's ~1% per-channel outlier density), not the 25% LLM.int8() bar.
    """
    threshold = LayerThreshold(
        channel_means=torch.tensor([0.0, 10.0], dtype=torch.float64),
        channel_stds=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    # 8 tokens, 2 channels.
    activations = torch.zeros(8, 2)
    activations[:, 0] = 5.0  # channel 0: |5-0|=5 > 3.0 -> outlier in all 8 tokens
    activations[:, 1] = 5.0  # channel 1: |5-10|=5 <= 6.0 -> NOT an outlier

    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, threshold)
    stats.update(activations)
    summary = stats.finalize()

    # Only channel 0 values (8 of 16) are outliers.
    assert summary.value_outlier_density_statistical == pytest.approx(8 / 16)
    # Channel 0 clears the 5% bar (8/8 tokens), channel 1 does not (0/8).
    assert summary.routing_fraction_statistical == pytest.approx(1 / 2)
    # Per-channel stats are surfaced for auditing.
    assert summary.channel_means == pytest.approx([0.0, 10.0])
    assert summary.channel_stds == pytest.approx([1.0, 2.0])


def test_statistical_threshold_differentiates_tight_vs_wide_channels() -> None:
    """
    A tight-variance channel and a wide-variance channel with the same mean
    should get different cutoffs. A value that is normal for the wide channel
    can be an extreme outlier for the tight channel.
    """
    threshold = LayerThreshold(
        channel_means=torch.tensor([0.0, 0.0], dtype=torch.float64),
        channel_stds=torch.tensor([0.5, 5.0], dtype=torch.float64),
    )
    # cutoffs: channel 0 = 1.5, channel 1 = 15.0
    activations = torch.zeros(8, 2)
    activations[:, 0] = 2.0  # channel 0: |2| > 1.5 -> outlier
    activations[:, 1] = 2.0  # channel 1: |2| <= 15.0 -> NOT an outlier

    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, threshold)
    stats.update(activations)
    summary = stats.finalize()

    # Only channel 0 values (8 of 16) are outliers.
    assert summary.value_outlier_density_statistical == pytest.approx(8 / 16)
    assert summary.routing_fraction_statistical == pytest.approx(1 / 2)


def test_statistical_routing_uses_5_percent_not_25_percent_bar() -> None:
    """
    The statistical routing fraction uses a 5% participation bar (calibrated to
    ViT's ~1% per-channel outlier density), NOT the 25% LLM.int8() bar used for
    the fixed threshold. A column with 10% participation should be flagged by
    the statistical threshold but NOT by the fixed threshold.

    With 100 tokens and 4 channels:
      - Channel 0: outlier in 10/100 tokens (10%) -> clears 5% bar, fails 25%
      - Channel 1: outlier in 30/100 tokens (30%) -> clears both bars
      - Channel 2: outlier in 3/100 tokens (3%)   -> fails both bars
      - Channel 3: outlier in 0/100 tokens (0%)   -> fails both bars
    """
    threshold = LayerThreshold(
        channel_means=torch.zeros(4, dtype=torch.float64),
        channel_stds=torch.ones(4, dtype=torch.float64),
    )
    activations = torch.zeros(100, 4)
    activations[:10, 0] = 10.0  # channel 0: 10% participation
    activations[:30, 1] = 10.0  # channel 1: 30% participation
    activations[:3, 2] = 10.0  # channel 2: 3% participation

    stats = LayerOutlierAccumulator("demo", LayerType.FEEDFORWARD, threshold)
    stats.update(activations)
    summary = stats.finalize()

    # Statistical routing (5% bar): channels 0 and 1 qualify -> 2/4 = 0.5.
    assert summary.routing_fraction_statistical == pytest.approx(2 / 4)
    # Fixed routing (25% bar): only channel 1 qualifies -> 1/4 = 0.25.
    assert summary.routing_fraction_fixed == pytest.approx(1 / 4)


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


def test_channel_persistence_is_invariant_to_batching() -> None:
    # Regression test for a fixed bug: channel persistence variance must be
    # computed once over the FULL run's cumulative per-column outlier counts,
    # not averaged across per-batch variances. Splitting the same data into
    # two update() calls must give the identical result as one update() call.
    full = torch.zeros(4, 8)
    full[:, 0] = 10.0
    full[0:2, 3] = 10.0

    single_batch_stats = LayerOutlierAccumulator(
        "single", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD
    )
    single_batch_stats.update(full)
    single_batch_variance = single_batch_stats.finalize().channel_persistence_variance

    two_batch_stats = LayerOutlierAccumulator(
        "two", LayerType.FEEDFORWARD, NEUTRAL_THRESHOLD
    )
    two_batch_stats.update(full[:2])
    two_batch_stats.update(full[2:])
    two_batch_variance = two_batch_stats.finalize().channel_persistence_variance

    assert two_batch_variance == pytest.approx(single_batch_variance)


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
    assert summary.channel_means == []
    assert summary.channel_stds == []


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
    # The threshold for our layer is retrievable and has per-channel tensors.
    layer_threshold = thresholds.for_layer("demo")
    assert layer_threshold.channel_means.numel() == 5
    assert layer_threshold.channel_stds.numel() == 5
    assert layer_threshold.global_std >= 0.0

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
    # Per-channel statistics are surfaced.
    assert len(summaries[0].channel_means) == 5
    assert len(summaries[0].channel_stds) == 5


def test_summary_to_dict_is_json_serializable(synthetic_activations: Tensor) -> None:
    stats = LayerOutlierAccumulator("demo", LayerType.ATTENTION, NEUTRAL_THRESHOLD)
    stats.update(synthetic_activations)
    payload = stats.finalize().to_dict()

    # The layer type must be a plain string, not an enum, for json.dump.
    assert payload["layer_type"] == "Attention_QKV"
    # The headline routing-fraction metric is present in the serialized payload.
    assert "routing_fraction_fixed" in payload
    assert "value_outlier_density_statistical" in payload
    # Per-channel statistics are present and serializable.
    assert "channel_means" in payload
    assert "channel_stds" in payload
    assert isinstance(payload["channel_means"], list)
    assert isinstance(payload["channel_stds"], list)

    # The whole payload round-trips through JSON without error.
    restored = json.loads(json.dumps(payload))
    assert restored["layer_name"] == "demo"
