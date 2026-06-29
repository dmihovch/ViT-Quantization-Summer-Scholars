"""
hooks.py
========

The measurement machinery for Experiment 1, implemented as a rigorous TWO-PASS
algorithm so the statistical (3-sigma) threshold uses EXACT global statistics
rather than a per-batch approximation.

Why two passes?
---------------
The fixed `|x| > 6.0` threshold needs no data statistics, so it could be applied
in a single streaming pass. The statistical threshold, however, is defined
relative to "the layer's mean" and "the layer's standard deviation" - quantities
that are only known once every image has been seen. To threshold a value exactly
we must therefore know the global mean and std *before* we start counting
outliers. Hence:

  * PASS 1 (`LayerMomentAccumulator`) - stream every image and accumulate, per
    layer, the exact global mean and standard deviation of its input activation.
    We use the Chan/Welford parallel merge in float64, which is numerically
    stable (no catastrophic cancellation) and exact up to floating-point round-
    off. The output is one `LayerThreshold` per layer.

  * PASS 2 (`LayerOutlierAccumulator`) - stream every image again and, using the
    FROZEN per-layer mean/std from pass 1, count outliers and routing fractions
    for both thresholds. Because the data loader is deterministic (no shuffling),
    both passes see byte-identical inputs, so the global statistics computed in
    pass 1 apply exactly in pass 2.

Speed is sacrificed (we read the data twice) in favour of mathematical
exactness, which is the right trade-off for the thesis numbers.

Why inputs, not outputs?
------------------------
For a matmul `Y = X @ W.T`, `LLM.int8()` inspects the INPUT activation `X` and
decides, per feature column of `X`, whether that column is an outlier column
(routed to FP16) or not (computed in INT8). So `X` - the post-LayerNorm hidden
state entering the matmul - is the exact tensor at which the routing decision is
made. We hook inputs (via forward PRE-hooks) so our statistics characterize
activations at precisely that decision point.

Structured (per-column) vs unstructured (per-value) outliers
------------------------------------------------------------
LLM.int8() is a STRUCTURED scheme: it routes ENTIRE outlier feature columns to
FP16, never arbitrary scattered scalars. So the routing cost is the fraction of
input feature COLUMNS that must go to FP16. We report both the per-column routing
fraction (PRIMARY) and the per-value outlier density (BASELINE); the gap between
them shows how much the whole-column constraint over-routes on each layer.

The hard memory rule (8 GB VRAM budget)
---------------------------------------
Every accumulator reduces its tensor to scalar counters (and, in pass 2, one
fixed-size per-column counter of length = feature width) *on the fly*, then lets
the raw activation tensor be discarded. No raw activations are ever stored.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
import torch.nn as nn
from torch import Tensor

from src.model_utils import LayerType

# --- what counts as an "outlier" ---------------------------------------------

# The fixed magnitude threshold used by the LLM.int8() paper. A value whose
# absolute size exceeds this is a candidate outlier.
FIXED_OUTLIER_THRESHOLD: float = 6.0

# The statistical threshold: a value is an outlier if it sits more than this
# many standard deviations away from its layer's (global) mean activation.
STD_THRESHOLD_MULTIPLIER: float = 3.0

# An input feature column counts as an "outlier column" (one LLM.int8 would route
# to FP16) only if it carries a threshold-exceeding value in at least this
# fraction of tokens. This mirrors LLM.int8()'s own criterion - outlier feature
# dimensions are those whose magnitude exceeds the threshold in >= 25% of
# sequence positions. Requiring PERSISTENCE (not a single stray spike) is what
# keeps the routing fraction meaningful and non-saturating.
OUTLIER_COLUMN_PARTICIPATION_FRACTION: float = 0.25


# =============================================================================
# PASS 1: exact global mean and standard deviation
# =============================================================================


@dataclass(frozen=True)
class LayerThreshold:
    """
    The exact global mean and standard deviation of one layer's input
    activation, computed in pass 1. These define that layer's statistical
    (3-sigma) outlier cutoff, frozen for use in pass 2.
    """

    global_mean: float
    global_std: float

    @property
    def statistical_cutoff(self) -> float:
        """A value `x` is a statistical outlier when `|x - global_mean|` exceeds
        this (i.e. it lies more than 3 standard deviations from the mean)."""
        return STD_THRESHOLD_MULTIPLIER * self.global_std


@dataclass
class LayerMomentAccumulator:
    """
    Pass-1 accumulator: folds in one batch at a time to compute the EXACT global
    mean and variance of a layer's input activation, holding only three scalars.

    Uses the Chan et al. parallel variance merge (a batched form of Welford's
    algorithm) in float64, which avoids the catastrophic cancellation that a
    naive sum-of-squares accumulation would suffer.
    """

    layer_name: str
    layer_type: LayerType

    # Running aggregate: total count, running mean, and M2 (sum of squared
    # deviations from the running mean).
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, activations: Tensor) -> None:
        """Merge one batch's values into the running mean/variance aggregate."""
        # Flatten across tokens and features so the statistic is taken over ALL
        # of the layer's activation values. We keep the original dtype here and
        # pass dtype=torch.float64 into the accumulation ops below. This avoids
        # materialising a full float64 copy of the activation tensor (what the
        # old `.double()` approach did, doubling peak VRAM per batch) while still
        # accumulating the mean and M2 in float64 for numerical rigour.
        values_flat: Tensor = activations.detach().reshape(-1)
        batch_count: int = values_flat.numel()
        if batch_count == 0:
            return

        batch_mean: float = float(values_flat.mean(dtype=torch.float64).item())
        # Reuse batch_mean rather than calling .mean() a second time; accumulate
        # the squared deviations in float64 via sum's dtype argument.
        batch_m2: float = float(
            ((values_flat - batch_mean).pow(2)).sum(dtype=torch.float64).item()
        )

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        # Chan/Welford parallel merge of two aggregates (numerically stable).
        delta: float = batch_mean - self.mean
        combined_count: int = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / combined_count)
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / combined_count)
        )
        self.count = combined_count

    def finalize(self) -> LayerThreshold:
        """Produce the exact global mean and (population) standard deviation."""
        if self.count == 0:
            return LayerThreshold(global_mean=0.0, global_std=0.0)
        # Population variance (divide by N): the exact second central moment of
        # the activation values we observed.
        variance: float = self.m2 / self.count
        return LayerThreshold(global_mean=self.mean, global_std=math.sqrt(variance))


@dataclass(frozen=True)
class LayerThresholds:
    """
    Immutable handoff between the two passes: the per-layer `LayerThreshold`s
    computed in pass 1, looked up by layer name in pass 2. Wrapping the lookup
    behind a method keeps a raw dictionary from being passed around as state.
    """

    _by_layer: dict[str, LayerThreshold]

    def for_layer(self, layer_name: str) -> LayerThreshold:
        return self._by_layer[layer_name]


# =============================================================================
# PASS 2: outlier counts and routing fractions, using the frozen thresholds
# =============================================================================


@dataclass(frozen=True)
class LayerOutlierSummary:
    """
    The final outlier report for ONE linear layer, after both passes complete.
    `frozen=True` makes it read-only: once computed, it cannot change.
    """

    layer_name: str
    layer_type: LayerType

    # The largest absolute input value seen anywhere in this layer.
    max_magnitude: float

    # The exact global mean/std (from pass 1) that define the 3-sigma cutoff,
    # surfaced here so the statistical threshold used is fully auditable.
    global_mean: float
    global_std: float

    # PRIMARY LLM.int8 METRICS (per-column). Fraction (0.0 .. 1.0) of input
    # feature columns that are "outlier columns": columns exceeding the threshold
    # in at least OUTLIER_COLUMN_PARTICIPATION_FRACTION of tokens. Because
    # LLM.int8 routes whole columns to FP16, this is the true fraction of the
    # matmul's contraction dimension pushed to FP16 - the real structured cost.
    routing_fraction_fixed: float  # threshold: |x| > 6.0
    routing_fraction_statistical: float  # threshold: |x - mean| > 3 std

    # UNSTRUCTURED BASELINES (per-value). Fraction (0.0 .. 1.0) of individual
    # input values exceeding each threshold, ignoring column structure. The GAP
    # between a routing fraction and its matching density quantifies the penalty
    # of LLM.int8's whole-column routing on this layer.
    value_outlier_density_fixed: float
    value_outlier_density_statistical: float

    # How concentrated outliers are within specific input feature columns:
    #   HIGH -> outliers cluster in a few persistent columns (good for routing)
    #   LOW  -> outliers are scattered everywhere (the LLM.int8 premise breaks)
    channel_persistence_variance: float

    # Transparency: how many individual values and how many tokens we examined.
    total_values_seen: int
    total_tokens_seen: int

    def to_dict(self) -> dict[str, object]:
        """Convert to plain Python types so `json.dump` can serialize it."""
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type.value,
            "max_magnitude": self.max_magnitude,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "routing_fraction_fixed": self.routing_fraction_fixed,
            "routing_fraction_statistical": self.routing_fraction_statistical,
            "value_outlier_density_fixed": self.value_outlier_density_fixed,
            "value_outlier_density_statistical": self.value_outlier_density_statistical,
            "channel_persistence_variance": self.channel_persistence_variance,
            "total_values_seen": self.total_values_seen,
            "total_tokens_seen": self.total_tokens_seen,
        }


@dataclass
class LayerOutlierAccumulator:
    """
    Pass-2 accumulator: folds in one batch at a time to count outliers and
    per-column routing fractions, using the FROZEN per-layer threshold from
    pass 1 for the statistical cutoff.

    Holds only scalar counters plus two fixed-size per-column counters (length =
    the layer's input feature width), never a raw activation tensor, so memory
    stays flat regardless of how many images we process.
    """

    layer_name: str
    layer_type: LayerType
    threshold: LayerThreshold  # exact global mean/std from pass 1

    max_magnitude: float = 0.0

    # Per-value outlier counts (numerators of the unstructured densities).
    fixed_value_outlier_count: int = 0
    statistical_value_outlier_count: int = 0
    total_value_count: int = 0
    total_token_count: int = 0

    # Per-COLUMN token-participation counters, lazily allocated on the first
    # batch. Each holds, per input feature column, the number of TOKENS in which
    # that column was an outlier under the respective threshold.
    fixed_tokens_per_channel: Tensor | None = None
    statistical_tokens_per_channel: Tensor | None = None

    # Channel persistence (fixed threshold) is measured per batch and averaged.
    channel_variance_sum: float = 0.0
    batch_count: int = 0

    def update(self, activations: Tensor) -> None:
        """Fold one batch of INPUT activations into the running outlier counts."""
        activations = activations.detach()

        # Collapse [Batch, Sequence] into a single "one row per token" axis:
        # rows are tokens, columns are input feature channels.
        feature_count: int = activations.shape[-1]
        tokens: Tensor = activations.reshape(-1, feature_count)  # [Tokens, Features]
        token_count: int = tokens.shape[0]
        abs_tokens: Tensor = tokens.abs()

        # Lazily allocate the per-column counters now that the width is known.
        fixed_per_channel: Tensor = (
            self.fixed_tokens_per_channel
            if self.fixed_tokens_per_channel is not None
            else torch.zeros(feature_count, dtype=torch.long, device=tokens.device)
        )
        statistical_per_channel: Tensor = (
            self.statistical_tokens_per_channel
            if self.statistical_tokens_per_channel is not None
            else torch.zeros(feature_count, dtype=torch.long, device=tokens.device)
        )

        # --- Maximum magnitude ----------------------------------------------
        self.max_magnitude = max(self.max_magnitude, float(abs_tokens.max().item()))

        # --- Fixed-threshold mask (|x| > 6.0) -------------------------------
        fixed_mask: Tensor = abs_tokens > FIXED_OUTLIER_THRESHOLD
        self.fixed_value_outlier_count += int(fixed_mask.sum().item())

        # --- Statistical-threshold mask (|x - global_mean| > 3 * global_std) -
        # Uses the EXACT global mean/std frozen from pass 1, not per-batch stats.
        deviations: Tensor = (tokens - self.threshold.global_mean).abs()
        statistical_mask: Tensor = deviations > self.threshold.statistical_cutoff
        self.statistical_value_outlier_count += int(statistical_mask.sum().item())

        self.total_value_count += tokens.numel()
        self.total_token_count += token_count

        # Per-column participation: how many tokens make each column an outlier.
        self.fixed_tokens_per_channel = fixed_per_channel + fixed_mask.sum(dim=0)
        self.statistical_tokens_per_channel = (
            statistical_per_channel + statistical_mask.sum(dim=0)
        )

        # --- Channel persistence (variance of per-column outlier counts) ----
        outliers_per_channel: Tensor = fixed_mask.sum(dim=0).float()  # [Features]
        mean_outliers_per_channel: Tensor = outliers_per_channel.mean()
        squared_deviations: Tensor = (
            outliers_per_channel - mean_outliers_per_channel
        ) ** 2
        self.channel_variance_sum += float(squared_deviations.mean().item())
        self.batch_count += 1

        # `activations`, `tokens`, and the masks all fall out of scope here, so
        # their GPU memory is freed before the next layer's hook runs.

    def finalize(self) -> LayerOutlierSummary:
        """Turn the running accumulators into the final per-layer summary."""
        fixed_per_channel: Tensor | None = self.fixed_tokens_per_channel
        statistical_per_channel: Tensor | None = self.statistical_tokens_per_channel

        # Defensive guard: if a layer somehow never fired, report all zeros
        # instead of dividing by zero.
        if (
            self.total_value_count == 0
            or self.total_token_count == 0
            or self.batch_count == 0
            or fixed_per_channel is None
            or statistical_per_channel is None
        ):
            return LayerOutlierSummary(
                layer_name=self.layer_name,
                layer_type=self.layer_type,
                max_magnitude=0.0,
                global_mean=self.threshold.global_mean,
                global_std=self.threshold.global_std,
                routing_fraction_fixed=0.0,
                routing_fraction_statistical=0.0,
                value_outlier_density_fixed=0.0,
                value_outlier_density_statistical=0.0,
                channel_persistence_variance=0.0,
                total_values_seen=0,
                total_tokens_seen=0,
            )

        fixed_density: float = self.fixed_value_outlier_count / self.total_value_count
        statistical_density: float = (
            self.statistical_value_outlier_count / self.total_value_count
        )

        # A column is an outlier column if it exceeded the threshold in at least
        # the participation fraction of all tokens seen. The routing fraction is
        # the share of such columns = the fraction of the matmul's contraction
        # dimension LLM.int8 would send to FP16.
        min_outlier_tokens: float = (
            OUTLIER_COLUMN_PARTICIPATION_FRACTION * self.total_token_count
        )
        feature_count: int = fixed_per_channel.numel()
        fixed_outlier_columns: int = int(
            (fixed_per_channel.float() >= min_outlier_tokens).sum().item()
        )
        statistical_outlier_columns: int = int(
            (statistical_per_channel.float() >= min_outlier_tokens).sum().item()
        )

        return LayerOutlierSummary(
            layer_name=self.layer_name,
            layer_type=self.layer_type,
            max_magnitude=self.max_magnitude,
            global_mean=self.threshold.global_mean,
            global_std=self.threshold.global_std,
            routing_fraction_fixed=fixed_outlier_columns / feature_count,
            routing_fraction_statistical=statistical_outlier_columns / feature_count,
            value_outlier_density_fixed=fixed_density,
            value_outlier_density_statistical=statistical_density,
            channel_persistence_variance=self.channel_variance_sum / self.batch_count,
            total_values_seen=self.total_value_count,
            total_tokens_seen=self.total_token_count,
        )


# =============================================================================
# Collectors: one accumulator per layer, for each pass
# =============================================================================


class MomentCollector:
    """Owns one `LayerMomentAccumulator` per layer for PASS 1."""

    def __init__(self) -> None:
        self._stats_by_layer: dict[str, LayerMomentAccumulator] = {}

    def register_layer(self, layer_name: str, layer_type: LayerType) -> None:
        """Create an empty moment accumulator for a layer before pass 1 starts."""
        self._stats_by_layer[layer_name] = LayerMomentAccumulator(
            layer_name=layer_name,
            layer_type=layer_type,
        )

    def record_activations(self, layer_name: str, activations: Tensor) -> None:
        """Fold one batch of activations into the named layer's accumulator."""
        self._stats_by_layer[layer_name].update(activations)

    def build_thresholds(self) -> LayerThresholds:
        """Finalize every layer's exact global mean/std into a frozen handoff."""
        return LayerThresholds(
            {name: stats.finalize() for name, stats in self._stats_by_layer.items()}
        )


class OutlierStatsCollector:
    """Owns one `LayerOutlierAccumulator` per layer for PASS 2."""

    def __init__(self, thresholds: LayerThresholds) -> None:
        self._thresholds: LayerThresholds = thresholds
        self._stats_by_layer: dict[str, LayerOutlierAccumulator] = {}

    def register_layer(self, layer_name: str, layer_type: LayerType) -> None:
        """Create an outlier accumulator seeded with this layer's pass-1 threshold."""
        self._stats_by_layer[layer_name] = LayerOutlierAccumulator(
            layer_name=layer_name,
            layer_type=layer_type,
            threshold=self._thresholds.for_layer(layer_name),
        )

    def record_activations(self, layer_name: str, activations: Tensor) -> None:
        """Fold one batch of activations into the named layer's accumulator."""
        self._stats_by_layer[layer_name].update(activations)

    def build_summaries(self) -> list[LayerOutlierSummary]:
        """Finalize every layer into an immutable summary, in registration order."""
        return [stats.finalize() for stats in self._stats_by_layer.values()]


# =============================================================================
# Hook wiring (shared by both passes)
# =============================================================================


class ActivationRecorder(Protocol):
    """
    The capability both collectors share: register layers up front, then receive
    each layer's input activations during a forward pass. The hook machinery and
    the driver depend only on this protocol, so the same wiring drives pass 1
    (`MomentCollector`) and pass 2 (`OutlierStatsCollector`).
    """

    def register_layer(self, layer_name: str, layer_type: LayerType) -> None: ...

    def record_activations(self, layer_name: str, activations: Tensor) -> None: ...


def extract_activation_tensor(module_inputs: object) -> Tensor | None:
    """
    Reduce a forward pre-hook's `inputs` to a single activation tensor.

    A pre-hook receives the module's positional inputs as a tuple; for an
    `nn.Linear` the activation we want is the first (and only) element. We return
    the first Tensor we find, or None if there is nothing measurable (in which
    case the caller simply skips this batch). A bare Tensor is also accepted and
    returned unchanged, which keeps the helper easy to unit-test.
    """
    if isinstance(module_inputs, Tensor):
        return module_inputs
    if isinstance(module_inputs, (tuple, list)):
        for element in module_inputs:
            if isinstance(element, Tensor):
                return element
    return None


# A forward pre-hook is called as hook(module, inputs) and returns nothing.
# `inputs` is the tuple of positional arguments about to enter the module;
# `extract_activation_tensor` pulls the input activation out of it.
PreForwardHook: TypeAlias = Callable[[nn.Module, tuple[Tensor, ...]], None]


def make_measurement_hook(
    recorder: ActivationRecorder, layer_name: str
) -> PreForwardHook:
    """
    Build a forward PRE-hook bound to one specific layer and recorder.

    PyTorch calls the returned `hook` right BEFORE the module runs, handing us
    its `inputs`. We normalize that to a single activation tensor - the matmul's
    input, i.e. the tensor `LLM.int8()` would decompose - and pass it to the
    recorder, which reduces it to scalars and discards it. The same factory works
    for both passes; only the recorder differs. The `_module` argument is part of
    the hook signature but unused here.
    """

    def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        activations = extract_activation_tensor(inputs)
        if activations is not None:
            recorder.record_activations(layer_name, activations)

    return hook
