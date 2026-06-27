"""
hooks.py
========

The measurement machinery for Experiment 1.

What is a forward hook?
-----------------------
PyTorch lets us attach a "forward hook" to any module. After that module runs
during a forward pass, PyTorch calls our hook function and hands it the
module's output tensor. This lets us *observe* the activations flowing through
each linear layer without modifying the model's code at all.

The hard memory rule (8 GB VRAM budget)
---------------------------------------
A single activation tensor can be large (tens of MB), there are dozens of
layers, and we stream thousands of images. If we ever stored the raw tensors we
would run out of memory almost immediately. So every hook reduces its tensor to
a handful of scalar numbers *on the fly* and then lets the tensor be discarded.

Design: three small types
--------------------------
  1. `LayerActivationStats` - a mutable accumulator that folds in one batch at a
     time. It holds ONLY scalar counters, never a raw activation tensor.
  2. `LayerOutlierSummary`  - the final, immutable, human-readable result for one
     layer. This is what we save to JSON and plot.
  3. `OutlierStatsCollector`- owns one accumulator per layer and routes each
     layer's activations to the right place.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch.nn as nn
from torch import Tensor

from src.model_utils import LayerType

# --- what counts as an "outlier" ---------------------------------------------

# The fixed magnitude threshold used by the LLM.int8() paper. Any activation
# value whose absolute size exceeds this would be routed to FP16 under a
# mixed-precision scheme, so this fraction is the "routing cost".
FIXED_OUTLIER_THRESHOLD: float = 6.0

# The statistical threshold: a value is an outlier if it sits more than this
# many standard deviations away from its layer's mean activation.
STD_THRESHOLD_MULTIPLIER: float = 3.0


@dataclass(frozen=True)
class LayerOutlierSummary:
    """
    The final outlier report for ONE linear layer, after every image has been
    processed. `frozen=True` makes it read-only: once computed, it cannot change.
    """

    layer_name: str
    layer_type: LayerType

    # The largest absolute activation value seen anywhere in this layer.
    max_magnitude: float

    # Fraction (0.0 .. 1.0) of activation values with |x| > 6.0.
    # This is the share of math a fixed-threshold scheme would push to FP16.
    fixed_outlier_density: float

    # Fraction (0.0 .. 1.0) of activation values lying > 3 std-devs from the mean.
    statistical_outlier_density: float

    # How concentrated the outliers are within specific feature channels:
    #   HIGH -> outliers cluster in a few persistent channels (good for routing)
    #   LOW  -> outliers are scattered everywhere (the LLM.int8 premise breaks)
    channel_persistence_variance: float

    # How many individual activation numbers we examined (for transparency).
    total_values_seen: int

    def to_dict(self) -> dict[str, object]:
        """Convert to plain Python types so `json.dump` can serialize it."""
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type.value,
            "max_magnitude": self.max_magnitude,
            "fixed_outlier_density": self.fixed_outlier_density,
            "statistical_outlier_density": self.statistical_outlier_density,
            "channel_persistence_variance": self.channel_persistence_variance,
            "total_values_seen": self.total_values_seen,
        }


@dataclass
class LayerActivationStats:
    """
    Running statistics for ONE linear layer while images stream through it.

    Every field is a small scalar accumulator - never a raw activation tensor -
    so memory usage stays flat no matter how many images we process.
    """

    layer_name: str
    layer_type: LayerType

    # Running maximum of |activation| seen so far.
    max_magnitude: float = 0.0

    # Running totals, combined into densities only at the very end.
    fixed_outlier_count: int = 0
    statistical_outlier_count: int = 0
    total_value_count: int = 0

    # Channel persistence is measured once per batch; we average across batches.
    channel_variance_sum: float = 0.0
    batch_count: int = 0

    def update(self, activations: Tensor) -> None:
        """
        Fold one batch of activations into the running statistics.

        `activations` has shape [Batch, Sequence, Features]:
          * Batch    - images in this mini-batch (e.g. 64)
          * Sequence - tokens per image (197 for ViT-B/16: 196 patches + 1 CLS)
          * Features - this layer's output width (e.g. 768, or 3072 inside MLPs)
        """
        # We only READ these numbers, so detach from the autograd graph.
        activations = activations.detach()

        # Collapse [Batch, Sequence] into a single "one row per token" axis. The
        # result is a 2-D matrix: rows are tokens, columns are feature channels.
        feature_count: int = activations.shape[-1]
        tokens: Tensor = activations.reshape(-1, feature_count)  # [Tokens, Features]

        # --- Metric 1: maximum magnitude ------------------------------------
        # The single largest absolute value anywhere in this batch.
        batch_max_magnitude: float = float(tokens.abs().max().item())
        self.max_magnitude = max(self.max_magnitude, batch_max_magnitude)

        # --- Metric 2a: fixed-threshold outlier density ---------------------
        # A boolean matrix, True wherever |value| exceeds 6.0.
        fixed_outlier_mask: Tensor = tokens.abs() > FIXED_OUTLIER_THRESHOLD
        self.fixed_outlier_count += int(fixed_outlier_mask.sum().item())

        # --- Metric 2b: statistical-threshold outlier density ---------------
        # Here "outlier" means far from this batch's own mean activation. We use
        # the batch's mean and standard deviation to define the cutoff.
        mean: Tensor = tokens.mean()
        std: Tensor = tokens.std()
        statistical_cutoff: Tensor = STD_THRESHOLD_MULTIPLIER * std
        statistical_outlier_mask: Tensor = (tokens - mean).abs() > statistical_cutoff
        self.statistical_outlier_count += int(statistical_outlier_mask.sum().item())

        # The denominator shared by both densities.
        self.total_value_count += tokens.numel()

        # --- Metric 3: channel persistence ----------------------------------
        # For each feature channel (column), count how many fixed-threshold
        # outliers landed in it. Summing down dim=0 collapses the token axis.
        outliers_per_channel: Tensor = fixed_outlier_mask.sum(
            dim=0
        ).float()  # [Features]

        # Now ask: are those per-channel counts spread out, or all similar?
        # We compute the population variance explicitly so the math is visible:
        #     variance = average of (count - average_count) ** 2
        # A few channels hogging all the outliers -> large variance (persistent).
        # Outliers sprinkled evenly across channels -> small variance (scattered).
        mean_outliers_per_channel: Tensor = outliers_per_channel.mean()
        squared_deviations: Tensor = (
            outliers_per_channel - mean_outliers_per_channel
        ) ** 2
        batch_channel_variance: float = float(squared_deviations.mean().item())

        self.channel_variance_sum += batch_channel_variance
        self.batch_count += 1

        # `activations`, `tokens`, and the masks all fall out of scope here, so
        # their GPU memory is freed before the next layer's hook runs.

    def finalize(self) -> LayerOutlierSummary:
        """Turn the running accumulators into the final per-layer summary."""
        # Defensive guard: if a layer somehow never fired, report all zeros
        # instead of dividing by zero.
        if self.total_value_count == 0 or self.batch_count == 0:
            return LayerOutlierSummary(
                layer_name=self.layer_name,
                layer_type=self.layer_type,
                max_magnitude=0.0,
                fixed_outlier_density=0.0,
                statistical_outlier_density=0.0,
                channel_persistence_variance=0.0,
                total_values_seen=0,
            )

        fixed_density: float = self.fixed_outlier_count / self.total_value_count
        statistical_density: float = (
            self.statistical_outlier_count / self.total_value_count
        )
        average_channel_variance: float = self.channel_variance_sum / self.batch_count

        return LayerOutlierSummary(
            layer_name=self.layer_name,
            layer_type=self.layer_type,
            max_magnitude=self.max_magnitude,
            fixed_outlier_density=fixed_density,
            statistical_outlier_density=statistical_density,
            channel_persistence_variance=average_channel_variance,
            total_values_seen=self.total_value_count,
        )


class OutlierStatsCollector:
    """
    Owns one `LayerActivationStats` accumulator per hooked layer and routes each
    layer's activations to the right accumulator.

    We keep an internal dict for fast name-based lookup, but it is hidden behind
    methods so the rest of the program never passes a raw dictionary around as a
    stand-in for a real type.
    """

    def __init__(self) -> None:
        self._stats_by_layer: dict[str, LayerActivationStats] = {}

    def register_layer(self, layer_name: str, layer_type: LayerType) -> None:
        """Create an empty accumulator for a layer before measurement starts."""
        self._stats_by_layer[layer_name] = LayerActivationStats(
            layer_name=layer_name,
            layer_type=layer_type,
        )

    def record_activations(self, layer_name: str, activations: Tensor) -> None:
        """Fold one batch of activations into the named layer's accumulator."""
        self._stats_by_layer[layer_name].update(activations)

    def build_summaries(self) -> list[LayerOutlierSummary]:
        """Finalize every layer into an immutable summary, in registration order."""
        return [stats.finalize() for stats in self._stats_by_layer.values()]


def extract_activation_tensor(module_output: object) -> Tensor | None:
    """
    Reduce a module's forward output to a single activation tensor.

    Different modules return different shapes of output:
      * an `nn.Linear` returns a Tensor directly;
      * an `nn.MultiheadAttention` returns a tuple (attn_output, attn_weights),
        where we want the first element (and attn_weights is often None).

    We return the first Tensor we find, or None if there is nothing measurable
    (in which case the caller simply skips this batch).
    """
    if isinstance(module_output, Tensor):
        return module_output
    if isinstance(module_output, (tuple, list)):
        for element in module_output:
            if isinstance(element, Tensor):
                return element
    return None


# A forward hook is called as hook(module, inputs, output) and returns nothing.
# `output` is typed `object` because, depending on the module, it may be a
# Tensor or a tuple - `extract_activation_tensor` normalizes the difference.
ForwardHook: TypeAlias = Callable[[nn.Module, tuple[Tensor, ...], object], None]


def make_outlier_hook(collector: OutlierStatsCollector, layer_name: str) -> ForwardHook:
    """
    Build a forward hook bound to one specific layer.

    PyTorch will call the returned `hook` right after the module runs, handing us
    its `output`. We normalize that to a single activation tensor and pass it to
    the collector, which reduces it to scalars and discards it. The `_module`
    and `_inputs` arguments are part of the hook signature but unused here.
    """

    def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: object) -> None:
        activations = extract_activation_tensor(output)
        if activations is not None:
            collector.record_activations(layer_name, activations)

    return hook
