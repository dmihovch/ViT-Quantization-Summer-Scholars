# Course-Correction Report: Addressing Six Warnings from the Literature

**Date:** 2026-07-05
**Status:** Analysis and recommendations. No code changes have been made.
**Depends on:** `docs/literature-survey.md`

---

This document takes each of the six warnings raised in the literature survey,
verifies them against the source papers, assesses how they apply to this
codebase, and proposes concrete changes. Every proposed change is scoped to a
specific file and designed to integrate with the existing architecture.

---

## Warning 1: Symmetric Uniform Quantization May Be Wrong for Post-GELU Activations

**Source:** Yuan et al. (2022), *PTQ4ViT* (arXiv:2111.12293). The abstract
states: "We observe the distributions of activation values after softmax and
GELU functions are quite different from the Gaussian distribution." The paper
proposes twin uniform quantization: separate quantization ranges for positive
and negative values, motivated by the observation that post-GELU activations are
predominantly positive with a long tail.

**Verification against the source:** Confirmed. The abstract explicitly
identifies post-GELU and post-Softmax activations as non-Gaussian and proposes
separate treatment. The paper's Figure 2 (visible in the PDF) shows that
post-GELU activations in ViT MLP layers have a sharp peak near zero, a long
positive tail, and almost no negative values. Symmetric quantization centered at
zero wastes half the quantization bins on negative values that never occur.

**How this applies to the codebase:** In ViT-B/16, the `mlp.fc2` layers receive
post-GELU activations. The GELU activation function outputs values in
approximately [-0.17, infinity), with the vast majority being positive. Your
`src/quantization.py` uses symmetric quantization (`QMIN=-127, QMAX=127`,
zeropoint at zero) for all layers regardless of their activation distribution.
This means every `mlp.fc2` layer loses roughly 1 bit of effective precision
because half the quantization grid covers negative values that post-GELU
activations almost never occupy.

**Severity:** Medium. This is a systematic accuracy penalty on all 12 `mlp.fc2`
layers plus the `head` layer (which receives the final representation after the
last block's MLP). The penalty is uniform across layers, so it does not distort
the per-layer sensitivity ranking from Experiment 3. However, it means the
absolute accuracy numbers in Experiments 2-4 are lower than they would be with
an appropriate quantizer. If the goal is to measure the best achievable INT8
accuracy, symmetric quantization is the wrong tool for these layers.

**Proposed change (src/quantization.py):**

Add an asymmetric quantization function and a layer-type-aware dispatch in
`make_activation_quant_hook`. The implementation follows the same pattern as the
existing `quantize_per_tensor` but uses the observed min/max range rather than a
symmetric range around zero.

```python
# New function in src/quantization.py

def quantize_per_tensor_asymmetric(x: torch.Tensor) -> torch.Tensor:
    """
    Applies per-tensor asymmetric quantization.

    The quantization range is [min(x), max(x)] mapped to [0, 255] for
    unsigned INT8, then shifted to [-128, 127] for signed representation.
    This preserves the full dynamic range for distributions that are not
    centered at zero (e.g., post-GELU activations).

    Args:
        x: The input float tensor.

    Returns:
        The fake-quantized float tensor.
    """
    x_min = torch.min(x)
    x_max = torch.max(x)
    scale = (x_max - x_min) / 255.0
    # Avoid division by zero for constant tensors.
    scale = torch.clamp(scale, min=1e-8)
    zeropoint = torch.round(-x_min / scale) - 128
    zeropoint = torch.clamp(zeropoint, -128, 127)
    return quantize_dequantize(x, scale, zeropoint, -128, 127)
```

The dispatch logic in `make_activation_quant_hook` would accept an optional
`layer_type` parameter and select the quantizer accordingly:

```python
def make_activation_quant_hook(
    strategy: str,
    layer_name: str,
    mse_tracker: MSEAccumulator | None = None,
    layer_type: str | None = None,
) -> Callable:
    # ...
    def hook(module, args):
        x = args[0]
        if layer_type in ("FeedForward_fc2", "Other") and strategy == "per_tensor":
            x_q = quantize_per_tensor_asymmetric(x)
        elif strategy == "per_token" and x.dim() == 3:
            x_q = quantize_per_token(x)
        else:
            x_q = quantize_per_tensor(x)
        # ...
```

**Impact on existing experiments:**
- Experiment 2: Configs A and B (per-tensor activations) would use asymmetric
  quantization for `mlp.fc2` and `head` layers. Accuracy should improve by
  roughly 0.5-1.5% on ImageNet (based on PTQ4ViT's reported numbers).
- Experiment 3: The per-layer sensitivity ranking should not change materially
  because the asymmetric penalty is uniform across all `mlp.fc2` layers.
  However, the absolute accuracy drop for `mlp.fc2` layers will be smaller,
  which may change which layers cross a significance threshold.
- Experiment 4: The decomposition logic is unaffected because it operates on
  `mlp.fc1` layers (post-LayerNorm, not post-GELU).

**Risk:** Low. The change is additive (new function, optional parameter). All
existing tests pass because the default behavior (no `layer_type` provided) uses
symmetric quantization as before. The asymmetric quantizer follows the same
`quantize_dequantize` code path and is testable with hand-computed tensors.

**Test plan:** Add `test_quantize_per_tensor_asymmetric` to
`tests/test_quantization.py`. The test uses a tensor with all-positive values
(e.g., `[0.0, 1.0, 5.0]`) and verifies that the quantized output preserves the
zero value exactly (zeropoint maps min to -128) and that the max value is
preserved within quantization error.

---

## Warning 2: Operation-Specific Quantizers May Be Necessary Before Sensitivity Rankings Are Meaningful

**Source:** Liu et al. (2021), *Post-Training Quantization for Vision
Transformer* (arXiv:2106.14156). The abstract states: "We introduce a ranking
loss into the conventional quantization objective that aims to keep the relative
order of the self-attention results after quantization." The paper also analyzes
"the relationship between quantization loss of different layers and the feature
diversity."

**Verification against the source:** Confirmed. The paper's core argument is
that different operations in a ViT need different quantization strategies. The
abstract specifically mentions the self-attention mechanism as requiring special
treatment (ranking loss to preserve attention order). The paper's mixed-precision
scheme uses the nuclear norm of attention maps and output features to decide
bit-width per layer.

**How this applies to the codebase:** Your Experiment 3 applies
`quantize_per_tensor` (symmetric, uniform) to every `nn.Linear` regardless of
what tensor type enters it. The three tensor types are:

1. **Post-LayerNorm activations** (entering `attn.qkv` and `mlp.fc1`):
   approximately Gaussian with zero mean and unit variance per token, but with
   inter-channel scale variation. Symmetric uniform quantization is appropriate
   here.

2. **Post-attention activations** (entering `attn.proj`): the concatenated
   multi-head attention output. These values are bounded (weighted sums of
   Value vectors with attention weights summing to 1) and tend to have moderate
   dynamic range. Symmetric uniform quantization is reasonable.

3. **Post-GELU activations** (entering `mlp.fc2`): asymmetric, predominantly
   positive, with a long tail. Symmetric uniform quantization is inappropriate
   (see Warning 1).

The Liu et al. paper adds a fourth category: **post-Softmax attention weights**.
These are not inputs to `nn.Linear` layers (they are computed inside the
attention mechanism), so your hook-based measurement does not see them. However,
if you ever extend quantization to the attention computation itself (beyond just
the linear projections), the post-Softmax distribution requires a log2 quantizer.

**Severity:** Medium-High for Experiment 3 interpretability. The current setup
conflates two sources of accuracy loss: (a) the layer is intrinsically sensitive
to quantization, and (b) the quantizer is poorly matched to the layer's
activation distribution. If `mlp.fc2` layers show unexpected sensitivity in
Experiment 3, you cannot distinguish these two causes without running a control
experiment with asymmetric quantization.

**Proposed change:** The fix for Warning 1 (adding asymmetric quantization for
post-GELU layers) addresses the most consequential case of quantizer mismatch.
The post-attention and post-LayerNorm cases are adequately served by symmetric
quantization. The post-Softmax case is out of scope because your hooks only
instrument `nn.Linear` layers.

For Experiment 3 specifically, add a second pass that uses operation-specific
quantizers and compare the sensitivity rankings. If the rankings are materially
different, report both and discuss the quantizer-mismatch confound explicitly.

**Implementation approach for Experiment 3:**

Add a `--quantizer-aware` flag to `run_experiment3_sensitivity.py`. When set,
the script uses `quantize_per_tensor_asymmetric` for `mlp.fc2` and `head`
layers, and `quantize_per_tensor` (symmetric) for all others. The output CSV
gains a `quantizer_type` column. The visualization script plots both passes on
the same axes with different colors.

```python
# In run_experiment3_sensitivity.py, inside the per-layer loop:

from src.model_utils import classify_linear_layer, LayerType

layer_type = classify_linear_layer(layer_name)
if args.quantizer_aware and layer_type in (LayerType.FEEDFORWARD_FC2, LayerType.OTHER):
    layer.weight.data = quantize_per_tensor(layer.weight.data)
    # Activation quantization is handled by a hook (see Warning 1 fix).
    # For Experiment 3, we quantize weights only (the current behavior).
    # The activation quantizer mismatch is a separate concern.
```

Note: Experiment 3 currently only quantizes weights (`layer.weight.data =
quantize_per_tensor(layer.weight.data)`), not activations. The quantizer
mismatch concern applies to activation quantization in Experiments 2 and 4. For
Experiment 3, the weight quantizer is always symmetric per-tensor, which is
standard. The warning about Experiment 3 sensitivity rankings being confounded
by activation quantizer mismatch only applies if Experiment 3 is extended to
quantize activations as well.

**Risk:** Low. The `--quantizer-aware` flag is off by default, preserving the
current behavior. The additional pass doubles Experiment 3 runtime, but
Experiment 3 already runs per-layer (49 passes), so one extra full-model pass
for the baseline is a small relative cost.

---

## Warning 3: SmoothQuant May Not Transfer to Dense-Outlier Regimes

**Source:** Xiao et al. (2023), *SmoothQuant* (arXiv:2211.10438). The abstract
states: "Based on the fact that weights are easy to quantize while activations
are not, SmoothQuant smooths the activation outliers by offline migrating the
quantization difficulty from activations to weights." The paper evaluates on
OPT, BLOOM, GLM, MT-NLG, Llama-1/2, Falcon, Mistral, and Mixtral.

**Verification against the source:** Confirmed that SmoothQuant was designed and
tested exclusively on LLMs. The abstract lists eight model families, all of
which are language models. The paper does not claim to have tested on vision
transformers. The outlier topology in LLMs (sparse, extreme, channel-persistent)
is qualitatively different from what you observe at ViT blocks 9-10 (dense,
moderate, spread across many channels).

The SmoothQuant smoothing factor for channel `j` is:

```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
```

where `X_j` is the j-th input channel of the activation and `W_j` is the j-th
column of the weight matrix. When `α=0.5`, the factor balances the dynamic range
between activation and weight. When outliers are sparse (few channels have large
`max(|X_j|)`), only those few channels get aggressive smoothing. When outliers
are dense (many channels have large `max(|X_j|)`, as in your blocks 9-10), many
channels get aggressive smoothing simultaneously. The weight-side amplification
`1/s_j` accumulates across output channels in the subsequent matmul.

**How this applies to the codebase:** Your July roadmap (Section 5, Path B)
proposes applying SmoothQuant to blocks 9-10 fc1. At block 10, σ=11.68 and the
routing fraction is 99.74%. To bring σ below 6.0 (the INT8-safe threshold), the
smoothing factor would need to compress activation scale by roughly 2x across
most channels. This means the weight matrix would be amplified by roughly 2x
across most columns. The weight's per-channel dynamic range would double, which
may push it outside the representable range for INT8 quantization.

**Severity:** High for Path B. If SmoothQuant fails on blocks 9-10, the primary
remedy in the roadmap is invalidated. The fallback (Path A: validate and write
up) is already planned, so this is not a project-ending risk. However, it means
the SmoothQuant experiment should be designed as a quick feasibility check
before committing significant implementation time.

**Proposed change:** Before implementing the full SmoothQuant pipeline from the
July roadmap (Section 5, "Implementation plan"), run a one-day feasibility
check:

1. Compute the per-channel smoothing factors for blocks 9-10 fc1 using the
   per-channel activation statistics already in
   `outputs/exp1_outlier_maps/outlier_stats.json` (from Experiment 1 Pass 1).
   No new data collection needed.

2. Apply the smoothing transformation to the fc1 weight matrix in memory:
   `W_smoothed = W * diag(s)` where `s` is the per-channel smoothing factor.

3. Check whether the smoothed weight matrix is still quantizable to INT8 with
   acceptable error. The check: quantize `W_smoothed` with per-channel
   quantization and measure the MSE against the original `W_smoothed`. If the
   weight quantization MSE is more than 10x the activation quantization MSE
   reduction, the smoothing has shifted the problem rather than solving it.

4. If the weight remains quantizable at α values of 0.5, 0.7, and 0.9, proceed
   with the full SmoothQuant implementation. If the weight becomes unquantizable
   at all α values, document this as a negative result and fall back to Path A.

**Implementation (new file: `src/smoothing.py`):**

```python
"""
Activation-weight smoothing for ViT quantization.

Implements the SmoothQuant transformation (Xiao et al., 2023) adapted for
Vision Transformer linear layers.
"""

import torch
from torch import nn


def compute_smoothing_factors(
    activation_max_per_channel: torch.Tensor,
    weight_max_per_channel: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-channel smoothing factors s_j.

    Args:
        activation_max_per_channel: max(|X_j|) for each input channel j.
            Shape: (in_features,).
        weight_max_per_channel: max(|W_ij|) for each input channel j,
            taken over output channels i. Shape: (in_features,).
        alpha: Migration strength in [0, 1]. 0 = all difficulty on weights.
            1 = all difficulty on activations. 0.5 = balanced.

    Returns:
        Smoothing factors s_j. Shape: (in_features,).
    """
    eps = 1e-8
    s = (activation_max_per_channel + eps).pow(alpha) / (
        weight_max_per_channel + eps
    ).pow(1.0 - alpha)
    return s


def apply_smoothing_to_linear(
    layer: nn.Linear,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply SmoothQuant transformation to a linear layer in-place.

    Activation: X_smoothed = X / s  (applied via a pre-hook, not here)
    Weight:     W_smoothed = W * s  (applied in-place here)

    Args:
        layer: The nn.Linear to transform.
        s: Per-channel smoothing factors. Shape must broadcast with
           layer.weight (in_features,).

    Returns:
        (original_weight, s) so the transformation can be reversed.
    """
    original_weight = layer.weight.data.clone()
    # s has shape (in_features,). Broadcast to (out_features, in_features).
    layer.weight.data = layer.weight.data * s.unsqueeze(0)
    return original_weight, s


def reverse_smoothing(
    layer: nn.Linear,
    original_weight: torch.Tensor,
) -> None:
    """Restore the original weight, undoing the SmoothQuant transformation."""
    layer.weight.data = original_weight


def make_smoothing_activation_hook(
    s: torch.Tensor,
) -> callable:
    """
    Build a forward pre-hook that scales activations down by 1/s.

    This is the activation-side counterpart to apply_smoothing_to_linear.
    Together they implement Y = (X / s) @ (W * s)^T = X @ W^T.
    """
    s_inv = 1.0 / s

    def hook(module, args):
        x = args[0]
        # x shape: (batch, tokens, in_features) or (batch, in_features)
        # s_inv shape: (in_features,)
        return (x * s_inv,)

    return hook
```

**Test plan:** Add `tests/test_smoothing.py` with:
- `test_smoothing_preserves_matmul_output`: Apply smoothing to a small linear
  layer, run a forward pass with and without smoothing, verify the outputs match
  within floating-point error.
- `test_smoothing_factors_bounded`: Verify that for α in [0, 1], the smoothing
  factors are finite and positive.
- `test_reverse_smoothing_restores_weight`: Apply then reverse smoothing, verify
  the weight is bitwise identical to the original.

**Risk:** Low. The feasibility check uses existing data (no new experiment runs)
and a small amount of new code. If SmoothQuant is viable, the `src/smoothing.py`
module is ready for the full Path B implementation. If it is not viable, the
module is still useful for documenting the negative result.

---

## Warning 4: The Block 9-10 Explosion May Be Token-Driven Rather Than Channel-Driven

**Source:** Darcet et al. (2023), *Vision Transformers Need Registers*
(arXiv:2309.16588). The abstract states: "We identify and characterize artifacts
in feature maps of both supervised and self-supervised ViT networks. The
artifacts correspond to high-norm tokens appearing during inference primarily in
low-informative background areas of images."

**Verification against the source:** Confirmed. The paper identifies high-norm
**tokens** (specific spatial positions in the sequence), not high-norm
**channels** (specific feature dimensions). The artifacts appear in background
image regions. The paper's solution is to add learnable [REG] tokens that absorb
the high-norm information.

The distinction matters for your project: if the block 9-10 fc1 explosion is
caused by a few tokens with extreme norms (the Darcet phenomenon), then
per-channel smoothing (SmoothQuant) would not fix it because SmoothQuant
operates on feature channels, not tokens. Per-token scaling (your Experiment 2
Config C) would help in that case because it adapts the quantization scale to
each token's individual dynamic range.

**How this applies to the codebase:** Your Experiment 1 measures per-channel
statistics (routing fraction per input feature column) but does not measure
per-token statistics. You do not know whether the block 9-10 explosion is driven
by a few extreme tokens or by many tokens with moderately elevated values across
many channels.

**Severity:** High for Path B. If the explosion is token-driven, SmoothQuant is
the wrong remedy and Path B time would be wasted. The fix is a diagnostic that
can be run before committing to Path B.

**Proposed change:** Add a per-token analysis pass to Experiment 1. This is a
small extension to the existing hook infrastructure that answers a specific
question: at blocks 9-10 fc1, what fraction of tokens account for what fraction
of the total activation mass?

**Implementation (extension to src/hooks.py):**

Add a new dataclass to `src/hooks.py` that records per-token statistics for
specific layers of interest:

```python
@dataclass
class PerTokenOutlierProfile:
    """Per-token outlier statistics for a single layer."""
    layer_name: str
    # For each token position in the sequence (197 tokens for ViT-B/16),
    # the fraction of values exceeding the fixed 6.0 threshold.
    per_position_outlier_density: list[float]
    # The mean L2 norm per token position.
    per_position_mean_norm: list[float]
    # The fraction of total activation mass carried by the top-k tokens.
    topk_token_mass_fraction: dict[int, float]  # {k: fraction}
```

The measurement runs during Experiment 1 Pass 2 (when activations are already
being streamed through the model). For the layers of interest (blocks 8-11
fc1), the hook records per-token statistics in addition to the existing
per-channel statistics.

**Diagnostic question this answers:** If the top 5 tokens (out of 197) carry
more than 50% of the total activation mass at blocks 9-10 fc1, the explosion is
token-driven and SmoothQuant is the wrong approach. If the mass is evenly
distributed across tokens, the explosion is channel-driven and SmoothQuant is
appropriate.

**Integration with existing code:** The `LayerOutlierAccumulator.update()`
method already receives the full activation tensor. Adding per-token statistics
requires a few additional accumulators in the dataclass and a few lines in
`update()`. The memory overhead is small (197 floats per layer of interest, for
4 layers = ~3 KB).

**Risk:** Very low. The change is additive and does not modify any existing
metric. The per-token statistics are collected only for the 4 layers of interest
(blocks 8-11 fc1), so the runtime impact is negligible.

---

## Warning 5: The LLM.int8() Outlier Topology May Be a Scale Effect

**Source:** Dettmers et al. (2022), *LLM.int8()* (arXiv:2208.07339). The
abstract states the method was tested on models up to 175B parameters. The paper
describes "emergent features in transformer language models that dominate
attention and transformer predictive performance."

**Verification against the source:** Confirmed. The LLM.int8() paper studies
models from 6.7B to 175B parameters. The word "emergent" in the abstract is
significant: it implies that the outlier topology (sparse, channel-persistent,
extreme magnitude) is a property that emerges at scale. ViT-B/16 at 86M
parameters is two orders of magnitude smaller than the smallest model in the
LLM.int8() study.

This does not mean your measurement is wrong. It means the claim "ViTs are
different from LLMs" may be confounded by scale. A 6.7B-parameter ViT (if one
existed) might show the same sparse-outlier topology as a 6.7B-parameter LLM.
The difference you observe may be a small-model phenomenon, not a vision-vs-language
phenomenon.

**How this applies to the codebase:** This is a framing and interpretation
issue, not a code issue. Your measurement pipeline is correct. The warning
affects how you write up the results.

**Severity:** Low for the experimental pipeline. Medium for the thesis
contribution framing. If the contribution is framed as "ViTs have a different
outlier topology than LLMs," a reviewer can point out the scale confound. If the
contribution is framed as "ViT-B/16 at 86M parameters has a specific outlier
topology that implies a specific routing policy," the claim is scoped to what
you actually measured.

**Proposed change:** No code changes. Update the framing in the advisor
touchpoint doc and the thesis writeup:

- Replace "ViTs are different from LLMs" with "ViT-B/16 at 86M parameters shows
  a different outlier topology than the 6.7B-175B LLMs studied in Dettmers et
  al. (2022)."
- Add a sentence to the limitations section: "We cannot distinguish whether the
  observed outlier topology is a property of vision transformers generally or a
  property of this specific model scale. The LLM.int8() paper studied models two
  to three orders of magnitude larger. The outlier topology may converge to the
  sparse, channel-persistent pattern at larger ViT scales."
- If time permits (not in the July sprint), run Experiment 1 on ViT-L/16 (307M
  parameters) to get a second data point on the scale question.

---

## Warning 6: The Calibration/Evaluation Overlap Is a Methodological Weakness

**Source:** Bondarenko et al. (2021), *Understanding and Overcoming the
Challenges of Efficient Transformer Quantization* (arXiv:2109.12948). The
abstract does not explicitly address calibration/evaluation splits, but the
paper's methodology (Section 3) uses a separate calibration set for quantization
parameter estimation. Nagel et al. (2021), *A White Paper on Neural Network
Quantization* (arXiv:2106.08295), Section 4 recommends using a held-out
calibration set.

**Verification against the source:** Confirmed as standard practice. The Nagel
et al. white paper explicitly recommends: "The calibration data should be
representative of the input distribution but distinct from the evaluation data
to avoid overfitting the quantization parameters." Bondarenko et al. use a
separate calibration set for their per-embedding-group quantization parameter
estimation.

**How this applies to the codebase:** Your Experiment 1 uses the ImageNet
validation split for both Pass 1 (per-channel statistics) and Pass 2 (outlier
counting). Experiments 2-4 are planned to evaluate accuracy on the same
validation split. The routing policy is derived from the same images used to
evaluate it.

The advisor touchpoint doc (Section 4.7) already flags this and proposes three
options. Option A (hold out ~20% of the validation set for accuracy evaluation)
is the recommended path.

**Severity:** Medium. The overlap primarily affects the borderline layers (the
`attn.qkv` cluster at 0.39%-0.52% routing fraction). The blocks 9-10 fc1
catastrophe (97-100% routing fraction) is far too large for data overlap to
explain. The concern is about the integrity of the borderline-layer policy
assignments, not about the main finding.

**Proposed change:** Implement Option A from the advisor touchpoint doc. This
requires changes to the data pipeline only.

**Implementation (changes to src/data_loader.py and experiment scripts):**

1. Add a `validation_split` parameter to `create_imagenet_val_loader`:

```python
def create_imagenet_val_loader(
    batch_size: int,
    data_dir: str = "./data/imagenet-val",
    max_images: int | None = None,
    validation_split: float | None = None,
    split_seed: int = 42,
) -> DataLoader:
    """
    Creates a DataLoader for the ImageNet validation set.

    Args:
        validation_split: If provided, a fraction in (0, 1) specifying the
            portion of the data to use. The split is deterministic (fixed seed,
            no shuffling) so repeated runs see the same subset.
    """
    _, transform = load_vit_model()
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    if validation_split is not None:
        num_total = len(dataset)
        num_val = int(num_total * validation_split)
        indices = list(range(num_total))
        rng = random.Random(split_seed)
        rng.shuffle(indices)
        dataset = Subset(dataset, indices[:num_val])

    if max_images is not None:
        dataset = Subset(dataset, range(min(max_images, len(dataset))))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, ...)
```

2. Update `run_experiment1_mapping.py` to accept a `--calibration-split`
   argument (default 0.8, meaning 80% of images used for statistics).

3. Update Experiments 2-4 to use `validation_split=0.2` (the held-out 20%) for
   accuracy evaluation.

4. Document the split explicitly in all output artifacts (JSON metadata,
   CSV headers).

**Risk:** Low. The split is deterministic (fixed seed), so results are
reproducible. The main cost is re-running Experiment 1 on the 80% calibration
split, which takes a few hours on the RTX 3070. The 50,000-image thesis-print
run should use this split from the start.

---

## Summary: Prioritized Action Items

| Priority | Warning | Action | Effort | Blocks |
|---|---|---|---|---|
| 1 (now) | #6: Calibration/eval overlap | Implement data split in `src/data_loader.py` | ~1 hour | Experiments 2-4 accuracy numbers |
| 2 (now) | #1: Symmetric quantizer for post-GELU | Add `quantize_per_tensor_asymmetric` to `src/quantization.py` | ~2 hours | Experiment 2 accuracy, Experiment 3 interpretability |
| 3 (Week 1) | #4: Token-driven vs channel-driven explosion | Add per-token statistics to Experiment 1 hooks | ~3 hours | Path B (SmoothQuant) feasibility |
| 4 (Week 1) | #3: SmoothQuant dense-outlier viability | Feasibility check using existing Experiment 1 data | ~1 day | Path B go/no-go decision |
| 5 (Week 2) | #2: Operation-specific quantizers | Add `--quantizer-aware` flag to Experiment 3 | ~2 hours | Experiment 3 sensitivity ranking confidence |
| 6 (writeup) | #5: Scale-effect confound | Update framing in thesis and advisor doc | ~1 hour | Contribution framing |

Items 1 and 2 are prerequisites for any accuracy number collected from Week 2
onward. They should be implemented before Experiment 3 runs. Items 3 and 4
determine whether Path B is viable and should be completed before the Week 3
path decision. Items 5 and 6 are documentation changes that can be done during
the writeup phase.

---

## What Does Not Need to Change

The following aspects of the codebase are sound and should not be modified in
response to these warnings:

1. **The two-pass exact statistics algorithm** (`src/hooks.py`,
   `LayerMomentAccumulator`). The Chan/Welford merge in float64 is more
   numerically stable than the batch-averaging approach used in most PTQ papers.
   Do not replace it with a streaming approximation.

2. **The per-column routing fraction as the primary metric.** The LLM.int8()
   paper, PTQ4ViT, and FQ-ViT all confirm that structured (column-wise) routing
   is the correct cost model for cuBLAS INT8 GEMMs. Do not switch to a
   per-value metric as the primary signal.

3. **The fixed 6.0 threshold.** This is a constant from the LLM.int8() paper
   that maps to INT8's dynamic range. It is not fit to your data. The
   calibration/eval overlap concern (Warning 6) does not apply to this threshold
   because it is externally derived.

4. **The decision to use timm ViT rather than torchvision.** The 49
   independently hookable linear layers are essential for per-layer analysis.
   The fused `attn.qkv` measurement is a known limitation (advisor doc Section
   4.9), not a bug.

5. **The Experiment 3 protocol (quantize one layer at a time).** While isolated
   sensitivity does not capture compound error propagation, it is the standard
   approach in the literature (PTQ4ViT, HAWQ) and is the right first step. The
   compound effect is measured in Experiments 2 and 4.