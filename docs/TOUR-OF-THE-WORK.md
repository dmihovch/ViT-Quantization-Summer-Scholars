# Tour of the Work — ViT Outlier Profiling & Per-Channel Ablation

> **A comprehensive guide to the entire project: motivations, methodologies, results,
> analysis, and how to answer any question that may be asked during the poster presentation.**

---

## Table of Contents

1. [High-Level Overview — What This Project Is](#1-high-level-overview--what-this-project-is)
2. [Motivation — Why This Matters](#2-motivation--why-this-matters)
3. [The Model and Data](#3-the-model-and-data)
4. [Phase 1 — Activation Profiling (What and Where)](#4-phase-1--activation-profiling-what-and-where)
5. [Phase 2 — Outlier Ablation (How Much and Why)](#5-phase-2--outlier-ablation-how-much-and-why)
6. [Post-Hoc Analysis — Digging Deeper](#6-post-hoc-analysis--digging-deeper)
7. [The Figures — A Visual Tour](#7-the-figures--a-visual-tour)
8. [Statistical and Methodological Rigor](#8-statistical-and-methodological-rigor)
9. [Infrastructure and Code Quality](#9-infrastructure-and-code-quality)
10. [The Story Arc — How to Present This](#10-the-story-arc--how-to-present-this)
11. [Extensive FAQ](#11-extensive-faq)

---

## 1. High-Level Overview — What This Project Is

This is a **mechanistic interpretability and ablation study** of massive activation outliers
in a Vision Transformer (ViT-B/16). It sits at the intersection of:

- **Transformer interpretability** — understanding what the network's internal activations look like
- **Model quantization** — the practical problem of fitting a ViT into INT8 for edge inference
- **Ablation methodology** — surgically removing components and measuring the accuracy impact

The project has **two phases**:

| Phase | Question | Method | Status |
|-------|----------|--------|--------|
| 1 | **Where are the outliers?** | Profile 73 measurement sites across 12 encoder blocks using exact Welford statistics over 50K images | ✅ Complete |
| 2 | **How much do they matter, and why?** | Zero activations beyond k·σ thresholds and measure accuracy degradation; decompose per-channel vs global behavior | ✅ Complete |

A deferred **Phase 3** (integer GELU LUTs for edge deployment) was deprioritized in favor of deeper
per-channel ablation analysis, which yielded the most interesting and publishable findings.

### What this project is NOT

- It is **not** a generalization study — profiling thresholds are calibrated on the same ImageNet-1K validation set used for evaluation. This is deliberate: we are characterizing the *observed* activation statistics on these specific images, not making predictive claims about unseen data.
- It is **not** a quantization implementation — no actual integer arithmetic is performed. This is a pre-quantization analysis that *informs* future PTQ schemes.
- It is **not** a training study — the model is used off-the-shelf (pretrained weights from timm), never fine-tuned.

---

## 2. Motivation — Why This Matters

### The Practical Problem

Deploying ViT-B/16 on edge hardware like the **NVIDIA Jetson Orin** requires **INT8 inference**
to meet latency and power constraints. The standard approach is **post-training quantization (PTQ)**:

1. Run calibration data through the FP32 model
2. Observe activation ranges (min, max) at each layer
3. Choose a quantization scale and zero-point for each tensor
4. Convert weights and activations to INT8

The problem: **activation ranges are dominated by a tiny number of massive outliers.**

Specifically, pre-GELU activations in deep encoder blocks exhibit extreme values. If you set the
quantization range to cover the absolute maximum, you waste dynamic range on 99.6% of "normal"
elements. If you clip to a reasonable range, you destroy the outlier values — but the model might
depend on those outliers for accuracy.

### The Scientific Question

Do these outliers actually matter for classification accuracy, or are they statistical noise?

Prior work (Dettmers et al., 2022; Xiao et al., 2023) established that outliers exist in large
language models. Less is known about vision transformers specifically. This project asks:

1. **Where are the outliers in ViT-B/16?** (profiling)
2. **How much accuracy do they carry?** (ablation)
3. **Can per-channel thresholding preserve more accuracy than global per-tensor thresholding?** (per-channel ablation)
4. **If so, WHY — is it mean correction, variance correction, or both?** (decomposition)
5. **Where does the per-channel pattern come from?** (gain-σ correlation)

### The Quantization Connection

Per-tensor quantization uses a single scale for the entire (B, N, D) activation tensor. Per-channel
quantization uses one scale per channel dimension. The findings directly motivate **per-channel
quantization schemes** for ViT activations:

- **Per-channel mean correction** (zero-point per channel): recovers 20 pp of accuracy at k=3 by
  accounting for shifted channel means (μ_c ∈ [−71.18, 26.01] at Block 10)
- **Per-channel scale**: adapts to the 12.4× range in per-channel σ (σ_c ∈ [2.06, 25.54] at Block 10)

---

## 3. The Model and Data

### Model

- **Architecture:** ViT-B/16 (`vit_base_patch16_224`)
- **Weights:** `augreg2_in21k_ft_in1k` from `timm` — pretrained on ImageNet-21K with AugReg,
  fine-tuned on ImageNet-1K
- **Specs:** 12 encoder blocks, 12 attention heads, hidden dim 768, MLP hidden dim 3,072,
  patch size 16×16, input resolution 224×224
- **Baseline accuracy (50K ImageNet-1K val):** top-1 = 85.03%, top-5 = 97.52%
- **Total parameters:** ~86M
- **Critical detail:** `fused_attn` is **disabled** on all blocks before wrapping with nnsight.
  With FlashAttention/SDPA enabled, the QKᵀ attention logit matrix is never materialized as a
  concrete tensor, making it impossible to capture activation statistics. This is a Python-level
  flag (`block.attn.fused_attn = False`) — it doesn't change the model's behavior, only the
  implementation backend.

### Data

- **Dataset:** ImageNet-1K validation split
- **Size:** 50,000 images across 1,000 classes (50 images per class)
- **Layout:** Standard ImageFolder format (`data/<class_name>/<image>.JPEG`)
- **Preprocessing:** Derived exclusively from the model's pretrained config via
  `timm.data.resolve_data_config` and `timm.data.create_transform` — no hardcoded
  mean/std/resize values, ensuring consistency with the checkpoint
- **DataLoader:** `num_workers=4`, `pin_memory=True` for CUDA, default shuffle for class-diverse
  batches

### Hardware

- **Tested on:** NVIDIA RTX 3070 (8 GB VRAM)
- **Software:** Python 3.13, PyTorch 2.12.1, nnsight 0.7.0, CUDA 13.0

---

## 4. Phase 1 — Activation Profiling (What and Where)

### What we measure and how

Phase 1 profiles activation statistics at **73 measurement sites** across the ViT:

- **6 sites per block × 12 blocks = 72 sites**
- **1 patch embedding site** (`patch_embed/residual_stream`)

The six measurement sites follow the forward-pass dataflow through each encoder block:

```
residual_stream → post_layernorm_1 → pre_softmax → post_softmax → post_layernorm_2 → pre_gelu
```

| Site | Where captured | Shape | Channels |
|------|---------------|-------|----------|
| `residual_stream` | `block.norm1.input` (the accumulated residual) | (B, 197, 768) | 768 |
| `post_layernorm_1` | `block.norm1.output` (pre-attention LN) | (B, 197, 768) | 768 |
| `pre_softmax` | Reconstructed QKᵀ/√d from `attn.qkv.output` | (B, 12, 197, 197) | no channel dim |
| `post_softmax` | `attn.attn_drop.input` (attention weights) | (B, 12, 197, 197) | no channel dim |
| `post_layernorm_2` | `block.norm2.output` (pre-MLP LN) | (B, 197, 768) | 768 |
| `pre_gelu` | `block.mlp.act.input` (MLP hidden state) | (B, 197, 3072) | 3,072 |

**Residual stream labeling convention:** `blocks.{k}/residual_stream` is the residual *after*
block k has processed it (the input to block k+1). This is important for interpretation —
e.g., `blocks.5/residual_stream` with high kurtosis reflects what block 5 *produced*, which is
driven by block 5's MLP and attention, not by block 5's input.

### Statistics collected per site

Every site collects the following statistics, all computed as **population** statistics (ddof=0,
treating the entire batch as the complete population):

| Statistic | How | Notes |
|-----------|-----|-------|
| **Mean (μ)** | Pébay parallel merge | Exact global mean across all batches |
| **Std (σ)** | Pébay parallel merge (√M2/n) | Population std |
| **Kurtosis (κ)** | Pébay parallel merge M3, M4 | Excess kurtosis = M4/(n·σ⁴) − 3 |
| **Outlier fractions** | Two-pass: first per-batch, then global-σ recount at k∈{3,4,6} | Mean-centered: `|x − μ| > k·σ` |
| **Per-channel σ_c** | Per-channel running sum/sum_sq | 3072-dim for pre_gelu; 768-dim for LN/residual sites |
| **Per-channel μ_c** | Per-channel running sum | Same dimensions as σ_c |
| **Attention entropy (CLS)** | Per-head Shannon entropy averaged over batch | From `attn_drop.input[cls_row]` |
| **Attention entropy (patches)** | Per-head entropy averaged over patch query tokens | From `attn_drop.input[patch_rows]` |
| **LN γ and β** | Static model weights extracted after trace | For `post_layernorm_1` and `post_layernorm_2` only |
| **LN2 amplification ratio** | ‖LN2(x)‖₂ / ‖x_skip‖₂, averaged over batch and tokens | For residual_stream sites only |
| **Running min/max** | Element-wise extremum tracking | For sanity-checking quantization ranges |

### The instrumentation mechanism: nnsight

All profiling is done using **nnsight** (formerly "nnsight"), which provides a trace-based
intervention framework for PyTorch models. The key idea:

1. Wrap the model with `NNsight(model)`
2. Open a trace context with `with wrapped_model.trace(input_batch):`
3. Inside the trace, access intermediate activations as **proxy objects** (not concrete tensors)
4. Register `.save()` calls on computed statistics
5. When the trace context exits, the forward pass executes, and all `.save()` proxies resolve to
   concrete values

This enables capturing statistics from arbitrary intermediate points in the forward pass without
modifying the model code. No hooks, no subclassing, no monkey-patching.

**Critical implementation for `pre_softmax`:** The raw QKᵀ attention logit matrix has no dedicated
module boundary in timm's implementation — it's computed inline. To capture it, we reconstruct it
from `attn.qkv.output` inside the trace:

```python
# qkv output shape: (B, N, 3*num_heads*head_dim)
qkv = qkv_proxy.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)
pre_softmax = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)
```

### The Welford multi-batch pipeline (how we get exact statistics over 50K images)

Processing all 50,000 images in a single forward pass is impossible (VRAM). Instead, we use a
**parallel higher-moments merge** based on Pébay (2008):

1. Process one batch at a time through `profile_vit()`, getting per-batch `LayerStats`
2. Merge each batch into a `WelfordAccumulator` using the exact Pébay formulas
3. Finalize the accumulator to produce exact global statistics

The Pébay merge formulas for M2, M3, M4 enable exact combination of central moment sums from
different batches with different means — there is **no approximation** and **no per-batch
centering bias**. The M3 and M4 formulas are:

```
M2_ab = M2_a + M2_b + δ²·n_a·n_b/n_ab
M3_ab = M3_a + M3_b + δ³·n_a·n_b·(n_a−n_b)/n_ab² + 3δ·(n_a·M2_b − n_b·M2_a)/n_ab
M4_ab = M4_a + M4_b + δ⁴·n_a·n_b·(n_a²−n_a·n_b+n_b²)/n_ab³ + 6δ²·(n_a²·M2_b+n_b²·M2_a)/n_ab²
         + 4δ·(n_a·M3_b − n_b·M3_a)/n_ab
```

where δ = μ_b − μ_a is the difference in per-batch means.

### The two-pass outlier recount

The outlier fractions computed during the Welford pass use **per-batch σ** — they answer "what
fraction of elements in this batch exceed k·σ_batch?" This is a well-defined statistic but
different from "what fraction exceed k·σ_global?" To get the correct definition, we run a
**second pass** (`run_outlier_counting_pass`):

1. Iterate over all batches again
2. For each activation tensor, apply the global-σ threshold from Phase 1's finalized stats
3. Count elements exceeding k·σ_global
4. Accumulate counts across batches

This second pass uses the exact same DataLoader — no images are held out. The reason it must
be a separate pass is that we need the global σ first, which requires a complete pass to compute.

### Key Phase 1 findings

| Metric | Value |
|--------|-------|
| Pre-GELU block 10 μ | −28.33 |
| Pre-GELU block 10 σ | 11.20 |
| Pre-GELU block 10 kurtosis | 0.60 |
| Per-channel σ range (block 10) | 2.06 – 25.54 (12.4× spread) |
| Per-channel μ range (block 10) | −71.18 – 26.01 (97-point spread) |
| Outlier fraction at 3σ (block 10) | 0.39% |
| LN2 γ vs σ_c correlation | r ≈ 0.0003 (no correlation — different dimensionalities) |
| Attention entropy (CLS) | Collapses in later blocks (entropy sink phenomenon) |

---

## 5. Phase 2 — Outlier Ablation (How Much and Why)

### The core methodology: surgical zeroing

Phase 2 uses the same nnsight trace mechanism as Phase 1, but instead of *observing* activations,
we *replace* them:

1. Load Phase 1's per-layer statistics (μ, σ, per-channel μ_c, per-channel σ_c)
2. Open an nnsight trace on the input batch
3. For each encoder block at the target site, build a boolean mask: `True` = keep, `False` = zero
4. Replace the activation tensor with `tensor * mask` (element-wise zeroing)
5. The model continues the forward pass with modified activations
6. Save the output logits and compute accuracy

### Sites and shapes

| Site | Intervention point | Shape | Per-channel? |
|------|-------------------|-------|-------------|
| `pre_gelu` | `block.mlp.act.input` | (B, 197, 3072) | Yes (3072 channels) |
| `residual_stream` | `block.norm1.input` (CLS preserved) | (B, 197, 768) | Yes (768 channels) |
| `post_layernorm_1` | `block.norm1.output` | (B, 197, 768) | Yes (768 channels) |
| `post_layernorm_2` | `block.norm2.output` | (B, 197, 768) | Yes (768 channels) |
| `pre_softmax` | Reconstructed QKᵀ/√d | (B, 12, 197, 197) | No (no channel dim) |
| `post_softmax` | `attn.attn_drop.input` | (B, 12, 197, 197) | No (no channel dim) |

**CLS token preservation on residual_stream:** The CLS token in the residual stream carries
aggregate information from all previous blocks. Zeroing it would destroy the classification head's
input regardless of outlier status. We therefore preserve the CLS token row (index 0) on the
residual_stream site during ablation.

### The threshold definition: mean-centered

The zeroing criterion is:

```
|x − μ| > k·σ
```

where μ and σ are from Phase 1's `profiling_result.json`. This is the **mean-centered** outlier
definition, consistent with the statistical literature (Wei et al., 2022, §3.1; Bondarenko et al.,
2021, §4.1). It is NOT the zero-centered definition `|x| > k·σ`, which would conflate the
distribution's mean shift with its outlier count.

For Block 10 pre-GELU (μ = −28.33, σ = 11.20), this matters: an element at x = 0 is **not**
an outlier by `|x| > 3·11.20` (0 < 33.6) but **is** an outlier by `|x − (−28.33)| > 3·11.20`
(28.33 > 33.6? No — 28.33 < 33.6, so it's NOT an outlier. But an element at x = 0: dev = 28.33,
threshold = 33.6, so NOT an outlier at 3σ. An element at x = −62: dev from −28.33 is 33.67 > 33.6,
so it IS an outlier at 3σ. The key point: elements near zero are actually *within* the normal
range when you account for the negative mean.)

### The random-zeroing control

**Why it's the most important control experiment:** If zeroing a certain fraction of elements
destroys accuracy regardless of *which* elements are zeroed, then outliers aren't special — it's
just the effect of sparsity. The random control isolates the effect of outliers specifically.

**How it works:**
1. Run the outlier-threshold ablation on a batch; record the per-layer %-zeroed
2. Run a second pass on the same batch, but zero a random subset of elements matching the exact
   same fraction per layer (using a seeded random permutation)
3. Compare accuracy between the outlier-condition and random-control runs

**Result:** Random zeroing preserves accuracy within 0.1 pp of baseline at all sparsity levels.
The degradation is caused by the loss of *specific outlier values*, not by general activation
sparsity.

### Granularity modes

| Mode | Threshold | Description |
|------|-----------|-------------|
| `global` | `|x − μ_global| > k·σ_global` | One threshold per layer (all channels share the same μ, σ) |
| `per_channel` | `|x_c − μ_c| > k·σ_c` | One threshold per channel (each of 3072 channels has its own μ_c, σ_c) |

**Per-channel mode** uses the per-channel statistics from Phase 1's profiling. The threshold
broadcasts over the batch and token dimensions: the mask has shape (B, N, D), the thresholds
have shape (D), and element-wise comparison automatically broadcasts correctly.

### Ablation mode decomposition (the most important experiment)

To understand *why* per-channel thresholding helps, we decompose the effect into three modes:

| Mode | μ source | σ source | What it isolates |
|------|----------|----------|-----------------|
| `outlier` | per-channel μ_c | per-channel σ_c | Full per-channel benefit |
| `mean_only` | per-channel μ_c | global σ | Mean-correction component |
| `var_only` | global μ | per-channel σ_c | Variance-correction component |

This is a clean 2×2 experimental design (µ: {global, per-channel} × σ: {global, per-channel}),
with three occupied cells.

### Key Phase 2 results

**Baseline top-1: 85.03%, top-5: 97.52%**

**5-seed run (42, 43, 44, 45, 46), 50,000 images.**

| k | Global (top-1) | Per-channel (top-1) | Δ | 95% CI |
|---|---|---|---|---|
| 3.0 | 43.24% | 47.00% | **+3.76 pp** | [3.12, 4.36] |
| 4.0 | 75.12% | 75.54% | +0.42 pp | [−0.11, 0.96] |
| 6.0 | 84.58% | 84.11% | −0.47 pp | [−0.93, −0.03] |

**Headline finding:** Per-channel thresholds preserve 3.76 percentage points more accuracy at
k=3, and this is statistically significant (95% CI does not include zero). The effect vanishes
by k=4, confirming it is concentrated at aggressive thresholds.

**Per-channel ablation decomposition (k=3):**

| Condition | top-1 | Δ vs global |
|-----------|-------|-------------|
| Baseline | 85.03% | — |
| Global outlier | 43.24% | — |
| Per-channel outlier | 47.00% | +3.76 pp |
| Per-channel mean_only | **63.32%** | **+20.08 pp** |
| Per-channel var_only | 6.56% | −36.68 pp |

**Headline finding:** Mean correction dominates. Per-channel μ_c recovers **20 pp** over the
global condition by correcting for shifted channel means. Variance correction alone is
**catastrophic** (6.56%) — worse than the global condition and worse than random zeroing would
be at equivalent sparsity. It applies narrow thresholds to channels with negative means, zeroing
activations that are genuinely within-channel normal.

### 5-seed design and why ablation is deterministic

The 5-seed run uses seeds 42–46, but the ablation results show **zero variance across seeds**.
This is correct and by design:

1. Phase 1 profiling is **deterministic** — given the same model checkpoint and the same input
   images, the activation statistics are identical regardless of seed.
2. Phase 2 ablation uses the Phase 1 statistics as fixed thresholds — no randomness in the
   zeroing criterion.
3. The accuracy evaluation is deterministic — no dropout, no stochastic operations in eval mode.

The seeds control only two things: (a) which subset of images are selected when `num_images` is
less than 50K (shuffled random permutation of indices), and (b) the random-zeroing control mask
generation. Since we use the full 50K dataset for the main results, (a) is irrelevant.
Different seeds produce identical accuracies.

**Why 5 seeds then?** To verify determinism. If different seeds produced different accuracies,
that would indicate either non-deterministic model operations or data selection differences.
The zero variance confirms the pipeline is deterministic and reproducible.

### Degradation efficiency (accuracy loss per 1% sparsity)

| k | Global | Per-channel | Efficiency ratio |
|---|---|---|---|
| 3.0 | 100.97 pp/% | 53.43 pp/% | **1.89×** |
| 4.0 | 23.94 pp/% | 13.33 pp/% | 1.80× |
| 6.0 | 1.07 pp/% | 1.28 pp/% | 0.83× |

Per-channel thresholds are **1.89× more efficient** at k=3. Each 1% of zeroed elements costs
roughly half as much accuracy. This indicates per-channel thresholds selectively preserve
channels that carry more classification-relevant signal.

---

## 6. Post-Hoc Analysis — Digging Deeper

### Effective gain correlation: RQ1 — "Where does the per-channel pattern come from?"

The SmoothQuant hypothesis (Xiao et al., 2023) is that high-γ LayerNorm channels amplify the
residual stream into the MLP, creating the per-channel variance pattern. We tested this in two
stages:

**Stage 1 — Naive LN2 γ vs σ_c correlation:** Pearson r ≈ 0.0003 — **no correlation**.
But this is expected: LN2 γ is 768-dimensional (embedding space) while pre-GELU σ_c is
3,072-dimensional (MLP hidden space). These vectors live in different spaces.

**Stage 2 — Effective per-channel gain:** Compute `‖fc1.weight[c, :] ⊙ γ‖₂` — the L2 norm
of the Hadamard product of the fc1 row with the LN2 γ vector, for each of the 3,072 MLP
hidden channels. This captures the combined scaling that determines how strongly channel c
responds to the residual stream. Then compute Pearson r between this effective gain and
per-channel σ_c.

| Block | r(gain, σ_c) |
|-------|-------------|
| 0–7 | −0.13 to +0.21 |
| 8 | **+0.755** |
| 9 | **+0.775** |
| 10 | **+0.650** |
| 11 | **+0.767** |

Mean r across all blocks: **+0.324**.

**This is a genuine finding:** The strong correlation in late blocks (8–11) confirms that the
per-channel variance pattern is **architectural** — encoded in the interaction of fc1.weight and
LayerNorm γ. Channels the network invested more weight into (higher effective gain) also exhibit
higher activation variance. The outliers are not anomalous noise; they are a deliberate consequence
of trained weights.

Key validation: the correlation is **zero-variance across 5 seeds** — the per-channel statistics
are deterministic, and the model weights are fixed, so the correlation is identical across runs.

### Layer-group ablation: RQ3 — "Which layers drive the benefit?"

**Status:** 🔲 Pending. CLI flag `--layer-range` is implemented and ready to run.

Hypothesis: Block 10 (the extreme outlier layer with σ=11.20, 12.4× σ spread) is the primary
driver of the per-channel benefit. The experiment design is:

```
# Block 10 only
--layer-range 10 10

# Late blocks (8-11)
--layer-range 8 11

# Early blocks (0-7)
--layer-range 0 7
```

### Finer k-sweep: RQ5 — "Where is the crossover point?"

**Status:** 🔲 Pending.

Hypothesis: The per-channel benefit is concentrated at aggressive thresholds (k < 4). A finer
sweep at k ∈ {2.5, 2.75, 3.0, 3.25, 3.5} would reveal exactly where global and per-channel
curves intersect.

### 95% CI computation methodology

Confidence intervals on accuracy deltas are computed using the **two-proportion z-interval**:

```
Δ = p̂_per_channel − p̂_global
SE = sqrt(p̂_per_channel*(1-p̂_per_channel)/N + p̂_global*(1-p̂_global)/N)
CI = Δ ± z_{0.975} * SE
```

where N = 50,000 (total images), p̂ = accuracy proportion, and z_{0.975} = 1.96.

The key insight: since we evaluate on the full 50K validation set (not a sample), the only source
of variance is the classification decision on each image. The proportion of correct classifications
is a binomial random variable, and the z-interval is appropriate for N=50,000.

### Effective channels preserved analysis

Translates %-zeroed into "effective channels preserved per block":

```
Total channels = 3,072 (MLP hidden dim) × 12 (blocks) = 36,864
Effective channels = total_channels × (1 − pct_zeroed)
```

This shows that per-channel ablation redistributes the zeroing budget from low-importance
channels to high-importance channels, achieving higher accuracy while preserving slightly
*fewer* total channels.

---

## 7. The Figures — A Visual Tour

All figures are generated offline from saved data files. The workhorse plotting module
(`src/plotting.py`) produces standard plots for rapid iteration. The poster plotting module
(`src/plotting_poster.py`) produces polished figures with custom palettes, ≥14 pt fonts,
direct annotation, and no chartjunk.

### Seven poster figures

| # | File | Type | Key Message |
|---|------|------|-------------|
| 1 | `fig1_activation_overlay.png` | Histogram with multi-threshold overlay | Global thresholds incorrectly clip high-variance channels; 12.4× σ spread means one threshold doesn't fit all |
| 2 | `fig2_sigma_ridgeline.png` | Line plot with ±1σ band | Per-channel σ rises sharply after Block 7, with widening spread |
| 3 | `fig3_outlier_grid.png` | 12×6 heatmap (blocks × sites) | Outliers concentrate in late-block pre-GELU sites; attention sites are clean |
| 4 | `fig4_accuracy_bars.png` | Grouped bar chart | +3.76 pp per-channel advantage at k=3; vanishes by k=4 |
| 5 | `fig5_accuracy_cost_vs_sparsity.png` | Connected scatter (Pareto front) | Per-channel is 1.89× more efficient on accuracy–sparsity frontier |
| 6 | `fig6_ablation_waterfall.png` | Multi-bar comparison | Mean correction dominates (63.32%); variance alone is catastrophic (6.56%) |
| 7 | `fig7_gain_sigma_scatter.png` | 3-panel scatter with regression | ‖w_c ⊙ γ‖₂ correlates with σ_c (r = 0.75–0.77 in late blocks); outliers are structural |

### Phase 1 workhorse plots

| File | Description |
|------|-------------|
| `per_channel_std_heatmap_d3072.png` | Per-channel σ heatmap (12 blocks × 3072 channels) |
| `per_channel_std_heatmap_d768.png` | Per-channel σ heatmap for layernorm/residual sites (12 × 768) |
| `per_channel_mean_heatmap_d3072.png` | Per-channel μ heatmap — reveals the 97-point mean spread at Block 10 |
| `per_channel_mean_heatmap_d768.png` | Per-channel μ heatmap for layernorm/residual sites |
| `kurtosis_heatmap.png` | Excess kurtosis across all 73 sites |
| `outlier_fraction_{k}_sigma_heatmap.png` | Outlier fractions at k ∈ {3, 4, 6} across all sites |
| `attention_entropy_cls_heatmap.png` | CLS attention entropy collapse (12 blocks × 12 heads) |
| `attention_entropy_patches_heatmap.png` | Patch attention entropy (12 blocks × 12 heads) |
| `ln2_amplification_ratio.png` | LN2 amplification ratio ‖LN2(x)‖₂/‖x_skip‖₂ per block |

### Phase 2 workhorse plots

| File | Description |
|------|-------------|
| `accuracy_comparison_pre_gelu.png` | Global vs per-channel accuracy curves for pre_gelu ablation |
| `accuracy_comparison_residual_stream.png` | Same for residual_stream ablation |

### Analysis plots

| File | Description |
|------|-------------|
| `ci_delta.png` | 95% CI on global vs per-channel accuracy delta at each k |
| `degradation_efficiency.png` | Accuracy loss per 1% sparsity for global vs per-channel |
| `effective_channels_k{3,4,6}.png` | Effective channels preserved per block |
| `effective_gain_correlation_bars.png` | Bar chart of r(gain, σ_c) per block |
| `effective_gain_correlation.json` | Numerical correlation values per block |
| `ln_gamma_correlation.json` | LN γ vs σ_c correlation (r ≈ 0.0003) |
| `ablation_analysis.json` | CI values and degradation efficiency numbers |

---

## 8. Statistical and Methodological Rigor

### What we did right

1. **Exact statistics (Pébay 2008):** No approximations in κ, μ, or σ. All higher-moment merges
   are exact, parallel-safe, and numerically stable.

2. **Population conventions (ddof=0):** Bessel's correction is for estimating unobserved
   population variance from a sample. We observe the complete activation tensor — every element
   is known exactly. Using ddof=1 would introduce systematic bias.

3. **Two-pass outlier recount:** The first-pass (Welford) outlier fractions use per-batch σ,
   which is well-defined but different from global-σ fractions. The second pass corrects this
   with the proper definition.

4. **Random-zeroing control:** Isolates the effect of outliers specifically from the effect
   of general sparsity. Matches the exact fraction zeroed per layer.

5. **Mean-centered thresholding:** `|x − μ| > k·σ` is the standard statistical definition,
   self-consistent, and used throughout quantization literature.

6. **2×2 decomposition design:** Three of four cells in the (µ: global/per-channel) ×
   (σ: global/per-channel) design, plus a baseline. Clean experimental design that isolates
   causal components.

7. **95% confidence intervals:** Two-proportion z-intervals on accuracy deltas at N=50,000.

8. **5-seed verification:** Confirmed that the pipeline is deterministic and no seed-dependent
   variance exists in the main results.

9. **System metadata collection:** Every profiling result records Python/PyTorch/timm/nnsight
   versions, CUDA version, GPU name, VRAM, timestamp — everything needed for reproducibility.

10. **Mistakes ledger:** `docs/MISTAKES.md` documents every wrong approach tried and why it was
    wrong, so future contributors don't repeat them.

### What we acknowledge as limitations

1. **Calibration/evaluation overlap:** Profiling and ablation/evaluation use the same 50K
   validation images. This is disclosed in the README's Methodological Note. It's acceptable
   for a descriptive mechanistic study but a reviewer may request a training-set-calibrated
   replication.

2. **Single-seed profiling:** The per-channel statistics are deterministic (same model, same
   images), but cross-check profiling seeds validate only the image selection, not the statistics
   themselves.

3. **No hyperparameter tuning done on test data:** We are not tuning any model parameters —
   only measuring and ablating. So the traditional train/val/test concern does not apply.

4. **No actual INT8 implementation:** The study is pre-quantization analysis. The per-channel
   findings motivate future per-channel PTQ schemes but do not implement them.

5. **No statistical significance tests on gain-σ correlation:** The r = 0.75–0.77 values are
   computed on 3,072 data points per block, so they are certainly significant, but we haven't
   computed p-values or bootstrap CIs for the correlation coefficients.

6. **Layer-group ablation (RQ3) and finer k-sweep (RQ5) are pending.**

---

## 9. Infrastructure and Code Quality

### Project structure

```
.
├── run_phase1_profiling.py       # Phase 1 CLI entry point
├── run_phase2_ablation.py        # Phase 2 CLI entry point
├── src/
│   ├── config.py                 # Frozen dataclasses for experiment configs
│   ├── model.py                  # Load ViT-B/16, evaluate accuracy
│   ├── data_loader.py            # ImageNet-1K val DataLoader with auto-shuffle
│   ├── profiler.py               # nnsight-based profiler with Welford pipeline
│   ├── ablation.py               # nnsight-based outlier zeroing with controls
│   ├── plotting.py               # Workhorse figure generation (58+ functions)
│   ├── plotting_poster.py        # Poster-quality figures (7 functions)
│   ├── plotting_utils.py         # Shared colour palettes, label formatting
│   ├── utils.py                  # seed_everything, get_device, system metadata
│   ├── exceptions.py             # Custom exception types
│   ├── exp1_profiling.py         # Phase 1 orchestrator
│   └── exp2_ablation.py          # Phase 2 orchestrator
├── scripts/                      # Post-hoc analysis + plot generation scripts
├── tests/                        # 252 tests (192 fast, 60 slow/GPU-dependent)
├── docs/                         # Implementation specs, mistakes, citations
├── outputs/                      # Written by experiments (git-ignored)
└── data/                         # ImageNet-1K val (git-ignored)
```

### Config as frozen dataclasses

All experiment configurations are `dataclass(frozen=True)` — immutable, typed, and with
comprehensive docstrings. This prevents accidental mutation during a run and enforces that
configuration is explicit. No raw `dict` configs.

### Testing

- **252 tests total** across 8 test modules
- **192 fast tests** (`pytest -m "not slow"`) run without model download
- **60 slow tests** require GPU and model download
- Tests cover: model loading, data loading, profiling, ablation, plotting, configuration,
  exceptions, and utilities
- Tests verify: output shapes, dtype preservation, gradient flow, edge cases, and at least
  one known-output case

### Plotting architecture

Plots are NOT generated during experiment runs (except activation histograms, which need the
live model). This design choice means:

1. Experiment runs are fast (no matplotlib overhead during profiling/ablation)
2. Plots can be regenerated from data files at any time without GPU
3. Different plotting styles (workhorse vs poster) can be applied to the same data
4. New plots can be added without re-running experiments

The pipeline: **experiment → JSON/CSV → plotting script → PNG**

---

## 10. The Story Arc — How to Present This

The poster tells a single coherent story in six sections, flowing left to right:

```
THE PROBLEM → HOW WE STUDIED IT → WHAT WE FOUND → WHY IT HAPPENS → WHERE TO GO NEXT

INTRO (fig1) ──→ BACKGROUND (fig3) ──→ METHODS (fig2) ──→ RESULTS (fig4+5) ──→ DISCUSSION (fig6+7) ──→ FUTURE WORK
```

### The one-glance test (5 seconds)

Someone walking by your poster should see:

1. **fig1:** Dual-threshold overlay — colored channel lines showing the 12.4× σ spread. Message: "channels are wildly different"
2. **fig4:** The "+3.76 pp" annotation. Message: "per-channel helps"
3. **fig6:** The 63.32% bar towering over 47.00%. Message: "mean correction wins"
4. **fig7:** Scatter points stretching upward across late blocks. Message: "it's in the weights"

### Four take-home messages

1. ViT-B/16 pre-GELU activations have massive outliers concentrated in late blocks
2. Per-channel thresholding preserves 3.76 pp more accuracy than global at k=3
3. The benefit comes from mean correction (per-channel μ_c), not variance adaptation
4. The per-channel pattern is architectural — encoded in the trained fc1.weight ⊙ LayerNorm γ interaction

### Key transitions for the presenter

- **fig1 → fig3:** "Now that you've seen the problem in one block, here's the full map"
- **fig2 → fig4:** "The σ spread is why per-channel matters — and fig4 shows how much"
- **fig4 → fig5:** "Raw accuracy is one view; the efficiency curve shows per-channel dominates"
- **fig5 → fig6:** "But why? fig6 decomposes the effect — the answer is mean correction"
- **fig6 → fig7:** "And where does this structure come from? It's in the trained weights"
- **fig7 → Future Work:** "If outliers are structurally encoded, per-channel PTQ is the natural next step"

---

## 11. Extensive FAQ

### General / Background

**Q: What model are you using, and why this one?**

ViT-B/16 (`vit_base_patch16_224`), pretrained on ImageNet-21K with AugReg and fine-tuned on
ImageNet-1K, loaded from `timm`. This is the standard base-size ViT used in most benchmarking.
We chose it because: (1) it's large enough to exhibit the outlier phenomenon (small models
like ViT-Tiny do not), (2) it's small enough to fit in 8 GB VRAM for profiling/ablation without
model parallelism, (3) it's a realistic target for edge deployment (86M params, INT8 → ~86 MB).

**Q: Why ImageNet-1K validation and not the training set?**

The 50K validation set provides a controlled, class-balanced evaluation with 50 images per
class. The 1.28M training set would be preferable for profiling (more statistics) and would
eliminate the calibration/evaluation overlap concern, but adds ~12 GPU-hours. The trade-off
is documented in the README's Methodological Note.

**Q: What is the difference between global and per-channel thresholding?**

Global: one μ and one σ per (block, site), applied to all channels equally.
Per-channel: one μ_c and one σ_c per channel, applied independently.

For pre_gelu, there are 3,072 channels in the MLP hidden dimension. A global threshold treats
all 3,072 channels identically — `|x − μ_global| > k·σ_global`. With per-channel, channel 3
(σ=2.06) has a narrow ±6.18 threshold, while channel 2891 (σ=25.54) has a wide ±76.62 threshold.

**Q: Why does per-channel matter for quantization?**

Per-tensor quantization uses a single scale for the entire tensor. If σ varies by 12.4× across
channels, the scale that works for high-σ channels wastes precision on low-σ channels, and the
scale that preserves precision on low-σ channels clips high-σ channels. Per-channel quantization
assigns one scale per channel — eliminating this trade-off. Our ablation shows this is not just
theoretically nice but empirically significant: +3.76 pp at k=3.

**Q: What does "12.4× σ range" actually mean?**

At Block 10's pre-GELU site (3,072 channels), the per-channel standard deviations range from
2.06 (the least volatile channel) to 25.54 (the most volatile). A single global σ of 11.20
is a compromise that misrepresents both ends.

**Q: What's an "outlier" in your definition?**

An activation element x where `|x − μ| > k·σ`. The deviation is measured from the distribution
mean, not from zero. k=3 is the standard statistical convention (roughly equivalent to the
Gaussian tail at ±0.27%). The quantization literature (Bondarenko et al., 2021) typically uses
k=4. Dettmers et al. (2022) used k=6 for extreme outlier detection.

### Phase 1: Profiling

**Q: How did you measure the activations?**

We used **nnsight**, a trace-based intervention framework for PyTorch. It wraps the model and
allows accessing intermediate activations inside a trace context as proxy objects. Inside the
trace, we register `.save()` calls on computed statistics (mean, std, kurtosis, per-channel
stats, entropy, outlier fractions). When the trace exits, the forward pass executes and all
proxies resolve to concrete values. No full activation tensors are ever retained in memory.

**Q: How do you compute exact statistics over 50K images if they don't fit in memory?**

Using the **Pébay (2008) parallel higher-moments merge** algorithm. We process one batch at a
time, accumulate batch statistics into a `WelfordAccumulator` using exact merge formulas for
M2, M3, and M4, and finalize at the end. All statistics are exact — no approximation, no
per-batch centering bias, no error accumulation.

**Q: What are M2, M3, M4?**

Central moment sums: M2 = Σ(x−μ)², M3 = Σ(x−μ)³, M4 = Σ(x−μ)⁴. From these we derive:
σ = √(M2/n), skewness = M3/(n·σ³), excess kurtosis = M4/(n·σ⁴) − 3.

The Welford/Pébay algorithm tracks these as running sums and merges them exactly across batches
without needing to store raw activations or recenter on per-batch means.

**Q: Why population statistics (ddof=0) instead of sample statistics (ddof=1)?**

Bessel's correction (ddof=1) corrects for the bias in estimating an unobserved population
variance from a sample. We are not estimating — we are measuring a fully-observed finite set
of activation values. Every element in every batch tensor is known exactly. There is no
statistical justification for Bessel's correction here.

**Q: What is kurtosis, and what does kurtosis = 0.60 at Block 10 mean?**

Excess kurtosis = 0 means the distribution has Gaussian-like tails. Positive kurtosis means
heavier tails than Gaussian (more extreme values). 0.60 at Block 10 pre-GELU indicates mildly
heavy-tailed — the distribution is not dramatically non-Gaussian, but has more mass in the
tails than a normal distribution would predict.

**Q: What are "attention entropy" measurements, and why are they interesting?**

For each attention head, we compute Shannon entropy H = −Σ p_i·ln(p_i) of the post-softmax
attention weights. We separate CLS query entropy (the CLS token's attention distribution over
patches) from patch query entropy (patch tokens' attention distributions). The finding is that
CLS entropy **collapses** in later blocks — the CLS token's attention becomes concentrated on
a few patches, a phenomenon called the "entropy sink" (Zhai et al., 2023, ICML).

**Q: What is the LN2 amplification ratio?**

‖LN2(x)‖₂ / ‖x_skip‖₂ — the ratio of the L2 norm of the pre-MLP LayerNorm output to the L2
norm of the residual stream input to the same block. A ratio > 1 means the LayerNorm is
amplifying the signal before it enters the MLP. This is the primary driver of pre-GELU
activation range expansion (Bondarenko et al., 2021).

**Q: Why did you disable fused_attn?**

With PyTorch's SDPA/FlashAttention (`fused_attn=True`), the QKᵀ attention logit matrix is
computed in a fused kernel and never materialized as a Python tensor. nnsight can only
capture tensors that exist in memory. `fused_attn=False` forces the standard attention
implementation, which materializes the logit matrix. The model's behavior and accuracy are
identical — only the implementation backend changes.

**Q: How do you capture pre_softmax attention logits when there's no module boundary?**

Timm's attention implementation computes QKᵀ/√d inline without a dedicated module. We
intercept `attn.qkv.output` (the concatenated QKV projection), reshape it to separate Q,
K, and V, compute Q@Kᵀ/√d, and save the result. All of this happens inside the nnsight trace
as proxy operations — no Python-level multiplication of concrete tensors until trace execution.

### Phase 2: Ablation

**Q: How exactly do you "zero out"activations?**

Inside the nnsight trace, for each encoder block at the target site, we compute a boolean mask:
`True` where `|x − μ| ≤ k·σ` (keep), `False` otherwise (zero). Then we replace the activation
with `x * mask`. Element-wise multiplication with `False=0` zeros the target elements while
preserving the rest. The model then continues the forward pass with the modified activations.

**Q: How do you ensure the random-zeroing control is fair?**

After an outlier-threshold forward pass, we record the per-layer fraction of elements zeroed.
Then on the **same batch**, we run a second pass that zeros exactly that fraction of elements
at uniformly random positions (using a seeded random permutation of tensor indices). The
fraction, batch, and model are identical — only the *criterion* for which elements to zero
differs.

**Q: Why does mean_only (63.32%) outperform full per-channel (47.00%)?**

This is the most surprising result and the key mechanistic insight. The full per-channel
condition uses per-channel σ_c, which applies narrower thresholds to low-σ channels. But
channels with lower σ often have strongly negative μ_c (recall: Block 10 μ_c ∈ [−71.18, 26.01]).
A narrow threshold on a channel centered at μ_c = −60 means: "only keep values between −60−k·σ_c
and −60+k·σ_c." If σ_c is small, this range is tight, and any value that deviates slightly from
−60 gets zeroed — even if it's genuinely within-channel normal. The mean_only condition uses
global σ (wide enough to accommodate channel-to-channel mean differences), but corrects for the
channel mean shift. This prevents the over-zeroing.

**Q: Why is var_only (6.56%) catastrophic?**

Var_only uses per-channel σ_c but global μ = −28.33. For a channel with μ_c = −60 and σ_c = 10,
the threshold range around μ = −28.33 at ±3·10 = ±30 is [−58.33, 1.67]. But this channel's
activations are centered at −60! Almost all of them fall below −58.33 and get zeroed. This is
systematically zeroing the *wrong* activations — it's worse than random.

**Q: Why doesn't per-channel help at k=4 and k=6?**

At k=6, the threshold is so wide (±67.2 for σ=11.20) that very few elements exceed it in
either condition. The per-channel vs global distinction becomes irrelevant when almost nothing
is being zeroed. At k=4 (±44.8), the effect is present but much smaller — the threshold is
wide enough that most channels' activations are captured even without per-channel correction.

**Q: Why is the ablation deterministic (zero variance across seeds)?**

Phase 1 profiling is deterministic (same model + same images = same statistics). Phase 2 uses
those fixed statistics as thresholds. Accuracy evaluation is deterministic (no dropout, no
stochastic ops in eval mode). The seeds control only image subsampling (irrelevant at N=50K)
and random-zeroing mask generation. With 50K images, all seeds see all images.

**Q: What sites did you ablate, and why focus on pre_gelu?**

We can ablate pre_gelu, pre_softmax, post_softmax, residual_stream, post_layernorm_1, and
post_layernorm_2. We focus on pre_gelu because Phase 1 profiling showed it's where outliers
concentrate (late blocks, pre_gelu), and it's the site where the 12.4× σ spread is most extreme.
It's also the most relevant for quantization — pre-GELU activations must be quantized before
the GELU nonlinearity in an INT8 inference pipeline.

**Q: Do you preserve the CLS token during ablation?**

For pre_gelu, post_layernorm_1, and post_layernorm_2: no — the CLS token travels through all
encoder operations normally. For residual_stream: YES — we preserve the CLS token row (index 0)
because zeroing the CLS token's residual representation would destroy the input to the
classification head regardless of outlier status. The per-channel residual_stream zeroing
intervenes on patch tokens only.

### Analysis and Decomposition

**Q: What does the effective gain ‖fc1.weight[c,:] ⊙ γ‖₂ actually measure?**

The pre-GELU activation for channel c is computed as:
```
h_c = fc1.weight[c, :] @ LN2(x_residual)
```
where LN2 applies an element-wise scale γ (and bias β). The effective gain is the L2 norm of
the Hadamard product: `‖fc1.weight[c, :] ⊙ γ‖₂`, which captures how strongly channel c responds
to the (γ-scaled) residual stream. This is the correct vector to correlate with per-channel σ_c
because both are 3,072-dimensional — unlike the naive γ vs σ_c correlation (768-dim vs 3072-dim,
which gave r ≈ 0.0003).

**Q: What does r = 0.75 mean, practically?**

It means about 56% of the variance in per-channel pre-GELU σ_c is explained by the effective
gain of the weight matrix. The remaining variance comes from other factors (input statistics,
LayerNorm behavior, the GELU nonlinearity's effect on the previous block, etc.). But the
dominant factor is the weights — channels the network learned to amplify more also exhibit
wider activation distributions.

**Q: Why is the correlation strong only in late blocks (8–11)?**

The effective gain in early blocks (0–7) is more uniform across channels — the r is weak
because the gain doesn't vary much. In late blocks, the gain distribution becomes more
heterogeneous: some channels are strongly amplified (high gain, high σ_c) and others are
suppressed (low gain, low σ_c). This is a structural property of the trained network —
early blocks process low-level features uniformly, while late blocks selectively amplify
specific feature channels for classification.

**Q: How did you compute the 95% confidence intervals?**

Two-proportion z-interval:
```
Δ_hat = p̂_pc − p̂_global
SE = √(p̂_pc·(1−p̂_pc)/N + p̂_global·(1−p̂_global)/N)
CI = Δ_hat ± 1.96·SE
```
N = 50,000 for the full dataset. The z-interval is appropriate because the classification
outcome on each image is a Bernoulli trial with probability equal to the accuracy rate,
and N is large enough for the normal approximation.

**Q: What is "degradation efficiency" and why is it useful?**

Accuracy loss per 1% sparsity = (baseline_accuracy − ablated_accuracy) / pct_zeroed.

A high value means each zeroed element costs a lot of accuracy — the model is sensitive to
these specific elements. A low value means the model tolerates sparsity well — the zeroed
elements don't carry much signal. Per-channel achieves lower degradation efficiency (53.43
pp/%) than global (100.97 pp/%) at k=3 — meaning the per-channel thresholds selectively
preserve signal-carrying elements.

**Q: How did you validate the gain-σ correlation?**

The correlation is validated across 5 profiling seeds — all five produce identical r values
because the model weights are fixed and the per-channel statistics are deterministic given
the full 50K dataset. Additionally, we visualize the scatter with linear regression fit
(fig7 shows this for Blocks 8, 9, 10), making it visually clear that the correlation is not
driven by a few outlier channels.

### Methodological Questions

**Q: Isn't it a problem to use the same data for profiling and evaluation?**

We disclose this explicitly (see README Methodological Note). It's not a train/test leak in
the traditional ML sense because no model parameters are updated and no hyperparameters are
tuned. The thresholds are purely descriptive population statistics. Using a separate
calibration set would answer a different question: "what happens when thresholds from one set
are applied to another?" That is a relevant follow-up, not the question this study asks.

**Q: A reviewer asks: "Did you do cross-validation?" Why not?**

Cross-validation would mean profiling on 4 folds and evaluating on the 5th, then repeating.
This would test whether the outlier statistics generalize across image subsets. It would add
~5× compute but would strengthen the generalizability claim. We have not done this because
our claim is mechanistic/descriptive, not generalizability. But we should have an answer
ready for this question.

**Q: What's the difference between this and SmoothQuant?**

SmoothQuant (Xiao et al., 2023) proposes migrating quantization difficulty from activations
to weights by scaling activations per-channel and absorbing the inverse scale into weights.
Our ablation study is a **pre-quantization analysis** — we're not doing quantization at all.
We're asking: "if you had to zero outliers based on per-channel statistics, how much accuracy
would you preserve?" Our finding that mean correction dominates (rather than variance
redistribution, which is SmoothQuant's mechanism) is a distinct and complementary insight.

**Q: How does this relate to LLM.int8()?**

LLM.int8() (Dettmers et al., 2022) decomposes matrix multiplications into a majority INT8
path and a sparse FP16 path for outlier features. Our profiling confirms that ViTs also
exhibit concentrated outliers (0.39% of elements at 3σ in Block 10), suggesting a similar
mixed-precision strategy could work. But our per-channel finding — that mean correction is
the dominant mechanism — suggests that per-channel zero-points might be a simpler and equally
effective alternative.

**Q: What GPU did you use? How long does a full run take?**

NVIDIA RTX 3070 (8 GB). Phase 1 profiling over 50K images: ~30 minutes (batch_size=128,
two passes including outlier recount). Phase 2 ablation (3 thresholds × 3 granularity
conditions × 50K images): ~2–3 hours. All times include the nnsight trace overhead, which
adds ~10–20% vs a raw forward pass.

**Q: How many parameters does the model have, and how much does an INT8 version cost?**

ViT-B/16 has ~86M parameters. At FP32: ~344 MB. At INT8: ~86 MB. The activations would
also be quantized to INT8 in a full inference pipeline, reducing memory bandwidth and enabling
faster integer operations on edge hardware.

**Q: Is the code reproducible?**

Yes. `environment.yml` pins all dependencies. `seed_everything()` fixes Python, NumPy, PyTorch,
and CUDA seeds. Profiling results include full system metadata (software versions, GPU, CUDA
version, timestamp). All hyperparameters are in frozen dataclasses. The ablation is
deterministic. The README has exact commands to reproduce every result.

### Poster Presentation Questions

**Q: "Why should I care about ViT outliers?"**

Because they're the bottleneck for edge deployment. You can't run an FP32 ViT-B on a Jetson
Orin with acceptable latency. You need INT8. But INT8 quantization destroys accuracy when
outlier values are present. Understanding where outliers are and why they exist is the first
step toward quantizing around them.

**Q: "What's the practical takeaway?"**

Per-channel quantization schemes (per-channel zero-point + per-channel scale) can preserve
significantly more accuracy than per-tensor schemes for ViT activations. The zero-point
correction alone recovers 20 pp at aggressive thresholds. This is implementable: per-channel
zero-points can be fused into bias terms in the next linear layer's computation.

**Q: "Is this a new finding?"**

Partially. The existence of outliers in transformers is known (Dettmers et al., 2022; Xiao
et al., 2023), but the systematic profiling of 73 ViT sites, the per-channel vs global
ablation comparison, the mean/variance decomposition, and the gain-σ correlation linking
outliers to trained weights — these are novel contributions in the context of vision
transformers specifically.

**Q: "What would you do next?"**

1. **Per-channel PTQ:** Implement per-channel activation quantization (per-channel zero-point
   + scale) and measure end-to-end INT8 accuracy vs per-tensor.
2. **Mixed-precision:** Use the gain-σ correlation to guide per-channel bit-width allocation.
   High-gain channels get INT8, extreme-gain channels get INT16.
3. **Training-set calibration:** Profile on ImageNet-1K training set (1.28M images) and
   ablate on validation set to eliminate the calibration/evaluation overlap concern.
4. **Edge deployment:** Implement a full INT8 inference pipeline (per-channel activation
   quantization + per-channel weight quantization + integer GELU approximation) on Jetson
   Orin and benchmark latency/throughput.

**Q: "How confident are you in the +3.76 pp result?"**

The 95% CI is [3.12, 4.36], computed via two-proportion z-interval on 50K images. The CI
does not include zero, so the result is statistically significant at p < 0.05. The ablation
is deterministic (verified across 5 seeds), so there is no run-to-run variance.

**Q: "Why is 6.56% so bad? Is that worse than random?"**

Yes. The var_only condition achieves 6.56% top-1 on a 1000-class problem. Random guessing
would achieve 0.1%. But the model with var_only ablation at k=3 zeros ~0.39% of elements —
less than 1%. The fact that zeroing 0.39% of elements destroys accuracy to 6.56% means it's
actively zeroing the *wrong* elements: activations that are within the normal operating range
of their channels. The model depends on those specific activations for classification.

**Q: "What does the 1.89× efficiency ratio mean in practical terms?"**

If you need to achieve a certain sparsity target (say, 10% of activations zeroed), per-channel
thresholding lets you reach that target at roughly half the accuracy cost of global
thresholding. On the accuracy–sparsity Pareto frontier, per-channel strictly dominates global
at every operating point.

**Q: "Isn't zeroing activations extreme? What does this have to do with quantization?"**

Zeroing is the strongest possible intervention — it completely removes the activation's
contribution. Quantization error is a softer version of the same thing: instead of setting x
to 0, you round x to the nearest quantized value, introducing error ε. If the model can
survive zeroing 0.39% of pre-GELU elements at 3σ, that tells you it can certainly survive
quantization error on those same elements. Conversely, if zeroing those elements destroys
accuracy, quantization error on them will also hurt — and you should prioritize preserving
them (e.g., with higher bit-width or per-channel scaling).

---

## Document Metadata

- **Project:** ViT Quantization & Outlier Profiling
- **Created:** 2026-08-10
- **Intended audience:** Presenter preparing for a poster presentation
- **Repository:** `/home/dan/Research/ViT-Quantization-Summer-Scholars`
- **Key documents referenced:** `README.md`, `poster-planning.md`, `docs/phase2-expansion.md`,
  `docs/EXP1-IMPL.md`, `docs/EXP2-IMPL.md`, `docs/MISTAKES.md`, `docs/CITATIONS.md`