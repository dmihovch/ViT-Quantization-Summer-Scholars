# PROBLEMS.md — Known Research and Engineering Weaknesses

This document is a candid inventory of every area where the project is standing on shaky ground. It is written adversarially: assume a reviewer who wants to find holes. Each problem is tagged with its source document and given a severity rating.

**Severity scale:**
- 🔴 **Critical** — A reviewer or peer would reject this claim outright. Blocks publication.
- 🟠 **Serious** — Weakens the research story significantly. Must be addressed before presenting results.
- 🟡 **Moderate** — A gap in rigor that a careful reader will notice. Should be fixed.
- 🟢 **Minor** — Polish and precision issues. Low priority but worth knowing.

---

## Section 1: Hypothesis and Scope Problems

### P1 — The "100% Integer-Only" Claim is Architecturally False 🔴
**Source:** `METHODOLOGIES.md §1`

The document asserts "100% integer-only inference." This is demonstrably wrong for `vit_base_patch16_224`. A ViT has three sources of non-linearity beyond GELU:

1. **LayerNorm** — requires computing a mean and variance over the token dimension, then dividing by a standard deviation. Division is not integer-friendly. I-BERT handles this with an integer-only Newton-Raphson approximation for the reciprocal square root.
2. **Softmax (in self-attention)** — requires `exp()`, a transcendental function. FQ-ViT handles this with a Log2-based integer approximation.
3. **The [CLS] token classification head** — typically outputs raw logits fed into a softmax, again transcendental.

The pipeline as written quantizes only FFN activations. Claiming "100% integer-only" from that partial coverage is a false statement. The correct claim is: *"we establish an integer-only FFN execution path."*

---

### P2 — OVERVIEW.md and METHODOLOGIES.md Directly Contradict Each Other 🔴
**Source:** `OVERVIEW.md §Phase 3` vs. `METHODOLOGIES.md §Phase C`

`OVERVIEW.md` explicitly frames Phase 3 as testing *Dr. Yang's hypothesis*: "strip out outliers to 0." The ablation sweep includes a `ZERO_STRIP` strategy as an explicit treatment.

`METHODOLOGIES.md §Phase C` then states: *"Destructively zeroing out these outliers eliminates network knowledge,"* and mandates MSE clamping instead — i.e., it has already decided the outcome before the experiment is run.

This means either:
- The research question changed and OVERVIEW.md was never updated (documentation drift), or
- The design is pre-determining the result of the ablation study, which is a form of hypothesis injection that invalidates the study's scientific value.

A proper ablation must remain agnostic about which strategy wins until after the data is collected. The OVERVIEW and METHODOLOGIES must agree on what question is being asked.

---

### P3 — The Core Hypothesis is Not Falsifiable as Stated 🟠
**Source:** `OVERVIEW.md §Phase 4`

> *"A higher accuracy in the Co-optimized Test empirically validates the core hypothesis."*

This framing has a serious flaw: any positive delta — even 0.01% — would be declared a "validation." There is no pre-specified minimum effect size, no null hypothesis, and no statistical test. A well-formed quantitative research hypothesis requires:

1. A null hypothesis: *"Clipping provides no accuracy benefit over the unclipped integer GELU baseline."*
2. A pre-specified minimum delta that would be considered meaningful (e.g., < 1% Top-1 accuracy drop from FP32 baseline).
3. An acknowledgment that the result is a single point estimate with no variance estimate across calibration seeds.

Without these, the conclusion is unfalsifiable — any positive result "validates" and any negative result can be explained away.

---

### P4 — The Research Claims No Novelty Relative to Prior Work 🟠
**Source:** `METHODOLOGIES.md` (entire document), `OVERVIEW.md`

The LUT-based GELU replacement is a known technique. The MSE-optimized clipping threshold (ACIQ, Banner et al. 2019) is a known technique. The integer-only requantization (Jacob et al. 2018) is a known technique. The project combines them for ViT-B/16 FFN blocks.

That combination *could* be the novelty claim — but nowhere in any document is it stated: "Prior work X did A, prior work Y did B. Nobody has done A+B jointly for ViT FFNs, specifically because of problem Z." Without a stated novelty claim grounded in a literature gap, this reads as a re-implementation exercise, not a research contribution.

---

## Section 2: Mathematical Specification Problems

### P5 — MSE Objective is Underspecified 🔴
**Source:** `METHODOLOGIES.md §Phase C`

The optimization objective is written as:
$$\min_{T} \| X - \hat{X} \|_{2}^{2}$$

But $\hat{X}$ is never formally defined. The full round-trip is:
$$\hat{X} = S \cdot \text{clip}\left(\left\lfloor \frac{\text{clip}(X, -T, T)}{S} \right\rceil, -128, 127\right)$$

where $S = T/127$. Without this definition, the formula is mathematically incomplete. A reader cannot implement it from the document alone.

Additionally:
- The optimization domain for $T$ is not stated (what range? coarse-to-fine grid search? gradient-based?).
- Per-tensor or per-channel? This changes the result substantially for transformer activations with high inter-channel variance.
- The same MSE objective is used for both activations and weights in the ACIQ literature — are you applying this to activations only? That must be stated.

---

### P6 — Dyadic Scale Approximation Error is Unquantified 🟠
**Source:** `METHODOLOGIES.md §Phase B`

The dyadic decomposition $S \approx M_0 \cdot 2^{-p}$ introduces an approximation error because $S$ is irrational in general and $M_0 \cdot 2^{-p}$ is rational with a power-of-two denominator. The document says this "guarantees the tensor is mathematically scaled correctly" — but that is only true up to the approximation error, which depends on the chosen bit-width of $M_0$.

The document never specifies:
- How many bits are used for $M_0$ (typically 32).
- What the worst-case rounding error on the scale is.
- Whether the approximation error compounds across layers or is absorbed by the next calibration step.

---

### P7 — "Zero Online Approximation Error" is Factually Wrong 🔴
**Source:** `METHODOLOGIES.md §Phase D`

> *"This guarantees an O(1) memory fetch with zero online approximation error."*

This is false. The LUT maps integer inputs to integer outputs. The output quantization step:
$$LUT[x_q] = \left\lfloor \frac{GELU(x_q \cdot S_{in})}{S_{out}} \right\rceil$$

introduces a rounding error of up to $\pm \frac{S_{out}}{2}$ per element. This is the *output quantization error*, and it is nonzero. The correct statement is: *"The LUT eliminates polynomial approximation error; residual quantization error is bounded by $\frac{S_{out}}{2}$ per activation."*

This is not minor pedantry — you will need to know this bound to discuss your results in a paper.

---

### P8 — σ-Based Thresholds Assume Gaussian Distributions 🟠
**Source:** `OVERVIEW.md §Phase 2 and Phase 3`

The ablation uses 2σ and 3σ clipping thresholds. σ-based reasoning (the 68-95-99.7 rule) is only meaningful for Gaussian distributions. ViT pre-GELU activations are empirically known to be heavy-tailed and non-Gaussian. On a heavy-tailed distribution, 3σ may clip far more (or far less) than 0.3% of values depending on the actual kurtosis.

Using kurtosis as a metric (Phase 1) is good — but then using σ-based thresholds in Phase 3 without accounting for the kurtosis is inconsistent. If kurtosis >> 3, the σ-based thresholds are not statistically meaningful. The thresholds should be stated as percentile-based (e.g., 99th percentile) or derived from the ACIQ closed-form for the measured distribution shape.

---

## Section 3: Experimental Design Problems

### P9 — Calibration Set Size Has No Justification 🟡
**Source:** `OVERVIEW.md §Phase 1`

The protocol specifies 1,024 images for calibration. This number appears without justification. The literature (Hubara et al. 2021) shows that calibration set size significantly affects PTQ performance — too few images leads to scale factors that do not generalize to the full validation distribution.

No ablation over calibration set sizes is planned. There is no argument for why 1,024 is sufficient for 12 transformer blocks × 2 MSE threshold sweeps = 24 separate calibration decisions.

---

### P10 — The Ablation Runs Once with No Variance Estimate 🟠
**Source:** `OVERVIEW.md §Phase 3`

Each clipping strategy is evaluated exactly once on the full validation set. This gives a point estimate with no measure of variance. For a fixed model+dataset, validation accuracy is deterministic (no stochasticity in eval), so this is less catastrophic than it would be in a training experiment. However:

- The calibration set used to compute σ and MSE thresholds *is* stochastic (it is a sample). The reported accuracy is therefore sensitive to which 1,024 images were selected.
- No experiment repeats the calibration step with different random seeds to measure this sensitivity.

This means the reported accuracy numbers are not reproducible by definition.

---

### P11 — No Comparison to a Published Baseline 🟠
**Source:** `OVERVIEW.md §Phase 4`, `METHODOLOGIES.md §1`

The final comparison is: *our clipped integer GELU* vs. *our unclipped integer GELU*. The baseline is internal. There is no comparison to:

- FP32 baseline (stated, this is fine)
- A published PTQ method on the same model (e.g., FQ-ViT, Q-ViT, EfficientFormer)
- A standard 8-bit PTQ without the integer approximation (W8A8)

Without an external baseline, the result cannot be situated in the literature. You cannot answer: *"Is this better or worse than the best existing method?"*

---

### P12 — Hooks Target `nn.GELU` but May Miss Some 🟡
**Source:** `OVERVIEW.md §Phase 1`, `REPO-STRUCTURE.md §src/hooks.py`

`vit_base_patch16_224` from `timm` uses `nn.GELU` in its FFN blocks, but the exact module structure depends on the `timm` version. Some variants use functional `F.gelu` calls or fused attention implementations that bypass `nn.GELU` module instances entirely. Forward hooks on `nn.GELU` modules will silently miss any activations computed through the functional API or through fused kernels.

The protocol does not include a verification step to confirm that hooks were registered on *all* expected GELU instances (should be 24 for a 12-block ViT-B: 2 per block).

---

## Section 4: Implementation and Code Problems

### P13 — `_VIT_B16_TRANSFORM` is Hardcoded with Wrong Normalization 🟠
**Source:** `src/data_loader.py` lines 37-46`

```python
_VIT_B16_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
_VIT_B16_STD:  tuple[float, float, float] = (0.5, 0.5, 0.5)
```

The `vit_base_patch16_224.orig_in21k_ft_in1k` model loaded in `src/model.py` uses **ImageNet normalization**: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. The `(0.5, 0.5, 0.5)` figures are the normalization used by models trained on CIFAR or some HuggingFace pipelines, not the `timm` ViT-B/16 trained on ImageNet-21k.

This discrepancy means `create_imagenet_val_loader` preprocesses images incorrectly relative to what the model expects. This will silently degrade accuracy (the model sees out-of-distribution inputs) without throwing any errors. The correct transform should be fetched from `timm` via `resolve_data_config`, exactly as `load_vit_b_16` already does in `src/model.py`.

This is a concrete, currently-broken bug.

---

### P14 — Two Separate Transform Paths Create a Silent Inconsistency 🟠
**Source:** `src/data_loader.py`, `src/model.py`

`src/model.py`'s `load_vit_b_16` correctly fetches the transform from `timm`. `src/data_loader.py` defines its own hardcoded `_VIT_B16_TRANSFORM`. Any code that uses `create_imagenet_val_loader` directly (without passing the model's transform) will use the wrong preprocessing. The two paths will silently produce different normalization and different accuracy numbers, making it impossible to compare results across experiments that use different loaders.

The fix is to eliminate the hardcoded transform entirely and always pass the transform returned by `load_vit_b_16` into the data loader.

---

### P15 — Non-Determinism is Not Fully Controlled 🟡
**Source:** `REPO-STRUCTURE.md §src/utils.py`

`seed_everything` is specified to cover `random`, `numpy`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`. This is incomplete for reproducibility on modern hardware:

- `torch.use_deterministic_algorithms(True)` is not set. Some CUDA operations (e.g., atomics in scatter operations) are non-deterministic even with a fixed seed.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are not specified.

For a PTQ experiment where you are measuring accuracy to one decimal place, a random seed variation of ±0.1% Top-1 from non-deterministic CUDA ops matters.

---

### P16 — No Model Checkpoint Pinning 🟡
**Source:** `src/model.py`

```python
model = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)
```

This downloads the latest available weights for this model name. If `timm` updates the weights (a new fine-tune, bug fix, or normalization change), the experiment silently changes its baseline. For a research project where Top-1 accuracy is the primary metric, the exact model checkpoint must be pinned by hash or version.

---

### P17 — `IntegerGELU` Module Does Not Yet Exist 🟡
**Source:** `REPO-STRUCTURE.md §src/integer_gelu.py`

The repo structure specifies `src/integer_gelu.py` as housing an `IntegerGELU` class, but the file does not exist in `src/`. The `METHODOLOGIES.md` and `OVERVIEW.md` both describe the LUT-based approach, but the repo structure document still references a *"polynomial approximation"*:

> *"Use a piecewise quadratic or bit-shift polynomial that restricts all math to INT8 or INT16 arithmetic."*

This contradicts `METHODOLOGIES.md §Phase D` which specifies a LUT. The implementation specification is inconsistent between documents. Before writing the code, decide: polynomial approximation or LUT? These have different accuracy, hardware, and latency tradeoffs.

---

## Section 5: What Is Genuinely Solid

To be fair, the following parts of the project are well-founded:

- ✅ The dyadic requantization formula (Phase B) is textbook-correct.
- ✅ The LUT construction formula is mathematically valid given a correctly specified $S_{in}$ and $S_{out}$.
- ✅ Layer-wise granularity for scale factors and LUTs is the right call.
- ✅ The four-phase experiment structure is logical and sequenced correctly.
- ✅ The code engineering (separation of concerns, frozen configs, typed interfaces) is high quality.
- ✅ Using kurtosis to measure tail heaviness in Phase 1 is the right diagnostic tool.
- ✅ The `hooks.py` spec correctly targets `nn.GELU` at module granularity rather than using generic activation hooks.
