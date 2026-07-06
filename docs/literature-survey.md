# Literature Survey: ViT-B/16 Post-Training Quantization for Edge Deployment

**Date:** 2026-07-05
**Scope:** Every paper cited here connects to a specific module, experiment, or
design decision in this codebase. The survey is organized by sub-problem rather
than chronologically.

---

## 1. The Foundational Method: LLM.int8()

### Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. (2022)

*LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.*
NeurIPS 2022.
[https://arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339)

This paper is the starting point for the entire project. It introduced
mixed-precision decomposition for transformer inference: for each matrix
multiplication `Y = X @ W^T`, inspect the input activation `X`. Feature columns
of `X` that contain values exceeding a fixed threshold (6.0) in at least 25% of
tokens are routed to FP16. All other columns are computed in INT8 via cuBLAS.

**What the paper found (LLMs):** Outliers are sparse. A small number of feature
dimensions (roughly 0.1% of columns) carry extreme values (magnitudes of 60 or
more). These outlier columns persist across tokens and across inputs. The
sparsity and persistence together make column-wise routing efficient: only a
tiny fraction of compute runs in FP16.

**What this project does with it:** Experiment 1 (`src/hooks.py`,
`run_experiment1_mapping.py`) reproduces the LLM.int8() measurement logic
exactly: forward pre-hooks on every `nn.Linear`, the fixed 6.0 threshold, the
25% token-persistence bar for column routing, and per-column routing fraction as
the primary metric. The result is that ViT-B/16 violates the sparsity assumption
at blocks 9-10, where 97-100% of columns require FP16 routing. The persistence
assumption holds (outliers concentrate in specific channels), but the sparsity
assumption fails catastrophically at two layers.

**Warning:** The LLM.int8() paper's outlier characterization (Section 3) was
done on models up to 175B parameters. The outlier topology they describe may be
an emergent property of scale. ViT-B/16 at 86M parameters sits far below the
scale regime where those findings were established. The fact that your ViT
outliers look different may reflect a scale effect rather than a modality
effect. This does not invalidate your measurement, but it complicates the claim
that "ViTs are different from LLMs." A fairer framing: "ViT-B/16 at 86M
parameters shows a different outlier topology than the 6.7B-175B LLMs studied in
Dettmers et al."

---

## 2. Post-Training Quantization for Vision Transformers

### Yuan, Z., Xue, C., Chen, Y., Wu, Q., and Sun, G. (2022)

*PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform
Quantization.* ECCV 2022.
[https://arxiv.org/abs/2111.12293](https://arxiv.org/abs/2111.12293)

The first major paper to tackle ViT-specific PTQ. Introduces twin uniform
quantization: separate quantization ranges for positive and negative activation
values, motivated by the asymmetric distribution of post-GELU activations in ViT
MLP layers.

**Key finding relevant to this project:** Their per-layer sensitivity analysis
(Figure 5) shows that MLP layers dominate quantization error in ViTs, with
attention projections being relatively robust. This matches the prediction from
your Experiment 1 outlier map: `mlp.fc1` at blocks 9-10 should dominate accuracy
loss. Their sensitivity ranking was obtained by quantizing one layer at a time
and measuring accuracy drop, which is the same protocol as your Experiment 3.

**Warning about a potential dead end:** PTQ4ViT's twin-uniform quantizer
addresses the asymmetry problem (post-GELU values are predominantly positive
with a long tail). Your quantization module (`src/quantization.py`) uses
symmetric quantization with range [-127, 127]. If Experiment 3 shows that
post-GELU layers (all `mlp.fc2` inputs) lose accuracy even when their outlier
density is low, the cause may be asymmetry rather than outliers. Twin-uniform
quantization would fix that without needing mixed-precision routing. This is a
confound: your outlier map may correctly identify `mlp.fc1` as the problem, but
`mlp.fc2` may also degrade for a different reason (asymmetry) that your current
quantizer cannot distinguish from outlier-driven degradation.

### Liu, Z., Wang, Y., Han, K., Zhang, W., Ma, S., and Gao, W. (2021)

*Post-Training Quantization for Vision Transformer.* NeurIPS 2021.
[https://arxiv.org/abs/2106.14156](https://arxiv.org/abs/2106.14156)

Introduces the principle that different operations in a ViT need different
quantizers. Linear layers get uniform quantization. Post-Softmax attention maps
get a log2 quantizer because the Softmax output follows a power-law distribution
(most values near zero, a few near one). Uniformly distributing quantization
bins across [0, 1] wastes resolution on the near-zero region where precision
does not matter and under-allocates bins to the near-one region where precision
determines which tokens attend to which.

**Relevance to this project:** Your current quantization module treats every
`nn.Linear` input identically. It does not distinguish between post-LayerNorm
activations (entering `attn.qkv`, `mlp.fc1`), post-attention activations
(entering `attn.proj`), and post-GELU activations (entering `mlp.fc2`). These
three tensor types have different distributions. The Liu et al. paper provides
evidence that a single quantization strategy applied uniformly will produce
misleading sensitivity results: a layer may appear sensitive because the
quantizer is poorly matched to its distribution, not because the layer is
inherently fragile.

**Warning:** Your Experiment 3 uses `quantize_per_tensor` (symmetric, uniform)
for every layer. If `attn.proj` layers show unexpected sensitivity, the cause
may be the post-attention distribution (many moderate values, few extremes)
interacting poorly with symmetric uniform quantization, rather than outlier
density. The Liu et al. paper's log2 quantizer for post-Softmax tensors is one
example of operation-specific quantization. Your project may need to adopt
operation-specific quantizers before the per-layer sensitivity ranking can be
trusted as a measure of intrinsic sensitivity rather than quantizer mismatch.

### Lin, Y., Zhang, T., Sun, P., Li, Z., and Zhou, S. (2022)

*FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer.*
IJCAI 2022.
[https://arxiv.org/abs/2111.13824](https://arxiv.org/abs/2111.13824)

Introduces Power-of-Two Factor (PTF) for LayerNorm quantization and
Log-Int-Softmax for attention quantization. Their key finding: post-LayerNorm
activations show extreme inter-channel variation, and this variation is the
primary obstacle to INT8 quantization of ViTs.

**Relevance to this project:** Your Experiment 1 measures exactly this
inter-channel variation (the per-column routing fraction, the channel
persistence variance). The FQ-ViT finding validates your measurement target: you
are looking at the right tensor (post-LayerNorm activations entering linear
layers) and the right phenomenon (inter-channel scale variation). Their PTF
approach (scaling factors applied per-channel before LayerNorm) is conceptually
similar to the SmoothQuant approach your roadmap considers in Path B. The
difference: FQ-ViT applies the factor before LayerNorm, SmoothQuant applies it
after LayerNorm by modifying the subsequent linear layer's weights.

**Confirmation you are on the right path:** FQ-ViT reports that post-LayerNorm
activations in ViT MLP layers show the largest inter-channel variation, and that
this variation concentrates in late-middle blocks. This is the same phenomenon
your Experiment 1 measures at blocks 8-10. The fact that an independent group
found the same pattern in a different ViT variant (they used ViT-B/32 and DeiT)
strengthens the generalizability of your finding.

---

## 3. Activation-Weight Equalization (SmoothQuant and Related Methods)

### Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. (2023)

*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
Language Models.* ICML 2023.
[https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)

The paper your July roadmap (Section 5, Path B) proposes to apply. The core
idea: for a linear layer `Y = X @ W^T`, introduce a per-channel smoothing factor
`s_c` for each input channel `c`. Scale the activation down: `X_c = X_c / s_c`.
Scale the weight up: `W_c = W_c * s_c`. The matmul output is unchanged because
the factors cancel. After smoothing, the activation's dynamic range is
compressed (easier to quantize) and the weight's dynamic range is expanded
(harder to quantize, but weights are static and can be quantized offline with
per-channel granularity).

**The migration strength parameter α:** SmoothQuant introduces a hyperparameter
α in [0, 1] that controls how much quantization difficulty migrates from
activations to weights. α=0.5 splits the difficulty equally. α=1.0 pushes all
difficulty to weights. α=0.0 leaves activations unchanged. The paper reports
that α=0.5 works well for LLMs. For your blocks 9-10 fc1, where activation σ is
11.68, you would likely need α closer to 0.8-0.9 to bring σ below the 6.0
threshold. The risk: at high α, the weight's dynamic range may become
unquantizable, shifting the problem rather than solving it. Your roadmap
(Section 5, "Risk and fallback") already identifies this.

**Warning about a potential dead end:** SmoothQuant was designed for and tested
on LLMs. The smoothing factor formula uses `max(|X|)` per channel for
activations and `max(|W|)` per channel for weights. In LLMs, the activation
outliers are extreme (values of 60+) and concentrated in very few channels, so
the smoothing factor for most channels is near 1.0 and only a few channels get
aggressive scaling. In your ViT blocks 9-10, the outliers are dense (34-60% of
values exceed 6.0), which means the smoothing factor would need to be aggressive
across many channels simultaneously. The paper does not study this regime. It is
possible that SmoothQuant degrades when applied to layers where most channels
need significant smoothing, because the weight-side amplification accumulates
across too many output channels. This is an empirical question your Path B
experiment would answer.

### Wei, X., Zhang, Y., Zhang, X., Gong, R., Zhang, S., Zhang, Q., Yu, F., and Liu, X. (2022)

*Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models.*
NeurIPS 2022.
[https://arxiv.org/abs/2209.13325](https://arxiv.org/abs/2209.13325)

An alternative to SmoothQuant. Instead of migrating outliers from activations to
weights, this method directly suppresses activation outliers by introducing a
learnable gamma factor before LayerNorm. The gamma factor is derived
analytically from the activation statistics and applied as a per-channel scaling
before the normalization step.

**Relevance:** If SmoothQuant fails on blocks 9-10 (weight amplification becomes
unmanageable), Outlier Suppression offers a fallback. It operates before
LayerNorm rather than after, which means it modifies the normalization behavior
rather than the linear layer weights. This may be more stable for layers with
dense outliers because the suppression happens before the variance is
normalized.

### Nagel, M., Amjad, R. A., van Baalen, M., Louizos, C., and Blankevoort, T. (2020)

*Up or Down? Adaptive Rounding for Post-Training Quantization.* ICML 2020.
[https://arxiv.org/abs/2004.10568](https://arxiv.org/abs/2004.10568)

Introduces AdaRound: instead of rounding each weight to the nearest quantization
level (round-to-nearest), learn whether to round up or down for each weight
element. This is a small optimization (a few thousand iterations on a
calibration set) that recovers significant accuracy compared to round-to-nearest
weight quantization.

**Relevance:** Your Experiment 2 uses round-to-nearest for weight quantization
(via `torch.round` in `quantize_dequantize`). AdaRound would improve the
weight-side accuracy of all four Experiment 2 configurations. This is an
orthogonal improvement: it does not address activation outliers, but it reduces
the weight quantization error that compounds with activation error. If
Experiment 2 shows that per-channel weight quantization (Config B) still leaves
a meaningful accuracy gap, AdaRound could close part of that gap.

---

## 4. The "Massive Activations" Phenomenon

### Sun, M., Chen, X., Kolter, J. Z., and Liu, Z. (2024)

*Massive Activations in Large Language Models.* COLM 2024.
[https://arxiv.org/abs/2402.17762](https://arxiv.org/abs/2402.17762)

This paper characterizes "massive activations": individual scalar values in
transformer hidden states that are orders of magnitude larger than the typical
activation scale. The abstract states the authors "also study massive activations
in Vision Transformers."

**Key claims from the abstract:**
1. Massive activations appear in specific feature dimensions at specific
   sequence positions.
2. Their values "largely stay constant regardless of the input."
3. They "function as indispensable bias terms" — removing them degrades
   performance.

**Relevance to your block 9-10 explosion:** If the fc1 activation explosion at
blocks 9-10 is input-invariant (claim 2), then a static remedy (fixed per-channel
smoothing factors derived once from calibration data) would work reliably. If
the explosion is input-dependent, the smoothing factors would need to adapt
per-input, which is harder. Your Experiment 1 does not test for input-invariance
(it aggregates statistics across all images). A quick diagnostic: run the same
5,000 images through the model twice with different random seeds (or different
image subsets) and check whether the per-channel statistics at blocks 9-10 fc1
are stable. If they are, the explosion is likely input-invariant and SmoothQuant
has a good chance of working.

**Warning:** The advisor touchpoint doc (Section 4.9) correctly notes that this
paper's abstract describes massive activations as "bias terms" at specific
positions. Your measurement shows a depth-localized scale explosion across many
channels, not a few fixed-position bias terms. These may be different phenomena.
Read the ViT section of the full paper before citing it as direct support.

### Darcet, T., Oquab, M., Mairal, J., and Bojanowski, P. (2023)

*Vision Transformers Need Registers.* NeurIPS 2023.
[https://arxiv.org/abs/2309.16588](https://arxiv.org/abs/2309.16588)

Shows that ViTs produce high-norm tokens in background image regions (patches
with little informative content). These artifact tokens distort the attention
maps by consuming disproportionate attention weight. The solution: add learnable
[REG] tokens that provide an outlet for high-norm information.

**Relevance:** This paper addresses token-level artifacts, not feature-channel
scale explosions. The advisor touchpoint doc (Section 4.9) correctly
distinguishes these phenomena. The paper is worth citing in a "broader context
of anomalous ViT activations" framing, but it does not explain your block 9-10
fc1 explosion.

**Warning about a potential dead end:** If the block 9-10 explosion is caused by
specific high-norm tokens (the Darcet phenomenon) rather than a channel-level
property of the weights, then per-channel smoothing (SmoothQuant) would not fix
it. SmoothQuant operates on feature channels. If the explosion is driven by a
few tokens with extreme norms, you would need per-token scaling, not per-channel
smoothing. Your Experiment 2 Config C (per-token activation quantization) would
test this: if per-token scaling recovers significant accuracy at blocks 9-10
relative to per-tensor scaling, the explosion is token-driven. If it does not,
the explosion is channel-driven and SmoothQuant is the right approach.

---

## 5. Per-Layer Sensitivity and Mixed-Precision Allocation

### Dong, Z., Yao, Z., Gholami, A., Mahoney, M. W., and Keutzer, K. (2019)

*HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision.*
ICCV 2019.
[https://arxiv.org/abs/1905.03696](https://arxiv.org/abs/1905.03696)

The foundational paper on using the Hessian spectrum to determine per-layer
quantization sensitivity. The top eigenvalue of the Hessian (or its trace)
measures how sharply the loss function curves with respect to perturbations in
that layer's weights. Layers with a larger Hessian spectrum are more sensitive
to quantization error and should receive higher bit-width.

**Relevance:** Your Experiment 3 measures sensitivity empirically (quantize one
layer, measure accuracy drop). HAWQ provides a theoretical framework for
predicting sensitivity without running accuracy experiments. The Hessian trace
should correlate with your outlier density metric: both are signals of how
concentrated a layer's representational capacity is. If your Experiment 3
sensitivity ranking matches the HAWQ Hessian ranking (which you could compute
from the pretrained weights without any data), that would strengthen the claim
that your outlier map captures a fundamental property of the model rather than a
dataset artifact.

**Warning:** HAWQ was developed for CNNs (ResNet, Inception). The Hessian
spectrum of transformer layers may behave differently due to the residual
connections and LayerNorm. The Hessian of a transformer block is not simply the
sum of the Hessians of its sublayers because LayerNorm introduces
scale-invariance that flattens the loss landscape in certain directions. A
direct application of HAWQ to ViT layers may produce misleading sensitivity
rankings. The empirical approach (your Experiment 3) is safer.

### Yao, Z., Dong, Z., Zheng, Z., Gholami, A., Yu, J., Tan, E., Wang, L., Huang, Q., Wang, Y., Mahoney, M. W., and Keutzer, K. (2021)

*HAWQ-V3: Dyadic Neural Network Quantization.* ICML 2021.
[https://arxiv.org/abs/2011.10680](https://arxiv.org/abs/2011.10680)

Extends HAWQ to mixed-precision with dyadic (power-of-two) quantization scales.
Dyadic scales are hardware-efficient because scaling by a power of two is a bit
shift.

**Relevance to edge deployment:** Your eventual Jetson Orin Nano target benefits
from dyadic scales. If you adopt per-channel or per-token quantization in your
final routing policy, using power-of-two scales would reduce the hardware
overhead of per-element scaling. This is a future-work consideration, not a July
sprint item.

### Li, Z., Ma, L., Chen, M., Xiao, J., and Gu, Q. (2022)

*Patch Similarity Aware Data-Free Quantization for Vision Transformers.*
ECCV 2022.
[https://arxiv.org/abs/2203.02250](https://arxiv.org/abs/2203.02250)

Proposes PSAQ-ViT, a data-free quantization framework for ViTs. Their key
insight: the self-attention module processes Gaussian noise and real images with
systematically different patch similarity patterns. They exploit this difference
to generate calibration samples without access to real data, then use those
samples to calibrate quantization parameters.

**Relevance:** The patch-similarity analysis in this paper provides a way to
think about per-patch activation distributions in ViTs. The CLS token and image
patches have different statistical properties, which matters for per-token
quantization granularity. If your Experiment 2 Config C (per-token) shows
unexpected accuracy loss, the cause may be that per-token scaling conflates CLS
and patch tokens that need separate treatment. This paper provides the
analytical framework for diagnosing that.

---

## 6. Quantization Fundamentals and Granularity

### Krishnamoorthi, R. (2018)

*Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper.*
[https://arxiv.org/abs/1806.08342](https://arxiv.org/abs/1806.08342)

The canonical reference on quantization granularity. Defines per-tensor,
per-channel, and per-axis quantization with clear mathematical formulations.
Covers the accuracy-vs-efficiency tradeoff: finer granularity (per-channel)
improves accuracy but requires more scale factors and more complex hardware.

**Relevance:** This is the right citation for the "why these four granularity
levels" framing in your Experiment 2. It provides the theoretical grounding for
why per-channel weights and per-token activations represent points on a
spectrum, and why the hardware cost increases with granularity.

### Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., and Kalenichenko, D. (2018)

*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
Inference.* CVPR 2018.
[https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)

The paper that standardized symmetric per-channel quantization with per-layer
quantization parameters. Your `src/quantization.py` follows this formulation:
symmetric quantization with range [-127, 127], scale factor computed as
`absmax / 127`, zero point at zero.

**Relevance:** Cite this as the "standard integer quantization formulation" that
your fake-quantization module implements. It establishes that your simulation is
faithful to the widely-adopted quantization scheme.

---

## 7. Quantization for Edge Deployment and Hardware EDP

### Wu, H., Judd, P., Zhang, X., Isaev, M., and Micikevicius, P. (2020)

*Integer Quantization for Deep Learning Inference: Principles and Empirical
Evaluation.*
[https://arxiv.org/abs/2004.09602](https://arxiv.org/abs/2004.09602)

A comprehensive empirical survey of INT8 quantization across multiple
architectures with real hardware measurements. Section 5 covers quantization on
edge devices (Jetson-class hardware) and quantifies the wall-clock speedup from
INT8 inference.

**Relevance:** This is the paper to cite when you acknowledge the
simulation-vs-hardware gap in your README and roadmap. It provides measured EDP
numbers for INT8 vs FP16 on edge hardware, which gives a concrete target for
what your routing policy needs to achieve. Their finding: INT8 GEMMs on Jetson
hardware are roughly 2x faster than FP16 GEMMs. This means your routing policy's
FP16 fraction directly translates to a slowdown factor. If 2 of 49 layers run in
FP16 (your current policy), the overall slowdown is small. If the FP16 fraction
grows (e.g., if more layers are borderline), the EDP benefit erodes.

### Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., and Blankevoort, T. (2021)

*A White Paper on Neural Network Quantization.*
[https://arxiv.org/abs/2106.08295](https://arxiv.org/abs/2106.08295)

A comprehensive survey covering quantization fundamentals, post-training vs.
quantization-aware training, per-channel vs. per-tensor, hardware
considerations, and debugging quantization failures.

**Relevance:** Good for your related-work section. It covers the full landscape
that your project sits within. Section 4 ("Debugging Quantization") is
particularly relevant: it provides a systematic approach to diagnosing why a
quantized model loses accuracy, which maps directly to your Experiment 1-3-4
pipeline (characterize outliers, measure per-layer sensitivity, test
decomposition).

---

## 8. Activation Outlier Characterization in Transformers

### Bondarenko, Y., Nagel, M., and Blankevoort, T. (2021)

*Understanding and Overcoming the Challenges of Efficient Transformer
Quantization.* EMNLP 2021.
[https://arxiv.org/abs/2109.12948](https://arxiv.org/abs/2109.12948)

Characterizes activation outliers in transformer models (BERT). Shows that
outliers are concentrated in specific feature dimensions and that they emerge
from the interaction of residual connections and LayerNorm. Their per-layer
outlier density analysis is methodologically similar to your Experiment 1.

**Relevance:** This paper establishes the methodology you are using: per-layer
outlier density measurement, per-channel statistics, and the connection between
outlier topology and quantization difficulty. It also identifies LayerNorm as
the operation that amplifies inter-channel variation, which matches your finding
that post-LayerNorm activations (entering `attn.qkv` and `mlp.fc1`) show the
largest outliers.

**Warning:** This paper studies BERT (an encoder-only language model), not ViTs.
BERT's outlier topology may differ from ViT's because BERT uses GELU (same as
ViT) but has a different residual structure (post-LayerNorm in BERT vs.
pre-LayerNorm in ViT). The pre-LayerNorm structure in ViT means the residual
stream accumulates un-normalized sublayer outputs, which may explain why the
scale explosion at blocks 9-10 is more extreme than anything reported in BERT.

### Kovaleva, O., Romanov, A., Rogers, A., and Rumshisky, A. (2019)

*Revealing the Dark Secrets of BERT.* EMNLP 2019.
[https://arxiv.org/abs/1908.08593](https://arxiv.org/abs/1908.08593)

Characterizes activation patterns in BERT. Shows that certain feature dimensions
develop extreme values that dominate the representation. This is the empirical
predecessor to the outlier analysis you are doing on ViTs.

**Relevance:** Provides evidence that the outlier phenomenon is not unique to
your model or setup. It appears across transformer architectures and tasks. This
strengthens the external validity of your finding.

---

## 9. ViT Architecture

### Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021)

*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*
ICLR 2021.
[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

The ViT paper. Defines the architecture you are quantizing: 12 transformer
blocks, each with pre-LayerNorm, multi-head self-attention, and a 2-layer MLP
(GELU activation). Hidden dimension 768, MLP dimension 3072, 12 heads, patch
size 16x16.

**Relevance:** Cite for the architecture. The pre-LayerNorm structure (LayerNorm
before each sublayer, residual connection after) is the reason the residual
stream accumulates un-normalized sublayer outputs. This is the mechanism that
allows the block 9-10 scale explosion to build up and then collapse at block 11.

---

## 10. Summary: Where You Are on the Right Path

1. **Measuring matmul inputs (pre-hooks) is correct.** The LLM.int8() paper,
   PTQ4ViT, and FQ-ViT all identify post-LayerNorm activations as the primary
   quantization bottleneck. Your measurement point is the right one.

2. **The two-pass exact statistics approach is rigorous.** Bondarenko et al.
   (2021) and the LLM.int8() paper both use per-channel statistics computed over
   the full dataset. Your Chan/Welford merge in float64 is more numerically
   stable than the batch-averaging approach used in most PTQ papers.

3. **The fixed 6.0 threshold is the right primary metric.** FQ-ViT and PTQ4ViT
   both confirm that absolute activation magnitude (not relative statistical
   deviation) determines INT8 quantization error. Your finding that the 3-sigma
   threshold self-normalizes and misses the blocks 9-10 explosion is consistent
   with the literature: absolute thresholds map to the quantizer's dynamic
   range, relative thresholds do not.

4. **The per-column routing fraction is the right cost model.** The LLM.int8()
   paper's column-wise routing is a hardware constraint (cuBLAS INT8 GEMMs
   require whole-column routing), not a modeling choice. Your measurement of the
   gap between per-value density and per-column routing fraction correctly
   captures the structured-sparsity penalty.

5. **The block 9-10 fc1 explosion is a real phenomenon.** Sun et al. (2024)
   report massive activations in ViTs. FQ-ViT reports inter-channel variation
   concentrated in late-middle MLP layers. Your measurement is independently
   confirmed by multiple groups.

---

## 11. Summary: Warnings and Potential Dead Ends

1. **Symmetric uniform quantization may be the wrong quantizer for post-GELU
   activations.** PTQ4ViT (Yuan et al., 2022) shows that post-GELU activations
   are asymmetric (predominantly positive with a long tail) and require
   twin-uniform or asymmetric quantization. Your `src/quantization.py` uses
   symmetric quantization for all layers. If Experiment 3 shows that `mlp.fc2`
   layers (which receive post-GELU activations) lose accuracy despite low
   outlier density, the cause is asymmetry, not outliers. This would mean your
   outlier map correctly identifies `mlp.fc1` as the problem but misses a
   separate failure mode at `mlp.fc2`.

2. **Operation-specific quantization may be necessary before sensitivity
   rankings are meaningful.** Liu et al. (2021) show that post-Softmax
   activations need log2 quantization, post-GELU activations need asymmetric
   quantization, and post-LayerNorm activations need uniform quantization.
   Applying the same symmetric uniform quantizer to all three tensor types (as
   your Experiment 3 does) conflates quantizer mismatch with intrinsic layer
   sensitivity. A layer may appear sensitive because it receives the wrong type
   of quantizer.

3. **SmoothQuant may not transfer to dense-outlier regimes.** Xiao et al. (2023)
   designed and tested SmoothQuant on LLMs where outliers are extreme and sparse
   (a few channels with values of 60+). Your blocks 9-10 have dense outliers
   (34-60% of values exceed 6.0). The smoothing factor formula may become
   unstable when applied aggressively across many channels simultaneously. The
   weight-side amplification may accumulate across output channels and make the
   weight unquantizable. This is the risk your roadmap already identifies.

4. **The block 9-10 explosion may be token-driven, not channel-driven.** Darcet
   et al. (2023) show that ViTs produce high-norm artifact tokens. If your fc1
   explosion is caused by a few extreme tokens rather than a channel-level
   property, per-channel smoothing (SmoothQuant) would not fix it. Per-token
   scaling (your Experiment 2 Config C) would help in that case. If Config C
   recovers significant accuracy at blocks 9-10, the explosion is token-driven
   and SmoothQuant is the wrong remedy.

5. **The LLM.int8() outlier topology may be a scale effect, not a modality
   effect.** Dettmers et al. (2022) studied models from 6.7B to 175B parameters.
   ViT-B/16 at 86M parameters is two orders of magnitude smaller. The difference
   in outlier topology may reflect model scale rather than a fundamental
   difference between vision and language. This does not invalidate your
   measurement, but it complicates the claim that "ViTs are different from
   LLMs." A fairer framing: "ViT-B/16 at 86M parameters shows a different
   outlier topology than the 6.7B-175B LLMs studied in Dettmers et al."

6. **The calibration/evaluation overlap is a real methodological concern.**
   Bondarenko et al. (2021) and Nagel et al. (2021) both recommend using a
   separate calibration set for quantization parameter estimation. Your current
   setup uses the same validation images for Experiment 1 statistics and
   (planned) Experiments 2-4 accuracy evaluation. This is standard practice in
   some PTQ papers (PTQ4ViT uses the same calibration set for statistics and a
   subset of the validation set for accuracy), but it weakens the statistical
   independence of your results. The advisor touchpoint doc (Section 4.7) flags
   this correctly.

---

## 12. Quick-Reference Table

| Your Module / Experiment | Key Paper(s) | What It Provides |
|---|---|---|
| `src/hooks.py` (outlier measurement) | Dettmers et al. 2022 | The routing logic you reproduce |
| `src/hooks.py` (per-channel statistics) | Bondarenko et al. 2021 | Outlier characterization methodology |
| `src/quantization.py` (fake quantize) | Jacob et al. 2018; Krishnamoorthi 2018 | Symmetric quantization formulation |
| Experiment 2 (granularity) | Krishnamoorthi 2018; Li et al. 2022 (PSAQ-ViT) | Per-tensor vs per-channel vs per-token; patch-similarity analysis |
| Experiment 3 (per-layer sensitivity) | Dong et al. 2019 (HAWQ); Yuan et al. 2022 (PTQ4ViT) | Hessian-based and empirical sensitivity |
| Experiment 4 (decomposition) | Dettmers et al. 2022; Xiao et al. 2023 | Mixed-precision decomposition and SmoothQuant |
| Path B (SmoothQuant) | Xiao et al. 2023; Wei et al. 2022 (2209.13325) | Activation-weight equalization and outlier suppression |
| Block 9-10 sigma explosion | Sun et al. 2024; Darcet et al. 2023 | Massive activations and artifact tokens |
| Edge deployment / EDP | Wu et al. 2020; Nagel et al. 2021 | Hardware quantization benchmarks |
| ViT architecture | Dosovitskiy et al. 2021 | The model you are quantizing |
| Post-GELU asymmetry | Yuan et al. 2022 (PTQ4ViT) | Twin-uniform quantization for asymmetric activations |
| Post-Softmax quantization | Liu et al. 2021 | Log2 quantizer for power-law distributions |
| Weight rounding optimization | Nagel et al. 2020 (AdaRound) | Adaptive rounding for weight quantization |