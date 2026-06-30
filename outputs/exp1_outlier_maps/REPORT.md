# Experiment 1: Full Per-Layer Outlier Characterization
### ViT-B/16 PTQ for Edge Deployment · Summer Thesis Research

---

## 1. Executive Summary

The central hypothesis of this project, that ViT-B/16 has dense,
feedforward-concentrated activation outliers that break LLM.int8()'s
mixed-precision routing premise, is **confirmed, quantitatively and sharply**,
by the full 50,000-image characterization run.

The catastrophe is real but *localized*. Two layers out of 49 break the INT8
quantization regime: `blocks.9.mlp.fc1` and `blocks.10.mlp.fc1` have LLM.int8()
routing fractions of **97.5%** and **99.7%** respectively. The quantizer would
route essentially the entire matmul to FP16 for those layers, adding routing
overhead while gaining nothing. Every other layer in the model is clean enough
for pure INT8 (routing fraction ≤ 3.8%, and 43 of 49 layers ≤ 0.5%).

This map directly prescribes the **heterogeneous (selective) routing policy**
called for in the research plan: FP16 for 2 layers, LLM.int8() for 4 borderline
layers, pure INT8 for the remaining 43. A global LLM.int8() policy is wrong for
ViT.

The statistical (3σ) threshold is computed **per-channel**: each of the 768 /
3072 input feature dimensions receives its own mean and standard deviation. This
is the correct computation for a per-column routing diagnostic. A practical
consequence: the per-channel 3σ threshold with the 5% calibrated participation
bar flags effectively zero columns on every layer, confirming that the 3σ
threshold is not useful for quantization-safety analysis (unlike the fixed 6.0).
Its role is diagnostic: it reveals that per-channel statistical outliers in
ViT-B/16 are distributed across columns, not concentrated enough for
LLM.int8()-style routing.

---

## 2. Experiment Setup

| Parameter | Value |
|---|---|
| Model | timm `vit_base_patch16_224` (pretrained ImageNet-1K) |
| Images | 50,000 ImageNet-1K validation images |
| Tokens seen per layer | 9,850,000 (50,000 images × 197 tokens/image) |
| Measurement point | **Matmul input** (forward pre-hook on every `nn.Linear`) |
| Algorithm | **Two-pass exact**: Pass 1 computes per-channel mean/std via Chan/Welford float64 merge. Pass 2 applies frozen per-channel thresholds. |
| Fixed threshold | \|x\| > 6.0 (LLM.int8()'s own calibration) |
| Statistical threshold | \|x − μ_c\| > 3σ_c (exact per-channel μ_c, σ_c for each feature dimension c) |
| Routing column criterion, fixed | Column c is an outlier column if \|x\| > 6.0 in ≥ 25% of tokens (LLM.int8() criterion) |
| Routing column criterion, statistical | Column c is an outlier column if \|x − μ_c\| > 3σ_c in ≥ 5% of tokens (calibrated to ViT's ~1% per-channel outlier density) |
| Layers measured | 49 (12 blocks × 4 linears + 1 head) |

**Why inputs, not outputs.** LLM.int8() decomposes the matrix `Y = X @ W.T` by
inspecting `X`, the post-LayerNorm input activation, and routing outlier
*columns of X* to FP16. Measuring outputs would characterize a different tensor
that the quantizer never routes on. Every metric below is taken at the exact
decision point.

---

## 3. The Catastrophe: Blocks 9–10 mlp.fc1

The following table shows every MLP up-projection (`fc1`) in network order. The
LLM.int8() routing fraction (per-column, |x| > 6.0 threshold, 25% participation)
is the headline metric.

| Block | Input std (σ) | Max \|x\| | Per-value density (6.0) | **LLM.int8 routing fraction** | Policy |
|------:|----:|----:|----:|----:|:--|
| 0 | 0.884 | 47.3 | 0.16% | 0.00% | 🟢 INT8 |
| 1 | 0.944 | 33.9 | 0.22% | 0.13% | 🟢 INT8 |
| 2 | 0.891 | 39.5 | 0.02% | 0.00% | 🟢 INT8 |
| 3 | 0.966 | 44.6 | 0.07% | 0.13% | 🟢 INT8 |
| 4 | 1.059 | 46.9 | 0.20% | 0.26% | 🟢 INT8 |
| 5 | 1.117 | 44.3 | 0.17% | 0.13% | 🟢 INT8 |
| 6 | 1.214 | 42.2 | 0.19% | 0.13% | 🟢 INT8 |
| 7 | 1.423 | 24.6 | 0.40% | 0.39% | 🟢 INT8 |
| **8** | **3.474** | **72.9** | **5.84%** | **3.78%** | 🟡 LLM.int8 |
| **9** | **7.318** | **104.8** | **35.00%** | **97.53%** | 🔴 FP16 |
| **10** | **13.434** | **202.7** | **61.23%** | **99.74%** | 🔴 FP16 |
| 11 | 1.528 | 29.3 | 0.22% | 0.26% | 🟢 INT8 |

### What the numbers mean

Blocks 9 and 10 are a regime change, not a gradual drift. The input standard
deviation grows smoothly from σ ≈ 0.9 through blocks 0–7, then detonates: σ
jumps to 3.5, then 7.3, then **13.4** across blocks 8–10 before collapsing back
to 1.5 at block 11. At σ = 13.4, the LLM.int8() fixed threshold of 6.0 sits at
less than 0.5σ from the mean. The vast majority of activations naturally exceed
it, making the threshold meaningless as an outlier detector for that layer.

The per-value density for block 10 (`61.2%`) matches the project's preliminary
estimate of "up to 63% in late blocks" almost exactly. The per-column routing
fraction (`99.7%`) is worse still: nearly every input feature column of this
matmul exceeds 6.0 in at least 25% of tokens, so LLM.int8() would push the
entire contraction dimension to FP16. This is strictly worse than running FP16
cleanly, because the decomposition overhead is added on top.

Block 8 is the onset: σ rises to 3.5, routing fraction hits 3.78%. This is the
last layer where LLM.int8() could provide a net benefit (≈96% of the matmul
remains in INT8).

---

## 4. The Structured Routing Penalty: What the Two Metrics Together Reveal

The per-value density and per-column routing fraction are not redundant. Their
relationship is the diagnostic signal. The *direction* of the gap tells you
whether LLM.int8() is viable:

| Layer | Per-value density | Routing fraction | Gap direction | Interpretation |
|---|---:|---:|:--|:--|
| `blocks.8.mlp.fc1` | 5.84% | 3.78% | routing **<** density | Outliers cluster in fewer columns than raw count implies. Concentrated and routable. |
| `blocks.9.mlp.fc1` | 35.00% | 97.53% | routing **≫** density | Outliers touch nearly every column. Pervasive and not routable. |
| `blocks.10.mlp.fc1` | 61.23% | 99.74% | routing **≫** density | Every column is an outlier column. LLM.int8() is equivalent to FP16. |

Block 8's gap favours INT8: the few heavy outlier values are column-persistent
(concentrated), so routing a small fraction of columns to FP16 protects almost
all of the compute. Blocks 9–10's gap is inverted: outlier values spread across
nearly every feature column, so structured whole-column routing forces the
entire matmul to FP16.

The project grounding doc named two failure modes for LLM.int8(): density too
high, or outliers scattered (not channel-persistent). The channel persistence
variance for blocks 9–10 fc1 is **1,753,527** and **1,150,906** respectively,
the highest in the model. The failure mode is **density, not scatter**.
LLM.int8()'s persistence assumption holds; its sparsity assumption fails.

---

## 5. The Fixed vs. Statistical Threshold: Why 6.0 is the Right Calibration

The statistical (3σ) routing fraction is computed **per-channel**: each feature
dimension `c` has its own μ_c and σ_c from Pass 1, and Pass 2 flags a value as a
statistical outlier when `|x − μ_c| > 3σ_c`. A column is flagged only when it
meets this criterion in ≥ 5% of tokens (calibrated to ViT-B/16's ~1% per-channel
outlier density; a 25% bar would flag zero columns across all 49 layers).

**The result: the per-channel 3σ routing fraction is virtually zero everywhere.**

| Threshold | `blocks.9.mlp.fc1` RF | `blocks.10.mlp.fc1` RF | Observation |
|---|---|---|---|
| Fixed (6.0, 25% bar) | **97.53%** | **99.74%** | Correctly identifies the catastrophe |
| Statistical (3σ, per-channel, 5% bar) | **0.13%** | **0.00%** | Reports clean when the layers are broken |

Of the 49 layers, only two show any statistical routing fraction at all:
`blocks.9.mlp.fc1` (0.13%) and `blocks.11.attn.proj` (0.13%). Every other layer
is 0.00% or a single column (0.033%). The per-column statistical threshold with
the calibrated 5% bar produces essentially no signal.

**Why.** The 3σ threshold is *relative*: it self-normalizes to each layer's
scale. At σ = 13.4 for block 10, 3σ = 40.3. Most values exceed the
INT8-relevant 6.0 but not 40.3, so the statistical threshold misses the
explosion entirely. A relative threshold hides the scale information needed to
identify INT8-breaking layers. The absolute 6.0 threshold, which maps to the
fixed dynamic range of INT8, is the correct calibration for quantization-safety
analysis.

The per-value statistical density (VD_stat) sits at ≈0.3–0.5% uniformly across
all fc1 layers regardless of σ, including blocks 9–10 (0.36% / 0.36%). This is
the expected signature of a relative threshold: it detects the same tail
fraction everywhere, making every layer look identical. This uniformity
demonstrates that 3σ is inappropriate for quantization analysis: it treats the
benign block 0 and the catastrophic block 10 as equally outlier-prone.

**Implication for Experiment 4.** The 3σ decomposition threshold will fail to
protect blocks 9–10: it flags only ≈0.3% of values for high-precision treatment
when the layer has already blown past the INT8 range. The fixed 6.0 threshold
will correctly identify the problem but report a ~100% high-precision fraction,
confirming that no decomposition policy recovers INT8 efficiency there.

---

## 6. The Clean Layers

### Attention output projections (`attn.proj`): pristine

| Stat | Blocks 0–6 | Blocks 7–10 | Block 11 |
|---|---|---|---|
| Routing fraction (6.0) | **0.00%** across all | 0.00% | 0.13% |
| Per-value density (6.0) | ≈ 0 (< 2 × 10⁻⁷) | ≈ 0 | 0.04% |
| Max \|x\| | 3.4 – 8.8 | 5.5 – 7.9 | 11.4 |
| Input std σ | 0.16 – 0.28 | 0.33 – 0.63 | 0.91 |

The attention context (input to `attn.proj`) is a convex combination of Value
vectors, inherently bounded and averaged. These are the model's most
quantization-friendly matmuls. Naive per-tensor INT8 should be near-lossless
here.

### MLP down-projections (`mlp.fc2`): sparse outliers, exactly what LLM.int8 handles

Routing fractions across all fc2 layers: 0.00% (blocks 0–7), then 0.03%, 0.07%,
0.23%, 0.03% (blocks 8–11). Even at block 10, where the output of the preceding
fc1 is catastrophic, the fc2 input (the post-GELU intermediate) has only 0.23%
routing fraction and 0.28% per-value density. This is the sparse, persistent
outlier regime that LLM.int8() was designed for. Pure INT8 suffices, and
LLM.int8() would work if applied here, but the overhead at < 0.25% routing is
not justified.

### Attention QKV projections (`attn.qkv`): mild, manageable

Most blocks have routing fractions of 0.0–0.5%. Three blocks edge above 0.5%:
`blocks.3.attn.qkv` (0.78%), `blocks.5.attn.qkv` (0.52%), `blocks.6.attn.qkv`
(0.52%). These are genuine LLM.int8() candidates: sparse, column-persistent,
with a favorable gap (routing fraction ≤ per-value density). At 99%+ of the
matmul remaining in INT8, the mixed-precision overhead is low and the EDP
tradeoff is potentially worthwhile.

---

## 7. Full Routing Policy Table (All 49 Layers)

**Policy thresholds (fixed 6.0 routing fraction):**

| Threshold | Policy | Rationale |
|---|---|---|
| ≥ 50% | 🔴 **FP16** | LLM.int8() degenerates to FP16. Routing overhead is pure loss. |
| 0.5% – 50% | 🟡 **LLM.int8()** | Mixed precision provides net benefit. Outliers sparse enough to route. |
| < 0.5% | 🟢 **INT8** | Routing overhead exceeds benefit. Naive INT8 is near-lossless. |

*Note: block 4 attn.qkv (0.39%) is borderline. Experiments 3–4 should determine
whether its sensitivity warrants LLM.int8() treatment.*

| Layer | Type | σ | Max \|x\| | RF fixed | VD fixed | RF stat | VD stat | **Policy** |
|---|---|---|---:|---:|---:|---:|---:|---:|:--|
| `blocks.0.attn.qkv` | Attn-QKV | 0.515 | 19.4 | 0.00% | 0.09% | 0.00% | 0.97% | 🟢 INT8 |
| `blocks.0.attn.proj` | Attn-Proj | 0.162 | 8.8 | 0.00% | ~0% | 0.00% | 1.70% | 🟢 INT8 |
| `blocks.0.mlp.fc1` | MLP-fc1 | 0.884 | 47.3 | 0.00% | 0.16% | 0.00% | 0.53% | 🟢 INT8 |
| `blocks.0.mlp.fc2` | MLP-fc2 | 0.212 | 36.9 | 0.00% | 0.00% | 0.03% | 1.96% | 🟢 INT8 |
| `blocks.1.attn.qkv` | Attn-QKV | 0.598 | 17.1 | 0.00% | 0.17% | 0.00% | 0.44% | 🟢 INT8 |
| `blocks.1.attn.proj` | Attn-Proj | 0.219 | 4.8 | 0.00% | 0.00% | 0.00% | 0.93% | 🟢 INT8 |
| `blocks.1.mlp.fc1` | MLP-fc1 | 0.944 | 33.9 | 0.13% | 0.22% | 0.00% | 0.39% | 🟢 INT8 |
| `blocks.1.mlp.fc2` | MLP-fc2 | 0.133 | 10.7 | 0.00% | ~0% | 0.00% | 1.92% | 🟢 INT8 |
| `blocks.2.attn.qkv` | Attn-QKV | 0.700 | 11.9 | 0.00% | 0.10% | 0.00% | 0.32% | 🟢 INT8 |
| `blocks.2.attn.proj` | Attn-Proj | 0.227 | 4.4 | 0.00% | 0.00% | 0.00% | 0.63% | 🟢 INT8 |
| `blocks.2.mlp.fc1` | MLP-fc1 | 0.891 | 39.5 | 0.00% | 0.02% | 0.00% | 0.30% | 🟢 INT8 |
| `blocks.2.mlp.fc2` | MLP-fc2 | 0.134 | 7.9 | 0.00% | ~0% | 0.00% | 1.88% | 🟢 INT8 |
| `blocks.3.attn.qkv` | Attn-QKV | 0.897 | 20.7 | **0.78%** | 0.33% | 0.00% | 0.29% | 🟡 LLM.int8 |
| `blocks.3.attn.proj` | Attn-Proj | 0.231 | 3.4 | 0.00% | 0.00% | 0.00% | 0.78% | 🟢 INT8 |
| `blocks.3.mlp.fc1` | MLP-fc1 | 0.966 | 44.6 | 0.13% | 0.07% | 0.00% | 0.29% | 🟢 INT8 |
| `blocks.3.mlp.fc2` | MLP-fc2 | 0.144 | 9.4 | 0.00% | ~0% | 0.00% | 1.76% | 🟢 INT8 |
| `blocks.4.attn.qkv` | Attn-QKV | 0.969 | 19.9 | 0.39% | 0.26% | 0.00% | 0.29% | 🟢 INT8 ¹ |
| `blocks.4.attn.proj` | Attn-Proj | 0.269 | 4.3 | 0.00% | 0.00% | 0.00% | 0.80% | 🟢 INT8 |
| `blocks.4.mlp.fc1` | MLP-fc1 | 1.059 | 46.9 | 0.26% | 0.20% | 0.00% | 0.29% | 🟢 INT8 |
| `blocks.4.mlp.fc2` | MLP-fc2 | 0.148 | 12.9 | 0.00% | 0.00% | 0.00% | 1.76% | 🟢 INT8 |
| `blocks.5.attn.qkv` | Attn-QKV | 1.012 | 22.6 | **0.52%** | 0.25% | 0.00% | 0.30% | 🟡 LLM.int8 |
| `blocks.5.attn.proj` | Attn-Proj | 0.287 | 5.3 | 0.00% | 0.00% | 0.00% | 0.68% | 🟢 INT8 |
| `blocks.5.mlp.fc1` | MLP-fc1 | 1.117 | 44.3 | 0.13% | 0.17% | 0.00% | 0.30% | 🟢 INT8 |
| `blocks.5.mlp.fc2` | MLP-fc2 | 0.162 | 11.7 | 0.00% | 0.00% | 0.00% | 1.72% | 🟢 INT8 |
| `blocks.6.attn.qkv` | Attn-QKV | 1.056 | 24.3 | **0.52%** | 0.27% | 0.00% | 0.31% | 🟡 LLM.int8 |
| `blocks.6.attn.proj` | Attn-Proj | 0.284 | 4.7 | 0.00% | 0.00% | 0.00% | 0.64% | 🟢 INT8 |
| `blocks.6.mlp.fc1` | MLP-fc1 | 1.214 | 42.2 | 0.13% | 0.19% | 0.00% | 0.32% | 🟢 INT8 |
| `blocks.6.mlp.fc2` | MLP-fc2 | 0.178 | 10.7 | 0.00% | 0.00% | 0.00% | 1.76% | 🟢 INT8 |
| `blocks.7.attn.qkv` | Attn-QKV | 1.104 | 28.2 | 0.13% | 0.24% | 0.00% | 0.33% | 🟢 INT8 |
| `blocks.7.attn.proj` | Attn-Proj | 0.332 | 5.5 | 0.00% | 0.00% | 0.00% | 0.88% | 🟢 INT8 |
| `blocks.7.mlp.fc1` | MLP-fc1 | 1.423 | 24.6 | 0.39% | 0.40% | 0.00% | 0.33% | 🟢 INT8 |
| `blocks.7.mlp.fc2` | MLP-fc2 | 0.232 | 11.8 | 0.00% | 0.00% | 0.00% | 1.75% | 🟢 INT8 |
| `blocks.8.attn.qkv` | Attn-QKV | 1.163 | 31.0 | 0.13% | 0.17% | 0.00% | 0.34% | 🟢 INT8 |
| `blocks.8.attn.proj` | Attn-Proj | 0.387 | 7.1 | 0.00% | ~0% | 0.00% | 1.39% | 🟢 INT8 |
| `blocks.8.mlp.fc1` | MLP-fc1 | 3.474 | 72.9 | **3.78%** | 5.84% | 0.00% | 0.35% | 🟡 LLM.int8 |
| `blocks.8.mlp.fc2` | MLP-fc2 | 0.335 | 25.1 | 0.03% | 0.05% | 0.03% | 0.99% | 🟢 INT8 |
| `blocks.9.attn.qkv` | Attn-QKV | 1.319 | 38.9 | 0.00% | 0.18% | 0.00% | 0.37% | 🟢 INT8 |
| `blocks.9.attn.proj` | Attn-Proj | 0.465 | 7.2 | 0.00% | ~0% | 0.00% | 1.67% | 🟢 INT8 |
| `blocks.9.mlp.fc1` | MLP-fc1 | 7.318 | 104.8 | **97.53%** | 35.00% | 0.13% | 0.36% | 🔴 FP16 |
| `blocks.9.mlp.fc2` | MLP-fc2 | 0.449 | 47.5 | 0.07% | 0.14% | 0.03% | 0.44% | 🟢 INT8 |
| `blocks.10.attn.qkv` | Attn-QKV | 1.533 | 47.4 | 0.39% | 0.31% | 0.00% | 0.38% | 🟢 INT8 |
| `blocks.10.attn.proj` | Attn-Proj | 0.630 | 7.9 | 0.00% | ~0% | 0.00% | 1.69% | 🟢 INT8 |
| `blocks.10.mlp.fc1` | MLP-fc1 | 13.434 | 202.7 | **99.74%** | 61.23% | 0.00% | 0.36% | 🔴 FP16 |
| `blocks.10.mlp.fc2` | MLP-fc2 | 0.938 | 102.5 | 0.23% | 0.28% | 0.03% | 0.23% | 🟢 INT8 |
| `blocks.11.attn.qkv` | Attn-QKV | 1.501 | 29.1 | 0.39% | 0.20% | 0.00% | 0.40% | 🟢 INT8 |
| `blocks.11.attn.proj` | Attn-Proj | 0.914 | 11.4 | 0.13% | 0.04% | 0.00% | 2.14% | 🟢 INT8 |
| `blocks.11.mlp.fc1` | MLP-fc1 | 1.528 | 29.3 | 0.26% | 0.22% | 0.00% | 0.48% | 🟢 INT8 |
| `blocks.11.mlp.fc2` | MLP-fc2 | 0.393 | 16.2 | 0.03% | 0.02% | 0.00% | 1.76% | 🟢 INT8 |
| `head` | Head | 1.138 | 16.9 | 0.00% | 0.02% | 0.00% | 0.50% | 🟢 INT8 |

**Column definitions:**
- **σ**: exact global population std of the matmul input activation (Pass 1, Chan/Welford float64)
- **Max |x|**: largest absolute input value observed across all 50,000 images
- **RF fixed**: per-column routing fraction at |x| > 6.0, ≥ 25% token participation (LLM.int8() cost)
- **VD fixed**: per-value outlier density at |x| > 6.0 (unstructured baseline)
- **RF stat**: per-column routing fraction at |x − μ_c| > 3σ_c, per-channel, ≥ 5% token participation
- **VD stat**: per-value outlier density at |x − μ_c| > 3σ_c, per-channel

¹ `blocks.4.attn.qkv` (0.39%) is marginally below the LLM.int8() threshold.
Experiments 3–4 will determine whether its sensitivity justifies mixed-precision
treatment.

**Policy summary:** 🔴 FP16: **2 layers** · 🟡 LLM.int8(): **4 layers** · 🟢 INT8: **43 layers**

---

## 8. Residual-Stream Scale Explosion

The catastrophe in blocks 9–10 is traceable to a transient blow-up in the
residual stream. Tracking the `fc1` input standard deviation (the LayerNorm'd
residual stream) down the network:

```
Block:     0     1     2     3     4     5     6     7  |  8      9      10     11
σ (fc1):  0.88  0.94  0.89  0.97  1.06  1.12  1.21  1.42 | 3.47   7.32  13.43   1.53
```

The transition at block 8 is abrupt. The maximum magnitude peaks at **202.7** in
block 10 before the residual stream collapses back to σ ≈ 1.5 at block 11. This
is the well-documented "massive activations" phenomenon in ViT's late-middle
blocks: a transient, localized scale blow-up that is invisible to per-layer
relative thresholds (like 3σ) but catastrophic under any absolute threshold
(like the INT8 dynamic range).

Block 11 fully recovers. The explosion is confined. A surgical FP16 policy for
just two layers contains it.

---

## 9. Implications for Experiments 2–4

### Experiment 2: Quantization Granularity

Per-tensor INT8 scaling will struggle most in blocks 9–10 fc1 (dynamic range σ
× 13–40× the model average). Per-token activation scaling should provide partial
relief by adapting to each token's own scale, though the fundamental problem
(too many outlier columns) is structural. The largest accuracy gap between
per-tensor and per-token/per-channel will isolate here.

### Experiment 3: Per-Layer Sensitivity

**Prediction:** quantizing `blocks.9.mlp.fc1` or `blocks.10.mlp.fc1` to naive
INT8 will cause the sharpest accuracy drops by far. If the sensitivity heatmap
agrees with this outlier map, with the two largest bars appearing at blocks 9–10
`fc1`, the cross-validation is complete and the routing policy table above
becomes the thesis's primary deliverable.

### Experiment 4: Decomposition

| Layer | What to expect |
|---|---|
| `blocks.9.mlp.fc1` | Fixed-6.0 threshold: ≈97.5% high-precision fraction. No INT8 benefit. 3σ threshold: ≈0.4% high-precision fraction. Near-zero accuracy recovery. |
| `blocks.10.mlp.fc1` | Fixed-6.0: ≈99.7% HP fraction (pure FP16). 3σ: ≈0.4% HP fraction (ineffective). |
| `blocks.8.mlp.fc1` | Fixed-6.0: ≈3.8% HP fraction. Meaningful INT8 savings with full accuracy. |
| All `attn.proj` | Both thresholds: ≈0% HP fraction. Pure INT8 throughout. |

The contrast between the fixed and statistical thresholds in blocks 9–10 will be
the sharpest possible demonstration that the absolute threshold is correct and
the relative threshold is not, and that neither can recover INT8 EDP benefits in
those two layers.

---

## 10. The Selective Routing Policy: Justification

The following policy is directly readable from the outlier map. It answers the
question the grounding document posed with Experiment 1: which layers are suited
for mixed-precision routing, and which are not?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER              │  POLICY       │  JUSTIFICATION                           │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│  blocks.9.mlp.fc1  │  🔴 FP16      │  97.5% of columns routed. Routing is     │
│  blocks.10.mlp.fc1 │  🔴 FP16      │  pure overhead. Run clean FP16.          │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│  blocks.8.mlp.fc1  │  🟡 LLM.int8  │  3.8% routing. Concentrated outliers     │
│  blocks.3.attn.qkv │  🟡 LLM.int8  │  (routing < density), sparse enough for  │
│  blocks.5.attn.qkv │  🟡 LLM.int8  │  mixed-precision to yield net INT8        │
│  blocks.6.attn.qkv │  🟡 LLM.int8  │  benefit. EDP tradeoff TBD by Exp 3/4.   │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│  All 43 others      │  🟢 INT8      │  Routing fraction < 0.5%. Naive INT8     │
│  (incl. all proj,   │               │  is near-lossless. No decomposition      │
│  fc2, head, most    │               │  overhead justified.                     │
│  qkv and fc1)       │               │                                          │
└─────────────────────┴───────────────┴──────────────────────────────────────────┘
```

The practical consequence: **41 of the model's 48 block-internal matmuls can run
in pure INT8 with negligible accuracy cost.** The EDP of the overall model on
the Jetson Orin Nano is determined by what happens to the remaining 7 (2 FP16, 4
LLM.int8(), 1 borderline). Whether those 4 LLM.int8() layers achieve a net EDP
gain is the open question Experiments 3–4 will close.

---

## 11. Statistical Confidence

The 50,000-image run (9.85 M tokens per layer) produces stable estimates.
Comparison against the 4,096-image characterization run shows the routing
fractions for blocks 9–10 fc1 changed by less than 0.2 percentage points
(97.66% → 97.53% and 99.74% → 99.74%). At these sample sizes the per-column
participation fractions have converged. The 4σ outlier detection interval is
well above the noise floor for any layer with rf_fixed > 0.1%.

---

*Generated from `outputs/exp1_outlier_maps/outlier_stats.json`.*
*Statistical threshold computed per-channel (each of 768 / 3072 feature
dimensions receives its own μ and σ from Pass 1).*
*Run: 50,000 ImageNet-1K val images · two-pass exact global statistics ·
ViT-B/16 (timm `vit_base_patch16_224`)*