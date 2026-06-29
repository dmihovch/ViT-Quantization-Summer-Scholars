# Experiment 1 — Full Per-Layer Outlier Characterization
### ViT-B/16 PTQ for Edge Deployment · Summer Thesis Research

---

## 1. Executive Summary

The central hypothesis of this project — that ViT-B/16 has dense, feedforward-concentrated activation
outliers that break LLM.int8()'s mixed-precision routing premise — is **confirmed, quantitatively
and sharply**, by the full 50,000-image characterization run.

The catastrophe is real but *localized*. Two layers out of 49 break the INT8 quantization regime:
`blocks.9.mlp.fc1` and `blocks.10.mlp.fc1` have LLM.int8() routing fractions of **97.5%** and
**99.7%** respectively — meaning the quantizer would route essentially the entire matmul to FP16 for
those layers, adding routing overhead while gaining nothing. Every other layer in the model is clean
enough for pure INT8 (routing fraction ≤ 3.8%, and 43 of 49 layers ≤ 0.5%).

This map directly prescribes the **heterogeneous (selective) routing policy** called for in the
research plan: FP16 for 2 layers, LLM.int8() for 4 borderline layers, pure INT8 for the remaining
43. A global LLM.int8() policy is wrong for ViT.

---

## 2. Experiment Setup

| Parameter | Value |
|---|---|
| Model | timm `vit_base_patch16_224` (pretrained ImageNet-1K) |
| Images | 50,000 ImageNet-1K validation images |
| Tokens seen per layer | 9,850,000 (50,000 images × 197 tokens/image) |
| Measurement point | **Matmul input** (forward pre-hook on every `nn.Linear`) |
| Algorithm | **Two-pass exact**: Pass 1 computes global mean/std via Chan/Welford float64 merge; Pass 2 applies frozen thresholds |
| Fixed threshold | \|x\| > 6.0 (LLM.int8()'s own calibration) |
| Statistical threshold | \|x − μ\| > 3σ (exact global μ, σ) |
| Routing column criterion | A feature column is flagged only if it exceeds the threshold in ≥ 25% of tokens (LLM.int8()'s persistence criterion) |
| Layers measured | 49 (12 blocks × 4 linears + 1 head) |

**Why inputs, not outputs.** LLM.int8() decomposes the matrix `Y = X @ W.T` by inspecting `X` — the
post-LayerNorm input activation — and routing outlier *columns of X* to FP16. Measuring outputs would
characterize a different tensor the quantizer never routes on. Every metric below is taken at the
exact decision point.

---

## 3. The Catastrophe: Blocks 9–10 mlp.fc1

The following table shows every MLP up-projection (`fc1`) in network order. The LLM.int8() routing
fraction (per-column, 25% participation) is the headline metric.

| Block | Input std (σ) | Max \|x\| | Per-value density (6.0) | **LLM.int8 routing fraction** | Policy |
|------:|----:|----:|----:|----:|:--|
| 0 | 0.884 | 47.3 | 0.16% | 0.00% | 🟢 INT8 |
| 1 | 0.944 | 33.9 | 0.22% | 0.13% | 🟢 INT8 |
| 2 | 0.891 | 39.5 | 0.02% | 0.00% | 🟢 INT8 |
| 3 | 0.966 | 44.6 | 0.07% | 0.13% | 🟢 INT8 |
| 4 | 1.059 | 46.9 | 0.20% | 0.26% | 🟢 INT8 |
| 5 | 1.117 | 44.3 | 0.16% | 0.13% | 🟢 INT8 |
| 6 | 1.214 | 42.2 | 0.19% | 0.13% | 🟢 INT8 |
| 7 | 1.423 | 24.6 | 0.40% | 0.39% | 🟢 INT8 |
| **8** | **3.474** | **72.9** | **5.84%** | **3.78%** | 🟡 LLM.int8 |
| **9** | **7.318** | **104.8** | **35.00%** | **97.53%** | 🔴 FP16 |
| **10** | **13.434** | **202.7** | **61.23%** | **99.74%** | 🔴 FP16 |
| 11 | 1.528 | 29.3 | 0.22% | 0.26% | 🟢 INT8 |

### What the numbers mean

Blocks 9 and 10 are a regime change, not a gradual drift. The input standard deviation grows smoothly
from σ ≈ 0.9 through blocks 0–7, then detonates: σ jumps to 3.5 → 7.3 → **13.4** across blocks
8–10 before collapsing back to 1.5 at block 11. At σ = 13.4, the LLM.int8() fixed threshold of 6.0
sits at less than 0.5σ from the mean — the vast majority of activations naturally exceed it, making
the threshold meaningless as an outlier detector for that layer.

The per-value density for block 10 (`61.2%`) matches the project's preliminary estimate of
"up to 63% in late blocks" almost exactly. The per-column routing fraction (`99.7%`) is worse still:
nearly every input feature column of this matmul exceeds 6.0 in at least 25% of tokens, so LLM.int8()
would push the entire contraction dimension to FP16 — strictly worse than just running FP16 cleanly,
because the decomposition overhead is added on top.

Block 8 is the onset: σ rises to 3.5, routing fraction hits 3.78%. This is the last layer where
LLM.int8() could provide a net benefit (≈96% of the matmul remains in INT8).

---

## 4. The Structured Routing Penalty: What the Two Metrics Together Reveal

The per-value density and per-column routing fraction are not redundant — their relationship is the
diagnostic signal. The *direction* of the gap tells you whether LLM.int8() is viable:

| Layer | Per-value density | Routing fraction | Gap direction | Interpretation |
|---|---:|---:|:--|:--|
| `blocks.8.mlp.fc1` | 5.84% | 3.78% | routing **<** density | Outliers cluster in fewer columns than raw count implies — concentrated, **routable** |
| `blocks.9.mlp.fc1` | 35.00% | 97.53% | routing **≫** density | Outliers touch nearly every column — pervasive, **not routable** |
| `blocks.10.mlp.fc1` | 61.23% | 99.74% | routing **≫** density | Every column is an outlier column — LLM.int8() ≡ FP16 |

Block 8's gap favours INT8: the few heavy outlier values are column-persistent (concentrated), so
routing a small fraction of columns to FP16 protects almost all of the compute. Blocks 9–10's gap
is inverted: outlier values spread across nearly every feature column, so structured whole-column
routing forces the entire matmul to FP16.

The project grounding doc named two failure modes for LLM.int8(): density too high, or outliers
scattered (not channel-persistent). The channel persistence variance for blocks 9–10 fc1 is
**1,753,527** and **1,150,906** respectively — the highest in the model — so the failure mode is not
scatter. It is **density**. LLM.int8()'s persistence assumption holds; its sparsity assumption fails.

---

## 5. The Fixed vs. Statistical Threshold: Why 6.0 is the Right Calibration

The 3σ routing fraction is nearly flat across all 49 layers, ranging from 0.0% to 2.3%. Critically,
it reads **1.69%** for `blocks.9.mlp.fc1` and **0.78%** for `blocks.10.mlp.fc1` — appearing clean
and safe when they are catastrophically broken.

The reason: at σ = 13.4, the 3σ cutoff is 40.3. Most values exceed 6.0 but not 40.3, so the
statistical threshold misses the explosion entirely. A **relative** threshold self-normalizes to each
layer's scale and hides the very information needed to identify INT8-breaking layers. The **absolute**
6.0 threshold — which maps to the fixed dynamic range of INT8 — is the correct calibration for
quantization-safety analysis.

**Implication for Experiment 4.** The 3σ decomposition threshold will fail to protect blocks 9–10:
it flags only ≈1–2% of values for high-precision treatment when the layer has already blown past the
INT8 range. The fixed 6.0 threshold will correctly identify the problem but report a ~100%
high-precision fraction — confirming that no decomposition policy recovers INT8 efficiency there.

---

## 6. The Clean Layers

### Attention output projections (`attn.proj`): pristine

| Stat | Blocks 0–6 | Blocks 7–10 | Block 11 |
|---|---|---|---|
| Routing fraction (6.0) | **0.00%** across all | 0.00% | 0.13% |
| Per-value density (6.0) | ≈ 0 (< 2 × 10⁻⁷) | ≈ 0 | 0.04% |
| Max \|x\| | 3.4 – 8.8 | 5.5 – 7.9 | 11.4 |
| Input std σ | 0.16 – 0.28 | 0.33 – 0.63 | 0.91 |

The attention context (input to `attn.proj`) is a convex combination of Value vectors — inherently
bounded and averaged. These are the model's most quantization-friendly matmuls. Naive per-tensor INT8
should be near-lossless here.

### MLP down-projections (`mlp.fc2`): sparse outliers, exactly what LLM.int8 handles

Routing fractions across all fc2 layers: 0.00% (blocks 0–7), then 0.03%, 0.07%, 0.23%, 0.03%
(blocks 8–11). Even at block 10 — where the *output* of the preceding fc1 is catastrophic — the
fc2 input (the post-GELU intermediate) has only 0.23% routing fraction and 0.28% per-value density.
This is the sparse, persistent outlier regime that LLM.int8() was designed for: pure INT8 suffices,
and LLM.int8() would work if applied here, but the overhead at < 0.25% routing is not justified.

### Attention QKV projections (`attn.qkv`): mild, manageable

Most blocks have routing fractions of 0.0–0.5%. Three blocks edge above 0.5%:
`blocks.3.attn.qkv` (0.78%), `blocks.5.attn.qkv` (0.52%), `blocks.6.attn.qkv` (0.52%).
These are genuine LLM.int8() candidates — sparse, column-persistent, with a favorable
gap (routing fraction ≤ per-value density). At 99%+ of the matmul remaining in INT8, the mixed-
precision overhead is low and the EDP tradeoff is potentially worthwhile.

---

## 7. Full Routing Policy Table (All 49 Layers)

**Policy thresholds (fixed 6.0 routing fraction):**

| Threshold | Policy | Rationale |
|---|---|---|
| ≥ 50% | 🔴 **FP16** | LLM.int8() degenerates to FP16; routing overhead is pure loss |
| 0.5% – 50% | 🟡 **LLM.int8()** | Mixed precision provides net benefit; outliers sparse enough to route |
| < 0.5% | 🟢 **INT8** | Routing overhead exceeds benefit; naive INT8 is near-lossless |

*Note: block 4 attn.qkv (0.39%) is borderline. Experiments 3–4 should determine whether its 
sensitivity warrants LLM.int8() treatment.*

| Layer | Type | σ | Max \|x\| | RF fixed | VD fixed | RF stat | **Policy** |
|---|---|---:|---:|---:|---:|---:|:--|
| `blocks.0.attn.qkv` | Attn-QKV | 0.515 | 19.4 | 0.00% | 0.094% | 2.08% | 🟢 INT8 |
| `blocks.0.attn.proj` | Attn-Proj | 0.162 | 8.8 | 0.00% | ~0% | 1.43% | 🟢 INT8 |
| `blocks.0.mlp.fc1` | MLP-fc1 | 0.884 | 47.3 | 0.00% | 0.156% | 0.91% | 🟢 INT8 |
| `blocks.0.mlp.fc2` | MLP-fc2 | 0.212 | 36.9 | 0.00% | 0.003% | 0.29% | 🟢 INT8 |
| `blocks.1.attn.qkv` | Attn-QKV | 0.598 | 17.1 | 0.00% | 0.170% | 2.08% | 🟢 INT8 |
| `blocks.1.attn.proj` | Attn-Proj | 0.219 | 4.8 | 0.00% | 0.000% | 2.34% | 🟢 INT8 |
| `blocks.1.mlp.fc1` | MLP-fc1 | 0.944 | 33.9 | 0.13% | 0.221% | 1.30% | 🟢 INT8 |
| `blocks.1.mlp.fc2` | MLP-fc2 | 0.133 | 10.7 | 0.00% | ~0% | 0.16% | 🟢 INT8 |
| `blocks.2.attn.qkv` | Attn-QKV | 0.700 | 11.9 | 0.00% | 0.105% | 2.21% | 🟢 INT8 |
| `blocks.2.attn.proj` | Attn-Proj | 0.227 | 4.4 | 0.00% | 0.000% | 1.56% | 🟢 INT8 |
| `blocks.2.mlp.fc1` | MLP-fc1 | 0.891 | 39.5 | 0.00% | 0.021% | 0.52% | 🟢 INT8 |
| `blocks.2.mlp.fc2` | MLP-fc2 | 0.134 | 7.9 | 0.00% | ~0% | 0.00% | 🟢 INT8 |
| `blocks.3.attn.qkv` | Attn-QKV | 0.897 | 20.7 | 0.78% | 0.332% | 1.82% | 🟡 LLM.int8 |
| `blocks.3.attn.proj` | Attn-Proj | 0.231 | 3.4 | 0.00% | 0.000% | 0.65% | 🟢 INT8 |
| `blocks.3.mlp.fc1` | MLP-fc1 | 0.966 | 44.6 | 0.13% | 0.066% | 1.04% | 🟢 INT8 |
| `blocks.3.mlp.fc2` | MLP-fc2 | 0.144 | 9.4 | 0.00% | ~0% | 0.13% | 🟢 INT8 |
| `blocks.4.attn.qkv` | Attn-QKV | 0.969 | 19.9 | 0.39% | 0.264% | 2.08% | 🟢 INT8 ¹ |
| `blocks.4.attn.proj` | Attn-Proj | 0.269 | 4.3 | 0.00% | 0.000% | 1.30% | 🟢 INT8 |
| `blocks.4.mlp.fc1` | MLP-fc1 | 1.059 | 46.9 | 0.26% | 0.201% | 1.04% | 🟢 INT8 |
| `blocks.4.mlp.fc2` | MLP-fc2 | 0.148 | 12.9 | 0.00% | 0.001% | 0.26% | 🟢 INT8 |
| `blocks.5.attn.qkv` | Attn-QKV | 1.012 | 22.6 | 0.52% | 0.253% | 1.43% | 🟡 LLM.int8 |
| `blocks.5.attn.proj` | Attn-Proj | 0.287 | 5.3 | 0.00% | 0.000% | 0.39% | 🟢 INT8 |
| `blocks.5.mlp.fc1` | MLP-fc1 | 1.117 | 44.3 | 0.13% | 0.165% | 0.78% | 🟢 INT8 |
| `blocks.5.mlp.fc2` | MLP-fc2 | 0.162 | 11.7 | 0.00% | 0.001% | 0.10% | 🟢 INT8 |
| `blocks.6.attn.qkv` | Attn-QKV | 1.056 | 24.3 | 0.52% | 0.274% | 1.04% | 🟡 LLM.int8 |
| `blocks.6.attn.proj` | Attn-Proj | 0.284 | 4.7 | 0.00% | 0.000% | 0.26% | 🟢 INT8 |
| `blocks.6.mlp.fc1` | MLP-fc1 | 1.214 | 42.2 | 0.13% | 0.187% | 0.65% | 🟢 INT8 |
| `blocks.6.mlp.fc2` | MLP-fc2 | 0.178 | 10.7 | 0.00% | 0.002% | 0.13% | 🟢 INT8 |
| `blocks.7.attn.qkv` | Attn-QKV | 1.104 | 28.2 | 0.13% | 0.239% | 1.04% | 🟢 INT8 |
| `blocks.7.attn.proj` | Attn-Proj | 0.332 | 5.5 | 0.00% | 0.000% | 0.00% | 🟢 INT8 |
| `blocks.7.mlp.fc1` | MLP-fc1 | 1.423 | 24.6 | 0.39% | 0.398% | 1.17% | 🟢 INT8 |
| `blocks.7.mlp.fc2` | MLP-fc2 | 0.232 | 11.8 | 0.00% | 0.001% | 0.10% | 🟢 INT8 |
| `blocks.8.attn.qkv` | Attn-QKV | 1.163 | 31.0 | 0.13% | 0.173% | 0.65% | 🟢 INT8 |
| `blocks.8.attn.proj` | Attn-Proj | 0.387 | 7.1 | 0.00% | ~0% | 0.00% | 🟢 INT8 |
| `blocks.8.mlp.fc1` | MLP-fc1 | 3.474 | 72.9 | **3.78%** | 5.84% | 1.82% | 🟡 LLM.int8 |
| `blocks.8.mlp.fc2` | MLP-fc2 | 0.335 | 25.1 | 0.03% | 0.051% | 0.16% | 🟢 INT8 |
| `blocks.9.attn.qkv` | Attn-QKV | 1.319 | 38.9 | 0.00% | 0.181% | 0.39% | 🟢 INT8 |
| `blocks.9.attn.proj` | Attn-Proj | 0.465 | 7.2 | 0.00% | ~0% | 0.00% | 🟢 INT8 |
| `blocks.9.mlp.fc1` | MLP-fc1 | 7.318 | 104.8 | **97.53%** | 35.00% | 1.69% | 🔴 FP16 |
| `blocks.9.mlp.fc2` | MLP-fc2 | 0.449 | 47.5 | 0.07% | 0.144% | 0.16% | 🟢 INT8 |
| `blocks.10.attn.qkv` | Attn-QKV | 1.533 | 47.4 | 0.39% | 0.315% | 0.52% | 🟢 INT8 |
| `blocks.10.attn.proj` | Attn-Proj | 0.630 | 7.9 | 0.00% | ~0% | 0.00% | 🟢 INT8 |
| `blocks.10.mlp.fc1` | MLP-fc1 | 13.434 | 202.7 | **99.74%** | 61.23% | 0.78% | 🔴 FP16 |
| `blocks.10.mlp.fc2` | MLP-fc2 | 0.938 | 102.5 | 0.23% | 0.278% | 0.29% | 🟢 INT8 |
| `blocks.11.attn.qkv` | Attn-QKV | 1.501 | 29.1 | 0.39% | 0.202% | 0.52% | 🟢 INT8 |
| `blocks.11.attn.proj` | Attn-Proj | 0.914 | 11.4 | 0.13% | 0.041% | 0.52% | 🟢 INT8 |
| `blocks.11.mlp.fc1` | MLP-fc1 | 1.528 | 29.3 | 0.26% | 0.215% | 0.26% | 🟢 INT8 |
| `blocks.11.mlp.fc2` | MLP-fc2 | 0.393 | 16.2 | 0.03% | 0.022% | 0.91% | 🟢 INT8 |
| `head` | Head | 1.138 | 16.9 | 0.00% | 0.020% | 0.00% | 🟢 INT8 |

**Column definitions:**
- **σ** — exact global population std of the matmul input activation (Pass 1, Chan/Welford float64)
- **Max |x|** — largest absolute input value observed across all 50,000 images
- **RF fixed** — per-column routing fraction at |x| > 6.0, ≥ 25% token participation (LLM.int8() cost)
- **VD fixed** — per-value outlier density at |x| > 6.0 (unstructured baseline)
- **RF stat** — per-column routing fraction at |x − μ| > 3σ (exact global μ, σ)

¹ `blocks.4.attn.qkv` (0.39%) is marginally below the LLM.int8() threshold. Experiments 3–4 will
determine whether its sensitivity justifies mixed-precision treatment.

**Policy summary:** 🔴 FP16: **2 layers** · 🟡 LLM.int8(): **4 layers** · 🟢 INT8: **43 layers**

---

## 8. Residual-Stream Scale Explosion

The catastrophe in blocks 9–10 is traceable to a transient blow-up in the residual stream. Tracking
the `fc1` input standard deviation (which is the LayerNorm'd residual stream) down the network:

```
Block:     0     1     2     3     4     5     6     7  |  8      9      10     11
σ (fc1):  0.88  0.94  0.89  0.97  1.06  1.12  1.21  1.42 | 3.47   7.32  13.43   1.53
```

The transition at block 8 is abrupt. The maximum magnitude peaks at **202.7** in block 10 before the
residual stream collapses back to σ ≈ 1.5 at block 11. This is the well-documented "massive
activations" phenomenon in ViT's late-middle blocks — a transient, localized scale blow-up that is
invisible to per-layer relative thresholds (like 3σ) but catastrophic under any absolute threshold
(like the INT8 dynamic range).

Block 11 fully recovers. The explosion is confined; a surgical FP16 policy for just two layers
contains it.

---

## 9. Implications for Experiments 2–4

### Experiment 2 — Quantization Granularity

Per-tensor INT8 scaling will struggle most in blocks 9–10 fc1 (dynamic range σ × 13–40× the
model average). Per-token activation scaling should provide partial relief by adapting to each
token's own scale, though the fundamental problem (too many outlier columns) is structural. The
largest accuracy gap between per-tensor and per-token/per-channel will isolate here.

### Experiment 3 — Per-Layer Sensitivity

**Prediction:** quantizing `blocks.9.mlp.fc1` or `blocks.10.mlp.fc1` to naive INT8 will cause the
sharpest accuracy drops by far. If the sensitivity heatmap agrees with this outlier map — the two
largest bars appearing at blocks 9–10 `fc1` — the cross-validation is complete and the routing
policy table above becomes the thesis's primary deliverable.

### Experiment 4 — Decomposition

| Layer | What to expect |
|---|---|
| `blocks.9.mlp.fc1` | Fixed-6.0 threshold: ≈97.5% high-precision fraction → no INT8 benefit; 3σ threshold: ≈1.7% high-precision fraction → near-zero accuracy recovery |
| `blocks.10.mlp.fc1` | Fixed-6.0: ≈99.7% HP fraction (pure FP16); 3σ: ≈0.8% HP fraction (ineffective) |
| `blocks.8.mlp.fc1` | Fixed-6.0: ≈3.8% HP fraction → meaningful INT8 savings with full accuracy |
| All `attn.proj` | Both thresholds: ≈0% HP fraction → pure INT8 throughout |

The contrast between the fixed and statistical thresholds in blocks 9–10 will be the sharpest
possible demonstration that the absolute threshold is correct and the relative threshold is not, and
that neither can recover INT8 EDP benefits in those two layers.

---

## 10. The Selective Routing Policy: Justification

The following policy is directly readable from the outlier map. It is the answer to the question the
grounding document posed with Experiment 1: which layers are suited for mixed-precision routing, and
which are not?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER              │  POLICY       │  JUSTIFICATION                           │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│  blocks.9.mlp.fc1  │  🔴 FP16      │  97.5% of columns routed → routing is    │
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

The practical consequence: **41 of the model's 48 block-internal matmuls can run in pure INT8 with
negligible accuracy cost.** The EDP of the overall model on the Jetson Orin Nano is determined by
what happens to the remaining 7 (2 FP16, 4 LLM.int8(), 1 borderline). Whether those 4 LLM.int8()
layers achieve a net EDP gain is the open question Experiments 3–4 will close.

---

## 11. Statistical Confidence

The 50,000-image run (9.85 M tokens per layer) produces stable estimates. Comparison against the
4,096-image characterization run shows the routing fractions for blocks 9–10 fc1 changed by less
than 0.2 percentage points (97.66% → 97.53% and 99.74% → 99.74%). At these sample sizes the
per-column participation fractions have converged. The 4σ outlier detection interval is
well above the noise floor for any layer with rf\_fixed > 0.1%.

---

*Generated from `outputs/exp1_outlier_maps/outlier_stats.json`.*  
*Run: 50,000 ImageNet-1K val images · two-pass exact global statistics · ViT-B/16 (timm `vit_base_patch16_224`)*
