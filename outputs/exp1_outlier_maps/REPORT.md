# Experiment 1: Full Per-Layer Outlier Characterization

### ViT-B/16 PTQ for Edge Deployment, Summer Thesis Research

## 1. Executive Summary

ViT-B/16's INT8-breaking activation outliers are real, dense, and concentrated
in two feedforward up-projection layers (`blocks.9.mlp.fc1` and
`blocks.10.mlp.fc1`), exactly as hypothesized. Of the model's 49 linear layers,
2 require FP16, 3 benefit from `LLM.int8()` mixed precision, and 44 can run in
pure INT8 with a routing fraction low enough to predict negligible accuracy
loss (pending Experiment 3 validation). The catastrophe is a transient
residual-stream scale explosion confined to blocks 8-10 that collapses again
by block 11.

**Run size note:** this report reflects a 5,000-image ImageNet-1K validation
run (985,000 tokens per layer), not the full 50,000-image thesis-print run
used in earlier drafts. The 5,000-image run was used to re-validate the
pipeline quickly after a channel-persistence-variance bug fix (Section 4). A
full 50,000-image re-run is still the right step before any number here is
treated as final; see `docs/advisor-touchpoint-guide.md` for the comparison
between the two run sizes and what changed.

## 2. Experiment Setup

- **Model:** `vit_base_patch16_224` (timm), ImageNet-1K pretrained.
- **Data:** 5,000 ImageNet-1K validation images, 197 tokens/image (196 patches
  + 1 CLS token), 985,000 tokens per layer.
- **Layers measured:** all 49 linear projections (`attn.qkv`, `attn.proj`,
  `mlp.fc1`, `mlp.fc2` per block, plus `head`), via forward pre-hooks on the
  matmul input.
- **Thresholds:** fixed `|x| > 6.0` (the `LLM.int8()` paper's threshold,
  faithful 25% column-participation bar) and per-channel `3σ` (research
  diagnostic, 5% column-participation bar).
- **Method:** two-pass exact statistics. Pass 1 computes each layer's exact
  per-channel mean/std (Chan/Welford merge, float64). Pass 2 freezes those
  statistics and counts outliers and routing fractions for both thresholds.

## 3. The Catastrophe: Blocks 9-10 mlp.fc1

| Block | Input σ | Max \|x\| | Per-value density (6.0) | **Routing fraction (6.0)** | Policy |
|------:|--------:|----------:|------------------------:|---------------------------:|:--|
| 0-7 | 0.83 → 1.33 | 23-47 | < 0.4% | ≤ 0.26% | 🟢 INT8 |
| **8** | **3.06** | **62.3** | **5.72%** | **3.78%** | 🟡 LLM.int8 |
| **9** | **6.42** | **104.1** | **34.4%** | **97.66%** | 🔴 FP16 |
| **10** | **11.68** | **201.0** | **60.45%** | **99.74%** | 🔴 FP16 |
| 11 | 1.28 | 29.3 | 0.25% | 0.26% | 🟢 INT8 |

### What the numbers mean

The fc1 input standard deviation tracks a transient blow-up down the network:

```
Block:     0     1     2     3     4     5     6     7  |  8      9      10     11
σ (fc1):  0.87  0.85  0.83  0.89  0.97  1.04  1.15  1.33 | 3.06   6.42  11.68   1.28
```

σ climbs gradually and monotonically from block 0 (0.87) to block 7 (1.33), a
1.5x increase over 7 blocks, then accelerates sharply from block 7 to block 10
(1.33 → 11.68, an 8.8x increase in 3 blocks), before collapsing 9.1x in a
single block (11.68 → 1.28 at block 11). The maximum observed activation
magnitude peaks at 201.0 at block 10. Block 11 fully recovers to 29.3. This is
consistent with the "massive activations" phenomenon documented in
Transformers: high-norm signals emerging at specific depths, here in the FFN
up-projection, and collapsing again once the next LayerNorm renormalizes the
residual stream.

The mechanism: the residual stream applies attention and the FFN sequentially,
`x'_i = x_i + Attention(LN(x_i))` then `x_{i+1} = x'_i + FFN(LN(x'_i))`.
LayerNorm normalizes each token's features before each sublayer, but the
sublayer's output is added back into the residual stream without
normalization. In blocks 9-10, the FFN `fc1` output escalates in magnitude and
this scale is deposited directly into the residual stream via the skip
connection. The next LayerNorm (entering block 11) renormalizes it. The
explosion is produced by the interaction of the FFN weights and the
accumulated residual signal at those specific depths, not by an unusually
large input image.

## 4. The Structured Routing Penalty: What the Two Metrics Together Reveal

The gap between per-value density and per-column routing fraction is itself
diagnostic:

| Layer | Density | Routing | Direction | Meaning |
|---|---:|---:|:--|:--|
| `blocks.8.mlp.fc1` | 5.72% | 3.78% | routing **<** density | Outliers concentrated. Routable. |
| `blocks.9.mlp.fc1` | 34.4% | 97.66% | routing **≫** density | Outliers everywhere. Not routable. |
| `blocks.10.mlp.fc1` | 60.45% | 99.74% | routing **≫** density | Every column is an outlier. `LLM.int8()` is equivalent to FP16. |

At block 8, outliers cluster efficiently into a minority of columns, so the
routing fraction is lower than the raw density. At blocks 9-10, the outliers
have saturated the feature space: nearly every column needs routing, making
the whole-column constraint pointless and `LLM.int8()` equivalent to running
the layer in plain FP16.

**Channel persistence, corrected.** An earlier version of this analysis
computed channel-persistence variance incorrectly (it averaged a per-batch
variance across batches rather than computing the variance once over the full
run; see `docs/advisor-touchpoint-guide.md` Section 4.8 for the bug and fix).
With the corrected metric, the three highest-persistence layers in the model
are blocks 9, 8, and 10, in that order (9.65×10⁹, 9.54×10⁹, and 5.35×10⁹).
Block 8 is now nearly tied with block 9 on this metric despite a routing
fraction two orders of magnitude lower, which shows that concentration within
columns is comparably high across all three layers. What separates a routable
layer (block 8) from the unroutable ones (blocks 9-10) is density (the share
of columns that clear the participation bar), not how tightly outliers cluster
within those columns. The `LLM.int8()` persistence assumption holds across all
three layers; only the sparsity assumption fails, and only at blocks 9-10.

This metric is computed in absolute outlier-count units, so it scales with how
many total outliers a layer has, not only with how unevenly they are
distributed. The ranking above is meaningful because blocks 8, 9, and 10 all
have a substantial outlier count; it should not be used to compare a
high-outlier-count layer against a near-zero-outlier-count layer without
normalizing first.

## 5. The Fixed vs. Statistical Threshold: Why 6.0 Is the Right Calibration

At σ = 11.68 (block 10), the 3σ cutoff is approximately 35.0, so the relative
threshold misses the explosion entirely: it self-normalizes to the layer's own
inflated scale. The absolute 6.0 threshold, which maps to INT8's fixed dynamic
range, correctly flags the broken layers.

Under the per-channel 3σ threshold with the 5% participation bar, the routing
fraction is effectively zero everywhere: only two layers register any signal
at all (`blocks.11.attn.qkv` and `blocks.11.mlp.fc1`, 0.13% each), and the
catastrophic blocks 9-10 `mlp.fc1` appear clean (0.00% each). The per-value
statistical density sits in a narrow 0.29%-0.54% band across all fc1 layers
regardless of σ, including blocks 9-10. A relative threshold detects roughly
the same tail fraction everywhere, which is exactly why it is inappropriate
for quantization analysis: it treats the benign block 0 and the catastrophic
block 10 as comparably outlier-prone.

**Recommendation:** the 3σ/statistical threshold should not be used as a
second decomposition arm in Experiment 4. It cannot detect the blocks 9-10
explosion by mathematical construction (it is defined relative to each layer's
own σ). See `docs/advisor-touchpoint-guide.md` Section 4.5 for the full
reasoning, pending advisor sign-off.

## 6. The Clean Layers

### Attention output projections (`attn.proj`): pristine

Every `attn.proj` layer has a routing fraction of 0.00% (with one exception:
`blocks.11.attn.proj` at 0.13%, still far below the 0.5% threshold). Max
magnitudes stay under 12 for blocks 0-10 and reach 9.3 at block 11. These
layers see the post-attention, concatenated multi-head output, a
mechanistically different tensor from the qkv projections, and show no
outlier behavior at all.

### MLP down-projections (`mlp.fc2`): sparse outliers, exactly what LLM.int8 handles

Every `mlp.fc2` layer has a routing fraction at or below 0.29%. Max magnitudes
rise with depth (block 10 reaches 92.5) but never trigger persistent
column-level routing. This is the textbook case `LLM.int8()` was designed for:
occasional large values that never persist in a specific column long enough to
justify routing.

### Attention QKV projections (`attn.qkv`): mild, manageable, and sample-size sensitive

Most `attn.qkv` layers sit well under the 0.5% INT8 cutoff. Two clearly clear
it: `blocks.3.attn.qkv` (0.78%) and `blocks.11.attn.qkv` (0.52%). A cluster of
four more sit at exactly 0.39%, just under the cutoff: blocks 4, 5, 6, and 10.
Comparing this run (5,000 images) against the historical 50,000-image run
shows that this cluster's bucket membership is not stable: `blocks.5` and
`blocks.6` were at 0.52% (LLM.int8()) in the 50K run and 0.39% (INT8) here;
`blocks.11` was at 0.39% (INT8) in the 50K run and 0.52% (LLM.int8()) here.
This is a real boundary effect, not a sign of a measurement bug: the
underlying routing-fraction estimate only moved by 0.13 percentage points in
each case. The policy bucket for this qkv cluster should be treated as
provisional until a larger run is repeated.

## 7. Full Routing Policy Table (All 49 Layers)

**Policy thresholds (fixed 6.0 routing fraction):**

| Threshold | Policy | Rationale |
|---|---|---|
| ≥ 50% | 🔴 **FP16** | LLM.int8() degenerates to FP16. Routing overhead is pure loss. |
| 0.5% - 50% | 🟡 **LLM.int8()** | Mixed precision provides net benefit. Outliers sparse enough to route. |
| < 0.5% | 🟢 **INT8** | Routing overhead exceeds benefit. Naive INT8 is near-lossless. |

*Note: `blocks.4/5/6/10.attn.qkv` (all at 0.39%) form a borderline cluster.
Experiments 3-4 should determine whether their sensitivity warrants
`LLM.int8()` treatment, and a larger re-run should confirm the bucket (Section
6 above).*

| Layer | Type | σ | Max \|x\| | RF fixed | VD fixed | RF stat | VD stat | **Policy** |
|---|---|---:|---:|---:|---:|---:|---:|:--|
| `blocks.0.attn.qkv` | Attn-QKV | 0.501 | 18.3 | 0.00% | 0.09% | 0.00% | 0.95% | 🟢 INT8 |
| `blocks.0.attn.proj` | Attn-Proj | 0.154 | 8.3 | 0.00% | 0.00% | 0.00% | 1.65% | 🟢 INT8 |
| `blocks.0.mlp.fc1` | MLP-fc1 | 0.866 | 44.1 | 0.00% | 0.15% | 0.00% | 0.54% | 🟢 INT8 |
| `blocks.0.mlp.fc2` | MLP-fc2 | 0.200 | 34.9 | 0.00% | 0.00% | 0.03% | 1.96% | 🟢 INT8 |
| `blocks.1.attn.qkv` | Attn-QKV | 0.579 | 16.3 | 0.00% | 0.17% | 0.00% | 0.43% | 🟢 INT8 |
| `blocks.1.attn.proj` | Attn-Proj | 0.200 | 4.3 | 0.00% | 0.00% | 0.00% | 0.87% | 🟢 INT8 |
| `blocks.1.mlp.fc1` | MLP-fc1 | 0.852 | 33.4 | 0.13% | 0.21% | 0.00% | 0.40% | 🟢 INT8 |
| `blocks.1.mlp.fc2` | MLP-fc2 | 0.124 | 10.2 | 0.00% | 0.00% | 0.00% | 1.89% | 🟢 INT8 |
| `blocks.2.attn.qkv` | Attn-QKV | 0.669 | 11.1 | 0.00% | 0.11% | 0.00% | 0.32% | 🟢 INT8 |
| `blocks.2.attn.proj` | Attn-Proj | 0.204 | 4.1 | 0.00% | 0.00% | 0.00% | 0.60% | 🟢 INT8 |
| `blocks.2.mlp.fc1` | MLP-fc1 | 0.826 | 39.5 | 0.00% | 0.02% | 0.00% | 0.30% | 🟢 INT8 |
| `blocks.2.mlp.fc2` | MLP-fc2 | 0.128 | 7.1 | 0.00% | 0.00% | 0.00% | 1.87% | 🟢 INT8 |
| `blocks.3.attn.qkv` | Attn-QKV | 0.866 | 20.7 | **0.78%** | 0.33% | 0.00% | 0.29% | 🟡 LLM.int8 |
| `blocks.3.attn.proj` | Attn-Proj | 0.207 | 3.1 | 0.00% | 0.00% | 0.00% | 0.75% | 🟢 INT8 |
| `blocks.3.mlp.fc1` | MLP-fc1 | 0.887 | 44.4 | 0.13% | 0.07% | 0.00% | 0.29% | 🟢 INT8 |
| `blocks.3.mlp.fc2` | MLP-fc2 | 0.136 | 8.7 | 0.00% | 0.00% | 0.00% | 1.75% | 🟢 INT8 |
| `blocks.4.attn.qkv` | Attn-QKV | 0.937 | 19.8 | 0.39% | 0.26% | 0.00% | 0.29% | 🟢 INT8 ¹ |
| `blocks.4.attn.proj` | Attn-Proj | 0.240 | 3.8 | 0.00% | 0.00% | 0.00% | 0.80% | 🟢 INT8 |
| `blocks.4.mlp.fc1` | MLP-fc1 | 0.969 | 46.9 | 0.26% | 0.20% | 0.00% | 0.29% | 🟢 INT8 |
| `blocks.4.mlp.fc2` | MLP-fc2 | 0.142 | 11.4 | 0.00% | 0.00% | 0.00% | 1.75% | 🟢 INT8 |
| `blocks.5.attn.qkv` | Attn-QKV | 0.982 | 22.6 | 0.39% | 0.25% | 0.00% | 0.30% | 🟢 INT8 ¹ |
| `blocks.5.attn.proj` | Attn-Proj | 0.264 | 4.5 | 0.00% | 0.00% | 0.00% | 0.68% | 🟢 INT8 |
| `blocks.5.mlp.fc1` | MLP-fc1 | 1.039 | 44.3 | 0.13% | 0.17% | 0.00% | 0.30% | 🟢 INT8 |
| `blocks.5.mlp.fc2` | MLP-fc2 | 0.159 | 11.3 | 0.00% | 0.00% | 0.00% | 1.71% | 🟢 INT8 |
| `blocks.6.attn.qkv` | Attn-QKV | 1.027 | 23.9 | 0.39% | 0.27% | 0.00% | 0.31% | 🟢 INT8 ¹ |
| `blocks.6.attn.proj` | Attn-Proj | 0.259 | 4.2 | 0.00% | 0.00% | 0.00% | 0.64% | 🟢 INT8 |
| `blocks.6.mlp.fc1` | MLP-fc1 | 1.150 | 42.2 | 0.13% | 0.18% | 0.00% | 0.32% | 🟢 INT8 |
| `blocks.6.mlp.fc2` | MLP-fc2 | 0.174 | 10.5 | 0.00% | 0.00% | 0.00% | 1.73% | 🟢 INT8 |
| `blocks.7.attn.qkv` | Attn-QKV | 1.068 | 27.1 | 0.13% | 0.23% | 0.00% | 0.33% | 🟢 INT8 |
| `blocks.7.attn.proj` | Attn-Proj | 0.302 | 5.3 | 0.00% | 0.00% | 0.00% | 0.89% | 🟢 INT8 |
| `blocks.7.mlp.fc1` | MLP-fc1 | 1.333 | 23.3 | 0.26% | 0.37% | 0.00% | 0.33% | 🟢 INT8 |
| `blocks.7.mlp.fc2` | MLP-fc2 | 0.204 | 9.3 | 0.00% | 0.00% | 0.03% | 1.69% | 🟢 INT8 |
| `blocks.8.attn.qkv` | Attn-QKV | 1.122 | 30.0 | 0.13% | 0.18% | 0.00% | 0.35% | 🟢 INT8 |
| `blocks.8.attn.proj` | Attn-Proj | 0.356 | 7.1 | 0.00% | 0.00% | 0.00% | 1.36% | 🟢 INT8 |
| `blocks.8.mlp.fc1` | MLP-fc1 | 3.061 | 62.3 | **3.78%** | 5.72% | 0.00% | 0.35% | 🟡 LLM.int8 |
| `blocks.8.mlp.fc2` | MLP-fc2 | 0.273 | 21.2 | 0.03% | 0.05% | 0.07% | 0.98% | 🟢 INT8 |
| `blocks.9.attn.qkv` | Attn-QKV | 1.257 | 38.7 | 0.13% | 0.20% | 0.00% | 0.37% | 🟢 INT8 |
| `blocks.9.attn.proj` | Attn-Proj | 0.437 | 6.0 | 0.00% | 0.00% | 0.00% | 1.63% | 🟢 INT8 |
| `blocks.9.mlp.fc1` | MLP-fc1 | 6.415 | 104.1 | **97.66%** | 34.39% | 0.00% | 0.37% | 🔴 FP16 |
| `blocks.9.mlp.fc2` | MLP-fc2 | 0.369 | 40.2 | 0.10% | 0.14% | 0.03% | 0.41% | 🟢 INT8 |
| `blocks.10.attn.qkv` | Attn-QKV | 1.423 | 46.1 | 0.39% | 0.37% | 0.00% | 0.38% | 🟢 INT8 ¹ |
| `blocks.10.attn.proj` | Attn-Proj | 0.581 | 7.2 | 0.00% | 0.00% | 0.00% | 1.63% | 🟢 INT8 |
| `blocks.10.mlp.fc1` | MLP-fc1 | 11.676 | 201.0 | **99.74%** | 60.45% | 0.00% | 0.35% | 🔴 FP16 |
| `blocks.10.mlp.fc2` | MLP-fc2 | 0.745 | 92.5 | 0.26% | 0.29% | 0.07% | 0.21% | 🟢 INT8 |
| `blocks.11.attn.qkv` | Attn-QKV | 1.344 | 27.0 | **0.52%** | 0.25% | 0.13% | 0.45% | 🟡 LLM.int8 |
| `blocks.11.attn.proj` | Attn-Proj | 0.682 | 9.3 | 0.13% | 0.04% | 0.00% | 1.97% | 🟢 INT8 |
| `blocks.11.mlp.fc1` | MLP-fc1 | 1.281 | 29.3 | 0.26% | 0.25% | 0.13% | 0.46% | 🟢 INT8 |
| `blocks.11.mlp.fc2` | MLP-fc2 | 0.325 | 15.8 | 0.00% | 0.02% | 0.00% | 1.77% | 🟢 INT8 |
| `head` | Head | 0.973 | 13.8 | 0.00% | 0.01% | 0.00% | 0.38% | 🟢 INT8 |

**Column definitions:**
- **σ**: exact global population std of the matmul input activation (Pass 1, Chan/Welford float64)
- **Max |x|**: largest absolute input value observed across all 5,000 images
- **RF fixed**: per-column routing fraction at |x| > 6.0, ≥ 25% token participation (LLM.int8() cost)
- **VD fixed**: per-value outlier density at |x| > 6.0 (unstructured baseline)
- **RF stat**: per-column routing fraction at |x − μ_c| > 3σ_c, per-channel, ≥ 5% token participation
- **VD stat**: per-value outlier density at |x − μ_c| > 3σ_c, per-channel

¹ `blocks.4/5/6/10.attn.qkv` (all 0.39%) form a borderline cluster just under
the 0.5% LLM.int8() threshold. This bucket boundary is sample-size sensitive
(Section 6); a larger re-run should confirm it before the policy is final.

**Policy summary:** 🔴 FP16: **2 layers** · 🟡 LLM.int8(): **3 layers** · 🟢 INT8: **44 layers**

## 8. Residual-Stream Scale Explosion

The catastrophe in blocks 9-10 is traceable to a transient blow-up in the
residual stream. Tracking the `fc1` input standard deviation (the LayerNorm'd
residual stream) down the network:

```
Block:     0     1     2     3     4     5     6     7  |  8      9      10     11
σ (fc1):  0.87  0.85  0.83  0.89  0.97  1.04  1.15  1.33 | 3.06   6.42  11.68   1.28
```

The transition at block 8 is sharp relative to the gradual rise in blocks 0-7.
The maximum magnitude peaks at **201.0** in block 10 before the residual
stream collapses back to σ ≈ 1.3 at block 11. This is the well-documented
"massive activations" phenomenon in ViT's late-middle blocks: a transient,
localized scale blow-up that is invisible to per-layer relative thresholds
(like 3σ) but catastrophic under any absolute threshold (like the INT8 dynamic
range).

Block 11 fully recovers. The explosion is confined. A surgical FP16 policy for
just two layers contains it.

## 9. Implications for Experiments 2-4

### Experiment 2: Quantization Granularity

Per-tensor INT8 scaling will struggle most in blocks 9-10 fc1 (dynamic range σ
6-12x the model average). Per-token activation scaling should provide partial
relief by adapting to each token's own scale, though the fundamental problem
(too many outlier columns) is structural. The largest accuracy gap between
per-tensor and per-token/per-channel will isolate here.

### Experiment 3: Per-Layer Sensitivity

**Prediction:** quantizing `blocks.9.mlp.fc1` or `blocks.10.mlp.fc1` to naive
INT8 will cause the sharpest accuracy drops by far. If the sensitivity heatmap
agrees with this outlier map, with the two largest bars appearing at blocks 9-10
`fc1`, the cross-validation is complete and the routing policy table above
becomes the thesis's primary deliverable.

### Experiment 4: Decomposition

| Layer | What to expect |
|---|---|
| `blocks.9.mlp.fc1` | Fixed-6.0 threshold: ≈97.7% high-precision fraction. No INT8 benefit. |
| `blocks.10.mlp.fc1` | Fixed-6.0: ≈99.7% HP fraction (effectively pure FP16). |
| `blocks.8.mlp.fc1` | Fixed-6.0: ≈3.8% HP fraction. Meaningful INT8 savings with full accuracy. |
| All `attn.proj` | Fixed-6.0: ≈0% HP fraction. Pure INT8 throughout. |

The 3σ/statistical threshold is not recommended for this experiment; see
Section 5 above and `docs/advisor-touchpoint-guide.md` Section 4.5.

## 10. The Selective Routing Policy: Justification

The following policy is directly readable from the outlier map. It answers the
question the grounding document posed with Experiment 1: which layers are
suited for mixed-precision routing, and which are not?

```
+-----------------------------------------------------------------------------------+
|  LAYER              |  POLICY       |  JUSTIFICATION                            |
+-----------------------------------------------------------------------------------+
|  blocks.9.mlp.fc1   |  FP16         |  97.7% of columns routed. Routing is      |
|  blocks.10.mlp.fc1  |  FP16         |  pure overhead. Run clean FP16.           |
+-----------------------------------------------------------------------------------+
|  blocks.8.mlp.fc1   |  LLM.int8     |  3.8% / 0.78% / 0.52% routing.            |
|  blocks.3.attn.qkv  |  LLM.int8     |  Concentrated outliers (routing <         |
|  blocks.11.attn.qkv |  LLM.int8     |  density), sparse enough for mixed        |
|                      |               |  precision to yield net INT8 benefit.    |
|                      |               |  EDP tradeoff TBD by Exp 3/4.             |
+-----------------------------------------------------------------------------------+
|  All 44 others       |  INT8         |  Routing fraction < 0.5%. Naive INT8      |
|  (incl. all proj,    |               |  is near-lossless. No decomposition       |
|  fc2, head, most     |               |  overhead justified.                      |
|  qkv and fc1)        |               |                                            |
+-----------------------------------------------------------------------------------+
```

The practical consequence: **42 of the model's 48 block-internal matmuls, plus
the `head` layer (44 of 49 total), have a routing fraction low enough to
predict near-lossless INT8.** The EDP of the overall model on the Jetson Orin
Nano is determined by what happens to the remaining 5 layers (2 FP16, 3
LLM.int8()), plus the borderline qkv cluster at 0.39% if a larger run moves it
into LLM.int8(). Whether the LLM.int8() layers achieve a net EDP gain is the
open question Experiments 3-4 will close.

## 11. Statistical Confidence

This run uses 5,000 images (985,000 tokens per layer), reduced from the
50,000-image thesis-print run used in earlier drafts, to re-validate the
pipeline quickly after a code fix. Comparing the two run sizes directly:

| Layer | 50K (historical) | 5K (current) | Stable? |
|---|---:|---:|:--|
| `blocks.9.mlp.fc1` | 97.53% | 97.66% | yes (Δ 0.13 pp) |
| `blocks.10.mlp.fc1` | 99.74% | 99.74% | yes |
| `blocks.8.mlp.fc1` | 3.78% | 3.78% | yes |
| `blocks.3.attn.qkv` | 0.78% | 0.78% | yes |
| `blocks.5.attn.qkv` | 0.52% | 0.39% | **no, bucket flip** |
| `blocks.6.attn.qkv` | 0.52% | 0.39% | **no, bucket flip** |
| `blocks.11.attn.qkv` | 0.39% | 0.52% | **no, bucket flip** |

The two FP16 layers and the clearest LLM.int8() layers are stable across a 10x
change in sample size. Three `attn.qkv` layers near the 0.5% cutoff are not
stable, though each one's underlying routing-fraction estimate only moved by
0.13 percentage points. This means the policy table's qkv bucket assignments
should be treated as provisional pending a full 50,000-image re-run.

---

*Generated from `outputs/exp1_outlier_maps/outlier_stats.json`.*
*Statistical threshold computed per-channel (each of 768 / 3072 feature
dimensions receives its own μ and σ from Pass 1).*
*Run: 5,000 ImageNet-1K val images, two-pass exact global statistics,
ViT-B/16 (timm `vit_base_patch16_224`).*
