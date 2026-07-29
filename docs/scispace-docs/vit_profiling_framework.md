# ViT Quantization & Outlier Profiling: Experimental Framework

> **Citations:** All literature references in this document are catalogued with
> full bibliographic details in [`docs/CITATIONS.md`](CITATIONS.md).

## 1. Project Objective
Profile intermediate activations in an encoder-only Vision Transformer (ViT) to quantify the impact of massive outliers, evaluate outlier ablation, and establish a pathway for integer-only GELU non-linearities. The experiments are scoped pragmatically to prioritize measurable, executable outcomes suitable for edge deployment profiling on NVIDIA Jetson hardware.

## 2. Literature Review: Targeted Reading List
Focus exclusively on literature that supports the defined boundaries. Avoid overly theoretical frameworks that do not directly translate to hardware-friendly quantization.

### Core Outlier & Sparsity Mechanics
*   **The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers (arXiv:2210.06313)**
    *   *Focus:* Understanding the baseline for activation sparsity and how a minority of neurons dictate magnitude.
    *   *Citation:* Li et al. (2023), ICLR 2023. See `docs/CITATIONS.md`.
*   **Massive Activations in Large Language Models (arXiv:2402.17762)**
    *   *Focus:* The structural role of extreme outliers. Extract the methodology for measuring their impact without adopting LLM-scale assumptions for a ViT.
    *   *Citation:* Sun et al. (2024), arXiv preprint. See `docs/CITATIONS.md`.

### Integer Non-linearities & ViT Quantization
*   **I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference**
    *   *Focus:* The `ShiftGELU` mechanism. This is critical for Dr. Yang's requirement to execute GELU on integers, demonstrating how to approximate the floating-point operation using integer bit-shifting or Lookup Tables (LUTs).
    *   *Citation:* Li & Gu (2023), ICCV 2023, arXiv:2207.01405. See `docs/CITATIONS.md`.

### Special Attentions
*   *Search Query Parameters:* "Vision Transformer attention outlier mitigation", "Sparse attention activation distribution".
*   *Focus:* Briefly survey if alternative attention mechanisms (e.g., Swin windowed attention) naturally suppress massive activations compared to standard global Softmax attention. Keep this scoped to vision models.

## 3. Experimental Design

**Target Model:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ViT-B/16, encoder-only, via timm)

### Phase 1: Baseline Activation Profiling

Establish the unquantized ground truth of activation distributions across **six measurement sites** within each encoder block. For each site, register a forward hook and collect statistics over the same validation subset.

#### Measurement Sites

| Site | Hook Target | Tensor Shape | What is Captured |
|------|-------------|--------------|------------------|
| **Pre-Softmax** | `nn.MultiheadAttention` — the raw `QKᵀ/√d` logit tensor before the softmax call | `[B, H, N, N]` | Unbounded attention logits; exposes massive positive outliers that drive Softmax saturation |
| **Post-Softmax** | `nn.MultiheadAttention` — the normalized attention weight matrix immediately after `F.softmax` | `[B, H, N, N]` | Probability mass concentration; reveals how many heads collapse to near-zero entropy ("massive attention" sinks) |
| **Post-LayerNorm** | Each `nn.LayerNorm` in the encoder — sampled on its output | `[B, N, D]` | Channel-wise distribution after normalization; should be approximately unit-variance but persistent per-channel outliers here indicate LN failing to suppress them |
| **Hidden-State Dimensions** | The output of the MLP's first `nn.Linear` (pre-GELU), viewed per output channel | `[B, N, D_mlp]` — analysed channel-wise | Per-channel variance map; identifies which MLP hidden dimensions are structurally outlier-producing independent of the input token |
| **Residual Stream** | The accumulated residual representation entering each encoder block, captured via `block.norm1.input` | `[B, N, D]` | Raw residual stream before any per-block normalization; tracks how the representation grows across layers. Labeled as `blocks.{k}/residual_stream` = residual **after** block k (input to block k+1). See §Document Conventions. |
| **Residual Update Stream** | The addition operation in each encoder block, captured by hooking the final `nn.LayerNorm` input (i.e., the accumulated residual before the closing LN) | `[B, N, D]` | Magnitude of the residual update relative to the skip connection; large residual norms here are the primary cause of quantization range blow-up. **Note:** The residual delta ratio ‖Δ‖/‖x_skip‖ is not yet computed — see `docs/issues.md` T-005. |

#### Per-Site Metrics

For **every** measurement site, compute and record the following statistics over the full collection pass:

*   **Per-tensor scalars:** `mean`, `std` ($\sigma$). `max` and `min` are not yet implemented — see `docs/issues.md` T-010.
*   **Kurtosis (excess):** $\kappa = \mathbb{E}[(x - \mu)^4] / \sigma^4 - 3$ — values $> 0$ confirm heavier tails than a Gaussian; values $\gg 0$ indicate quantization-hostile distributions.  **Computed exactly** via the Pébay (2008) parallel higher-moments merge formula (M3 and M4 tracked across batches).  No approximation caveat is needed.
*   **Outlier fraction:** percentage of elements with $|x| > k\sigma$ (strict inequality) for $k \in \{3, 5, 8\}$.  These are the **primary quantization-sensitivity metrics**.  Computed via a **two-pass recount**: Pass 1 accumulates exact global $\mu$ and $\sigma$ via Pébay merge; Pass 2 (`run_outlier_counting_pass`) counts elements exceeding $k\sigma_{\text{global}}$ using the fixed global thresholds.  This follows standard practice in the quantization literature (Bondarenko et al. 2021; Dettmers et al. 2022).  The 5$\sigma$ and 8$\sigma$ thresholds provide wider coverage than the more common (3, 4, 6); the 4$\sigma$ and 6$\sigma$ fractions can be approximately interpolated from the existing data if needed for literature comparison.
*   **Per-channel $\sigma$ map** (Post-LayerNorm and Hidden-State sites only): a vector of per-channel standard deviations of shape `[D]` or `[D_mlp]`, saved alongside the scalar stats; this is what makes per-channel quantization granularity decisions later
*   **Attention entropy** (Post-Softmax site only): $H = -\sum_j p_j \log p_j$ per head per token, averaged across the batch; near-zero entropy signals a sink token absorbing all probability mass

#### Pre-Softmax vs. Post-Softmax: Why Both Matter

The pre-softmax logit and the post-softmax weight occupy completely different numeric regimes. A single massive pre-softmax value (e.g., $+80$) maps to a post-softmax weight near $1.0$ while all others collapse to $\approx 0$. Measuring **both** lets you:
1. Quantify how wide the pre-softmax dynamic range actually is (the logit range sets the INT8 scale).
2. Confirm whether softmax entropy collapse is already present in the FP32 baseline or only emerges after quantization.

#### Post-LayerNorm: Why It Is Not Automatically Safe

LayerNorm is often assumed to bound activations, but it only normalizes across the hidden dimension for a single token — it does not bound the absolute scale of any individual channel. Channels with persistently large learned affine weights ($\gamma$) will produce systematic per-channel outliers on every token. Log the $\gamma$ weights alongside the post-LN activation std-per-channel to separate learned-scale outliers from distribution outliers. **Note:** $\gamma$/$\beta$ logging is not yet implemented — see `docs/issues.md` T-004.

#### Hidden-State Dimensions: Channel-wise Decomposition

Instead of flattening the MLP pre-GELU tensor, keep the channel dimension intact and compute a `[D_mlp]`-shaped vector of per-channel $\sigma$ values. This directly answers: *which hidden dimensions are quantization-hostile?* Channels with $\sigma_c / \bar{\sigma} > 4$ are flagged as outlier channels and tracked across all 12 blocks to identify structural (layer-agnostic) vs. incidental patterns.

#### Residual Update Stream: Measuring the Delta

The residual stream accumulates contributions from both the attention sub-block and the MLP sub-block. To isolate the update magnitude:
1. Hook the input to the final `nn.LayerNorm` in each block — this tensor is `residual_before_final_ln = skip + mlp_output`.
2. The update delta is `mlp_output = residual_before_final_ln - residual_after_attn_ln`. The magnitude of this delta relative to the skip norm, $\|\Delta\| / \|x_{\text{skip}}\|$, indicates how aggressively the MLP modifies the stream. Large ratios directly cause quantization range expansion in the residual.

**Note:** The residual delta ratio is not yet computed — see `docs/issues.md` T-005.

#### Data Collection

*   **Mechanism — `profiler.py` Welford multi-batch pipeline (Option C):**
    *   `profiler.py` wraps the model with `nnsight.NNsight` and captures all **six sites** per block inside each forward pass via `profile_vit`.
    *   A `WelfordAccumulator` per site aggregates per-batch statistics across the full dataset using the Pébay (2008) parallel higher-moments merge formula for M2, M3, and M4.  This gives exact global mean, std, and kurtosis for all six sites.
    *   Outlier fractions are computed in a **separate second pass** (`run_outlier_counting_pass`) using the exact global $\mu$ and $\sigma$ from the completed Welford accumulation as fixed thresholds.  This two-pass approach follows standard practice (Bondarenko et al. 2021; Dettmers et al. 2022) and avoids the per-batch-$\sigma$ overestimate that a single-pass approach would produce.
    *   `hooks.py` is **not** used in Phase 1.  It is retained for reference.
*   **Dataset:** Same validation subset used across all phases for comparability.
*   **Storage:** Reduce each batch to scalar statistics inside `profile_vit` (no raw tensors stored); merge into accumulators via `merge_batch_stats`; save the complete `ProfilingResult` to `profiling_result.json` via `profiler.save_profiling_result`.

#### Deliverables

*   Log-scale histograms per site showing heavy-tailed distributions; annotate $\pm 3\sigma$, $\pm 5\sigma$, and $\pm 8\sigma$ boundaries.
*   Per-channel $\sigma$ heatmaps (layers × channels) for Post-LayerNorm and Hidden-State sites.
*   Per-head attention entropy heatmap (layers × heads) for the Post-Softmax site.
*   A single summary table of kurtosis and outlier-fraction values across all sites and layers.

### Phase 2: Outlier Ablation (Zeroing)

Test the structural reliance on massive activations by forcing them to zero. Ablation is applied **site-selectively** to understand which locations are load-bearing.

*   **Mechanism:** Intercept activations at the targeted site and apply a hard mask: $x = 0$ if $|x| > \tau$.
*   **Sites ablated (in separate sweeps):**
    *   Pre-GELU hidden-state activations (original Phase 2 target).
    *   Pre-Softmax attention logits — ablating large logits tests whether attention sink tokens are load-bearing.
    *   Residual update stream — ablating large residual deltas tests whether specific MLP blocks dominate the stream.
*   **Thresholding:** Define $\tau$ dynamically using a scalar multiple of the layer's standard deviation: $\tau = k\sigma$ for $k \in \{3, 5, 8\}$ (matching the Phase 1 `OUTLIER_SIGMAS`).  The source of $\sigma$ for all sites is the exact global std from Phase 1 (`profiler.py` Welford multi-batch pass, `profiling_result.json`).  All six sites — including `pre_softmax` — are now covered dataset-wide, so no single-batch sigma estimation is required in Phase 2.
*   **Metrics:**
    *   Percentage of zeroed elements per layer and per site.
    *   Top-1 validation accuracy at each threshold step.
    *   Change in post-softmax attention entropy (for the pre-softmax ablation sweep) — a drop in entropy means outlier removal disrupts attention routing.
*   **Deliverable:** A degradation curve per site showing model accuracy vs. outlier ablation threshold, and a per-layer breakdown of what fraction of elements were zeroed at the accuracy break-point.

### Phase 3: Integer GELU Exploration
Reconcile the bounded distribution (post-ablation) into an integer space.
*   **Mechanism:** Utilize the profiling metrics from Phase 1 & 2 to define a symmetric or asymmetric dynamic range for INT8 or INT4 quantization.
*   **Implementation:** Construct a piece-wise linear approximation or a Lookup Table (LUT) that accepts the integer bounds and performs the GELU transformation without dequantizing to FP32/FP16.
*   **Metrics:** Compare the output distribution of the Integer GELU against the FP32 GELU outputs. 

## 4. Document Conventions

### Site Labeling Convention

The `residual_stream` site label `blocks.{k}/residual_stream` denotes the residual
stream **after** encoder block `k` has processed it — i.e., the input to block `k+1`.

| Label | Meaning |
|-------|---------|
| `patch_embed/residual_stream` | Patch embedding + positional encoding + CLS token (input to block 0) |
| `blocks.0/residual_stream` | Output of block 0 (input to block 1) |
| `blocks.5/residual_stream` | Output of block 5 (input to block 6) |
| `blocks.10/residual_stream` | Output of block 10 (input to block 11) |
| `blocks.11/residual_stream` | Output of block 11 — final encoder output, before head LayerNorm. **Not yet captured** — see `docs/issues.md` T-001. |

All other site labels `blocks.{k}/{site}` denote measurements **inside** block `k`
(e.g., `blocks.5/pre_gelu` is the pre-GELU activation inside block 5).

### Sigma Threshold Convention

Outlier fractions are reported at $k \in \{3, 5, 8\}$ standard deviations. This
provides wider coverage than the more common $(3, 4, 6)$:
- $3\sigma$: moderate outliers, Gaussian tail baseline ($\approx 0.27\%$)
- $5\sigma$: significant outliers ($\approx 5.7 \times 10^{-5}$ under Gaussian)
- $8\sigma$: extreme outliers (essentially impossible under Gaussian)

The $4\sigma$ and $6\sigma$ fractions can be approximately interpolated from the
existing data if needed for literature comparison.

## 5. Hardware Profiling Considerations
*   Ensure the custom profiling hooks and masking operations do not introduce artifact overhead that skews subsequent energy-delay product (EDP) measurements when transitioning the optimized model to the target hardware constraints.
