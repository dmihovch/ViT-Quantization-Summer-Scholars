# Phase 2b: Per-Channel Ablation — Implementation Plan

> **Created:** 2026-08-01
> **Status:** ✅ Implemented 2026-08-02 — see `docs/phase2-expansion.md` for results
> **Depends on:** Phase 1 profiling complete (✅), Phase 2 population ablation complete (✅)
>
> **Note:** The per-channel ablation described in this plan was implemented as part
> of the Phase 2 expansion.  This document is retained as a historical planning
> artifact.  The implementation differs from the plan in two key respects:
> 1. Per-channel ablation was applied to `pre_gelu` (not the planned `post_layernorm_2`),
>    because existing Phase 1 data provided per-channel stats for pre_gelu.
> 2. The `zeroing_mode` field was implemented as `granularity` + `ablation_mode`
>    on `AblationConfig` rather than as a field on `AblationResult`.

---

## 0. Research Hypothesis

> **H1:** Accuracy degradation from activation zeroing is driven by outliers in
> specific high-variance *channels*, not uniformly across all channels. A
> per-channel threshold (`|x_c − μ_c| > k·σ_c`) will preserve more accuracy
> than the global-σ threshold (`|x − μ| > k·σ_global`) while zeroing a
> comparable fraction of elements.

This directly tests whether per-channel quantization (the standard approach in
SmoothQuant, Xiao et al. 2023) is justified for ViT activations: if per-channel
zeroing preserves accuracy better than global zeroing at matched sparsity, then
per-channel quantization ranges are more efficient than per-tensor ranges.

---

## 1. Data Inventory — What We Have

### 1.1 Available per-channel statistics (from `profiling_result.json`)

| Site type | Per-channel std | Per-channel mean | LayerNorm γ |
|-----------|:---:|:---:|:---:|
| `post_layernorm_1` | ✅ | ✅ (recoverable from sum/n) | ✅ |
| `post_layernorm_2` | ✅ | ✅ (recoverable from sum/n) | ✅ |
| `pre_gelu` | ✅ | ✅ (recoverable from sum/n) | — |
| `residual_stream` | ❌ | ❌ | — |
| `pre_softmax` | ❌ | ❌ | — |
| `post_softmax` | ❌ | ❌ | — |

Per-channel mean is recovered as `per_channel_sum[c] / per_channel_n` where
`per_channel_n = n_samples / D`. The sum and sum_sq fields exist in the JSON.

### 1.2 LayerNorm γ/β availability

All 24 `post_layernorm_1` and `post_layernorm_2` sites (12 blocks × 2) have
`layernorm_gamma` and `layernorm_beta`. These are the α parameter in
SmoothQuant's `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)` formula.

### 1.3 Gap: residual_stream per-channel stats

Residual stream sites do not have per-channel statistics. This is a significant
gap because Phase 2 showed residual_stream is the most sensitive site (5.6%
accuracy at k=3σ). However, the residual stream is the sum of attention and MLP
outputs — its per-channel structure is less semantically meaningful as a
quantization target than LayerNorm outputs (which are normalized per-token and
have a clean channel axis).

**Decision:** Focus the per-channel ablation on `post_layernorm_1` and
`post_layernorm_2` (pre-attention and pre-MLP LayerNorm outputs). These are the
standard sites for per-channel quantization and the standard intervention point
for SmoothQuant-style smoothing.

---

## 2. Experiment Design

### 2.1 Experiment 2b: Per-channel outlier ablation

#### Independent variable
Zeroing mode: {global-σ, per-channel-σ} × site × k

#### Sites tested
- `post_layernorm_2` (pre-MLP LayerNorm output) — **primary site**
- `post_layernorm_1` (pre-attention LayerNorm output) — secondary site
- `pre_gelu` (already has per-channel data, useful as a control)

#### Sigma thresholds
Same as Phase 2: `{3.0, 4.0, 6.0}` for direct comparability.

#### Metrics
Same as Phase 2: top-1 accuracy, top-5 accuracy, per-layer %-zeroed.
No entropy (not applicable to these sites).

#### Random control
Same as Phase 2: per-batch matched random zeroing using the exact per-layer
%-zeroed from the corresponding outlier pass.

#### Comparison baseline
The existing Phase 2 global-σ results for `pre_gelu` (the closest comparable
site — `post_layernorm_2` feeds into the MLP that produces pre-GELU activations).

#### Expected result
Per-channel zeroing should preserve MORE accuracy than global zeroing at
matched sparsity levels, because:
- Channels with σ_c << σ_global will not have their values zeroed unnecessarily
- Channels with σ_c >> σ_global will have a higher (more permissive) threshold
- Only truly anomalous values within each channel's own distribution get zeroed

### 2.2 Experiment 2c (optional): Per-channel quantization simulation

If Phase 2b results are compelling, add a true quantization step that rounds
post-LayerNorm activations to INT8 values using per-channel scales derived
from `per_channel_std` and `layernorm_gamma`. This bridges the gap between
ablation (zeroing) and actual quantization (rounding to discrete levels).

---

## 3. Code Changes Required

### Step 1: Add `post_layernorm_1` and `post_layernorm_2` as ablation sites

**File:** `src/ablation.py`

1. Add `_intervene_post_layernorm(block, block_idx, sigma_k, layer_stats, pct_zeroed, mode, ...)`
   - Intervention point: `block.norm2.output` for LN2, `block.norm1.output` for LN1
   - Site ID: `f"blocks.{block_idx}/post_layernorm_2"` etc.
   - Shape: `(B, N, 768)`

2. Add to `zero_outliers_in_trace`: support `"post_layernorm_1"` and `"post_layernorm_2"` sites.

3. Add `_build_per_channel_zeroing_mask(tensor, sigma_k, per_channel_std, per_channel_mean)`:
   ```python
   def _build_per_channel_zeroing_mask(
       tensor: torch.Tensor,        # (B, N, D)
       sigma_k: float,
       per_channel_std: torch.Tensor,   # (D,)
       per_channel_mean: torch.Tensor,  # (D,)
   ) -> torch.Tensor:
       threshold = sigma_k * per_channel_std  # (D,)
       deviation = (tensor - per_channel_mean).abs()  # (B, N, D)
       return deviation <= threshold  # broadcasts (D,) over (B, N)
   ```
   The threshold broadcasts over the batch and token dimensions. Each channel `c`
   gets its own threshold `k * σ_c`.

4. Extend `AblationResult` with `zeroing_mode: str` field:
   - `"global"` — current population σ threshold (Phase 2)
   - `"per_channel"` — per-channel σ threshold (Phase 2b)
   This avoids creating a separate dataclass.

5. Update `save_ablation_results` to include `zeroing_mode` column.

### Step 2: Load per-channel stats from `profiling_result.json`

**File:** `src/exp2_ablation.py`

The `load_profiling_result` function already deserializes `per_channel_std`,
`per_channel_sum`, `per_channel_sum_sq`, and `n_samples` into `LayerStats`.
No changes needed for loading.

Need to add a helper:
```python
def _get_per_channel_mean(stats: LayerStats) -> list[float]:
    """Recover per-channel mean from per_channel_sum and n_samples."""
    if stats.per_channel_sum is None or stats.n_samples == 0:
        return []
    D = len(stats.per_channel_sum)
    per_ch_n = stats.n_samples // D
    return [s / per_ch_n for s in stats.per_channel_sum]
```

### Step 3: Extend the orchestrator

**File:** `src/exp2_ablation.py`

Add to `run()`:
```python
sites = ("pre_gelu", "pre_softmax", "residual_stream",
         "post_layernorm_2", "post_layernorm_1")
```

For `post_layernorm_1` and `post_layernorm_2`, run both:
- Global-σ zeroing (using `layer_stats[site_id].std` and `.mean`)
- Per-channel zeroing (using `_get_per_channel_mean` and `.per_channel_std`)
- Random control for each (per-batch matched)

This produces a 2×2 comparison: global vs per-channel × outlier vs random.

### Step 4: Tests

**File:** `tests/test_ablation.py`

1. Unit tests for `_build_per_channel_zeroing_mask`:
   - All-kept (low sigma_k, high std): `assert mask.all()`
   - All-zeroed (high sigma_k, low std): `assert not mask.any()`
   - Mixed: different channels, different thresholds
   - Shape preservation: `assert mask.shape == tensor.shape`
   - Broadcasting: verify per-channel thresholds apply correctly

2. Slow tests for `zero_outliers_in_trace` with `post_layernorm_2`:
   - Logits change at low sigma_k
   - Returns pct_zeroed
   - Logits shape correct
   - Per-channel mode works

3. Integration test: load real `profiling_result.json`, verify per-channel
   stats can be recovered for post_layernorm_1/2 sites.

### Step 5: Plotting

**File:** `src/plotting.py`

Add `plot_accuracy_vs_threshold_comparison` — overlays global vs per-channel
accuracy curves for the same site on one plot with distinct line styles.
The Phase 2 accuracy-vs-threshold plots already exist, so this is additive.

### Step 6: Documentation

- `docs/EXP2b-IMPL.md` — new implementation document
- `docs/issues.md` — new tickets for per-channel ablation
- `docs/NEXT-STEPS.md` — update roadmap

---

## 4. Quantization Simulation (Phase 3 Bridge)

If Phase 2b confirms per-channel thresholds matter, the natural next step is
actual quantization rather than zeroing. This forms the bridge to Phase 3
(integer GELU).

### 4.1 Fake-quantization of LayerNorm outputs

At `post_layernorm_2`, apply symmetric per-channel INT8 quantization:

```
scale_c = max(|x_c|) / 127     or     scale_c = σ_c * factor
x_quant_c = round(clip(x_c / scale_c, -128, 127))
x_dequant_c = x_quant_c * scale_c
```

Then pass the dequantized values to fc1 (the first MLP linear layer). Measure
accuracy degradation as a function of the scale factor.

### 4.2 SmoothQuant-style smoothing

Before quantizing, apply per-channel smoothing:

```
s_c = (max(|x_c|) ^ α) / (max(|W[:,c]|) ^ (1-α))
x_smoothed_c = x_c / s_c
W_smoothed[:,c] = W[:,c] * s_c
```

Then quantize `x_smoothed`. The MLP output is mathematically identical before
quantization (scale factor cancels out). After quantization, the activation
range is narrower, so quantization error is smaller.

This requires reading the fc1 weight matrix from the model, computing per-output-channel
max magnitudes, and the smoothing factor α (hyperparameter, sweep 0.3–0.7).

### 4.3 Implementation notes

The SmoothQuant smoothing is applied **outside the forward pass** — it's a
pre-processing step that modifies the weight matrix and applies a per-channel
scale to activations inline. The nnsight trace can apply the activation scaling
at `block.norm2.output` and we replace `block.mlp.fc1.weight` before the run.

---

## 5. Dependency Graph

```
Phase 1 profiling (✅)
    │
    ▼
Phase 2 population ablation (✅)
    │
    ├──► Phase 2b per-channel ablation (🔲 this plan)
    │       │
    │       ▼
    │    Phase 2c quantization simulation (🔲 optional)
    │
    └──► Phase 3 integer GELU (🔲)
```

Phase 2b does NOT depend on Phase 3. It can run immediately with existing
Phase 1 data. Phase 2c (quantization simulation) naturally follows Phase 2b
and feeds into Phase 3.

---

## 6. Compute Budget Estimate

Phase 2 took ~70 minutes for 5 sweeps (3 sites with random control for 2).
Phase 2b would add:

- `post_layernorm_2`: 2 modes (global + per-channel) × 3 thresholds × 2 passes (outlier + random) = 12 passes
- `post_layernorm_1`: same = 12 passes
- Total: 24 passes

At ~4 minutes per pass (pre_gelu was ~4 min/pass at 50k images), that's ~96 minutes.
Total Phase 2b runtime: ~1.5–2 hours on RTX 3070.

Optionally, skip `post_layernorm_1` initially (run `post_layernorm_2` only) for
a ~50-minute run to validate the approach before committing to both sites.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Per-channel zeroing may zero very few or very many elements on some channels, making the comparison unfair | Report per-channel %-zeroed distributions alongside accuracy |
| `post_layernorm_2` has zero-mean per-token (LayerNorm property) — global and per-channel thresholds may produce similar masks | Check this empirically first; the Phase 1 data shows per-channel σ varies 100×+, so thresholds will differ |
| nnsight proxy shape operations for per-channel broadcasting may fail | Test with a simple `torch.randn(B, N, D)` mock before running full experiment |
| Per-channel mean recovery from sum/n may have floating-point precision issues at large n | n ≈ 50k × 197 = ~10M per channel; sum may lose precision. Consider storing per-channel mean directly in Phase 1 (requires re-running profiling) |

---

## 8. Prioritized Action Items

1. **Add `_build_per_channel_zeroing_mask` to `src/ablation.py`** — pure function, testable without nnsight
2. **Add per-channel tests** — verify broadcasting logic
3. **Add `post_layernorm_2` as ablation site** — reuse existing `_intervene_*` pattern
4. **Extend orchestrator** — add per-channel mode to sweep
5. **Run `post_layernorm_2` only** — validate approach before committing to both sites
6. **Add comparison plots** — global vs per-channel accuracy curves
7. **If results are compelling:**
   a. Add `post_layernorm_1`
   b. Implement quantization simulation (Phase 2c)
   c. Feed per-channel scale factors into Phase 3 integer GELU design
