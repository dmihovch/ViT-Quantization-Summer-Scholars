# EXP2-IMPL: Experiment 2 — Outlier Ablation (Zeroing)

> **Status:** Phase 2 implementation complete (2026-08-02).
> Mean-centered thresholding (T-020), random-zeroing control (T-021),
> class-imbalance fix (T-022), per-channel ablation (T-024).
> See `docs/issues.md` for details.
> Tested with PyTorch 2.12.1, nnsight 0.7.0, CUDA 13.0, NVIDIA RTX 3070 (8 GB).

---

## 0. What "done" looks like

```bash
# Full ablation sweep across all three sites with default sigma thresholds.
python run_phase2_ablation.py --data-dir data --num-images 50000 \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json

# Fast subset for smoke-testing (now class-balanced via seeded permutation).
python run_phase2_ablation.py --data-dir data --num-images 1024 \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --sigma-thresholds 3.0 6.0
```

produces:

```
outputs/phase2-ablation/
├── ablation_results.csv              # (3 sites × 3 thresholds × N layers) rows
│                                       # + random-control rows (is_random=True)
│                                       # Includes entropy fields as JSON strings
├── entropy_deltas.csv                # Per-layer CLS/patch entropy deltas (pre_softmax only)
├── accuracy_vs_threshold_pre_gelu.png
├── accuracy_vs_threshold_pre_gelu_random.png   # Random control comparison
├── accuracy_vs_threshold_pre_softmax.png
├── accuracy_vs_threshold_residual_stream.png
├── accuracy_vs_threshold_residual_stream_random.png  # Random control comparison
├── pct_zeroed_pre_gelu_k3.0.png
├── ...                               # one pct_zeroed PNG per (site, k)
└── pct_zeroed_residual_stream_k6.0.png
```

---

## 1. Files and their status

| Step | File | Status |
|------|------|--------|
| 7 | `src/ablation.py` | ✅ Done — nnsight-based intervention, mean-centered thresholding, random-zeroing control |
| 7 | `tests/test_ablation.py` | ✅ Done — 30 fast + 13 slow tests |
| 8 | `src/exp2_ablation.py` | ✅ Done — orchestrator with outlier sweep + random-control sweep |
| 8 | `run_phase2_ablation.py` | ✅ Done — CLI with sigma_thresholds matching Phase 1 |
| — | `src/plotting.py` | ✅ Done — `plot_accuracy_vs_threshold`, `plot_pct_zeroed_per_layer` |
| — | `src/data_loader.py` | ✅ Done — class-balanced subset sampling (T-022) |

---

## 2. Architecture: nnsight-based intervention

Phase 2 uses the same nnsight trace mechanism as Phase 1, but with **tensor replacement** instead of observation. The core function is `zero_outliers_in_trace`, which:

1. Opens an nnsight trace context on the input batch
2. For each encoder block, replaces the activation tensor at the specified site with a zeroed version
3. For pre_softmax, also captures per-head CLS and patch entropy on the zeroed attention weights
4. Saves the output logits via `.save()`
5. Returns logits + per-layer %-zeroed + per-layer entropy data

### 2.1 Intervention sites

| Site | Intervention point | Shape | σ/μ source |
|------|-------------------|-------|----------|
| `pre_gelu` | `block.mlp.act.input` | `(B, N, 3072)` | `blocks.{i}/pre_gelu` |
| `pre_softmax` | Reconstructed QKᵀ/√d → `attn.proj.input` | `(B, H, N, N)` | `blocks.{i}/pre_softmax` |
| `residual_stream` | `block.norm1.input` (CLS preserved) | `(B, N, 768)` | `blocks.{i-1}/residual_stream` |

### 2.2 Threshold definition (mean-centered)

The zeroing threshold uses the **mean-centered** definition consistent with Phase 1:

```
|x − μ| > k·σ
```

where μ and σ are the per-layer population mean and standard deviation from
Phase 1's `profiling_result.json`.  This is the standard statistical definition
used in the quantization literature (Wei et al. 2022, §3.1; Bondarenko et al.
2021, §4.1).

Previously (before T-020 fix), the threshold used a zero-centered definition
(`|x| > k·σ`), which was inconsistent with Phase 1's outlier definition.

### 2.3 Random-zeroing control

For `pre_gelu` and `residual_stream` sites, a random-zeroing control condition
is run alongside the outlier-threshold sweep.  After each outlier forward pass,
the per-layer %-zeroed is collected and used as the target fraction for a
random pass on the **same batch**.  This ensures the random control zeros
exactly the same fraction of elements as the outlier condition on every batch,
eliminating the confound of differing zeroing rates.

The only difference between conditions is *which* elements are zeroed (outliers
vs. random), not *how many*.  This is the strongest possible control for
attributing accuracy degradation to outliers specifically.

Random-control results are marked with `is_random=True` in `ablation_results.csv`
and plotted separately (`accuracy_vs_threshold_{site}_random.png`).

`pre_softmax` is excluded from the random control because random zeroing of
attention logits would require reconstructing QKᵀ/√d inside the random path,
which is architecturally different from the outlier path.

### 2.4 pre_softmax reconstruction

The pre-softmax intervention reconstructs QKᵀ/√d from `attn.qkv.output` using the exact same computation as Phase 1's `_register_pre_softmax_saves` (profiler.py L1226-1237):

```
qkv → reshape(B, N, 3, H, D) → permute(3, B, H, N, D)
q_scaled = qkv[0] * scale
k = qkv[1]
v = qkv[2]
logits = q_scaled @ kᵀ
```

After zeroing outliers in the logit matrix, attention weights are recomputed via softmax. Per-head entropy is captured using `torch.special.entr` (consistent with Phase 1 after T-017 fix). The attention output is then injected via `block.attn.proj.input`.

### 2.5 Entropy delta computation

For `pre_softmax` ablation, per-head CLS and patch attention entropy is captured on the zeroed attention weights. These are compared against Phase 1 baseline entropy values from `profiling_result.json` using `compute_entropy_delta`, which returns mean per-head delta in nats. Results are saved to `entropy_deltas.csv`.

### 2.6 CLS token preservation

For `residual_stream` zeroing, the CLS token (position 0 along the token dimension) is explicitly preserved. Zeroing the CLS token would destroy the classification signal regardless of outlier status. Ref: Wei et al. (2022), arXiv:2209.13325, §3.1.

### 2.7 Per-channel ablation (granularity)

Per-channel ablation uses per-channel μ_c and σ_c from Phase 1's
``per_channel_mean`` and ``per_channel_std`` fields instead of the global
scalar μ and σ.  The threshold for channel ``c`` is:

```
|x_c − μ_c| > k·σ_c
```

This is only meaningful for ``pre_gelu``, where the MLP hidden dimension
(3,072) has a natural channel structure.  Per-channel mode:
- Only ablates ``pre_gelu`` (skips ``pre_softmax`` and ``residual_stream``).
- Does not run the random-zeroing control (per-channel random zeroing would
  require per-channel random fractions, which is over-parameterized).
- Writes ``granularity=per_channel`` in the CSV output.

Enabled via ``--granularity per_channel`` on the CLI or
``AblationConfig(granularity="per_channel")``.

**Hypothesis:** High-variance channels carry proportionally more
outlier-dependent signal.  A global threshold over-zeroes them while
under-zeroing low-variance channels.  Per-channel thresholds redistribute
the zeroing budget, preserving more accuracy at aggressive thresholds.

**Result (50k images):** Confirmed.  At k=3, per-channel preserves 47.00%
vs 43.24% global (+3.76%).  At k≥4 the difference vanishes.

---

## 3. `src/ablation.py` — Public API

### 3.1 `AblationResult`

```python
@dataclass
class AblationResult:
    site: str              # "pre_gelu", "pre_softmax", "residual_stream"
    sigma_threshold: float # k value
    site_identifier: str   # e.g. "blocks.3/pre_gelu"
    pct_zeroed: float      # [0, 100]
    top1_accuracy: float
    top5_accuracy: float
    baseline_top1: float   # unablated accuracy
    baseline_top5: float
    is_random: bool        # True for random-zeroing control condition
    granularity: str       # "global" or "per_channel"
    cls_entropy: list[float]           # per-head CLS entropy after ablation [H]
    patch_entropy: list[float]         # per-head patch entropy after ablation [H]
    baseline_cls_entropy: list[float]  # Phase 1 baseline [H]
    baseline_patch_entropy: list[float] # Phase 1 baseline [H]
```

### 3.2 `compute_pct_zeroed` — REMOVED

This function was dead code (never called in the codebase; pct_zeroed is computed
inline in the intervention functions via ``(~mask).float().mean().item()``).
Deleted 2026-08-03.

### 3.3 `_build_zeroing_mask(tensor, sigma_k, sigma, mean) -> Tensor`

Builds a boolean keep-mask where `|x − μ| ≤ k·σ`.  The `mean` parameter is
required and must always be provided from `layer_stats`.

### 3.4 `_build_per_channel_zeroing_mask(tensor, sigma_k, per_channel_sigma, per_channel_mean, device) -> Tensor`

Builds a boolean keep-mask where `|x_c − μ_c| ≤ k·σ_c` per channel.
Broadcasts per-channel statistics of shape `(D,)` against tensor of shape
`(B, N, D)`.  Only valid for pre_gelu (MLP hidden dimension).

### 3.5 `_build_random_mask(tensor, fraction, seed=None, salt=0) -> Tensor`

Builds a boolean keep-mask where exactly `fraction` of elements are zeroed
at uniformly random positions.  Used for the random-zeroing control condition.

### 3.6 `compute_entropy_delta(ablated_cls, ablated_patch, baseline_cls, baseline_patch) -> dict`

Returns `{"mean_cls_delta": float, "mean_patch_delta": float}`. Positive means entropy increased (more uniform attention) after zeroing.

### 3.7 `zero_outliers_in_trace(wrapped_model, input_batch, site, sigma_k, layer_stats, random_fractions=None, random_seed=None, per_channel=False) -> (logits, pct_zeroed, entropy_data)`

Core intervention function. Runs one forward pass with outlier zeroing (or
random zeroing if `random_fractions` is provided).  If `per_channel=True` and
`site="pre_gelu"`, uses per-channel μ_c and σ_c for thresholding.  Returns
3-tuple with logits, per-layer %-zeroed, and per-layer entropy (for pre_softmax).

### 3.8 `save_ablation_results(results, path) -> None`

Persists results to CSV. Includes `is_random` and `granularity` columns.
Entropy fields serialised as JSON strings.

### 3.9 `save_entropy_deltas(results, path) -> None`

Persists per-layer entropy deltas for pre_softmax ablation to a separate CSV.

---

## 4. `src/exp2_ablation.py` — `run(config)`

Pipeline:
1. Load ViT-B/16, wrap with NNsight
2. Load Phase 1 stats from `profiling_result.json` (includes baseline entropy and outlier fractions)
3. Build val loader (shuffle=False, class-balanced subset via seeded permutation)
4. Measure baseline accuracy (no intervention)
5. For each site × k:
   a. Outlier-threshold sweep: evaluate accuracy with mean-centered zeroing
   b. Random-control sweep (pre_gelu, residual_stream): evaluate accuracy with
      random zeroing at the **exact same per-batch per-layer fraction** as the
      outlier condition
6. Save `ablation_results.csv` + `entropy_deltas.csv` + generate plots (outlier + random)

---

## 5. Test coverage

### Fast tests (29)

### Slow tests (13, marked `@pytest.mark.slow`)
- 4 pre_gelu tests (logits-change, returns-pct, shape, random-mode)
- 3 residual_stream tests (logits-change, no-zeroing-at-high-k, random-mode)
- 3 pre_softmax tests (logits-change, returns-pct, returns-entropy)
- 3 edge case tests (invalid-site-raises, missing-stats, zero-sigma)

---

## 6. Design decisions

### 6.1 nnsight over raw PyTorch hooks

**Decision:** Use nnsight trace intervention, not `register_forward_pre_hook`.

**Rationale:** nnsight gives us the same granular access to intermediate activations that Phase 1 profiling uses. Raw PyTorch hooks cannot intercept the attention logit matrix (computed inline in `Attention.forward()`) or the residual stream at the correct point (before `norm1`).

### 6.2 CLS token preservation in residual_stream

**Decision:** Preserve the CLS token (position 0) when zeroing the residual stream.

**Rationale:** The CLS token is the classification signal. Zeroing it would destroy accuracy regardless of outlier status, confounding the ablation. Wei et al. (2022) preserve special tokens in their outlier suppression experiments.

### 6.3 pre_softmax reconstruction matches Phase 1

**Decision:** Use the exact same QKᵀ/√d reconstruction as `_register_pre_softmax_saves`.

**Rationale:** Consistency with Phase 1 profiling. The σ and μ values used for thresholding come from Phase 1's profiling of the same reconstructed tensor.

### 6.4 Baseline accuracy measured once

**Decision:** Measure baseline accuracy once before any ablation, not per-site.

**Rationale:** The baseline is the model's accuracy without any intervention. It doesn't depend on which site is being ablated.

### 6.5 Entropy deltas as separate CSV

**Decision:** Save entropy deltas to `entropy_deltas.csv` alongside `ablation_results.csv`.

**Rationale:** Entropy deltas are only meaningful for pre_softmax ablation. Keeping them in a separate file avoids cluttering the main results CSV with NaN/empty columns for pre_gelu and residual_stream sites. The full per-head entropy values are still embedded in `ablation_results.csv` as JSON strings for reproducibility.

### 6.6 Mean-centered thresholding (T-020)

**Decision:** Use `|x − μ| > k·σ` for zeroing, consistent with Phase 1's outlier definition.

**Rationale:** Phase 1 defines outliers as mean-centered. Phase 2 must use the same definition to ensure the experiment zeros the elements Phase 1 identified as outliers. For pre_gelu sites with large mean shifts (μ ≈ −28), the zero-centered definition would zero a different set of elements.

### 6.7 Random-zeroing control (T-021)

**Decision:** Run a random-zeroing sweep alongside the outlier-threshold sweep
for pre_gelu and residual_stream, using the **exact per-batch per-layer %-zeroed**
from the outlier pass as the target fraction for the random pass.

**Rationale:** Without a random-zeroing control, it is impossible to attribute
accuracy degradation to outliers specifically rather than to activation sparsity
in general.  Matching the zeroing rate on a per-batch basis eliminates the
confound of differing zeroing rates between conditions — the only difference is
*which* elements are zeroed, not *how many*.

### 6.8 Class-balanced subset sampling (T-022)

**Decision:** Use seeded random permutation for subset selection when `shuffle=False`.

**Rationale:** ImageFolder returns images grouped by class in alphabetical order. Taking the first N images would sample only from the first few classes, producing meaningless accuracy numbers for small `--num-images` values.

### 6.9 Per-channel ablation (T-024)

**Decision:** Add per-channel zeroing as a separate granularity mode for
pre_gelu, using per-channel μ_c and σ_c from Phase 1.

**Rationale:** Phase 1 profiling showed per-channel σ varies 12× within
block 10 (2.06–25.54).  A global threshold over-zeroes high-variance
channels while under-zeroing low-variance channels.  Per-channel ablation
tests whether this concentration drives accuracy degradation — and whether
per-channel quantization of the MLP hidden dimension would reduce the
accuracy penalty of INT8 range clipping.  The 3.76% improvement at k=3
confirms this hypothesis.

### 6.10 Per-channel mean added to Phase 1 (T-023)

**Decision:** Add ``per_channel_mean`` to ``LayerStats`` alongside existing
``per_channel_std``.

**Rationale:** Per-channel standard deviation without per-channel mean is
only half the picture.  Any mean-centered threshold (``|x_c − μ_c| > k·σ_c``)
requires both.  The data existed in memory during profiling but was discarded.
Phase 1 was re-run to regenerate the JSON with this field.

---

## 7. Critical constraints

- `fused_attn=False` must be set before wrapping with NNsight (done by `load_vit`).
- Model must be in `eval()` mode (done by `load_vit`, asserted in `profile_vit`).
- Phase 1 `profiling_result.json` must exist at `config.layer_stats_path` and contain `mean`, `std`, `outlier_fractions`, and `attention_entropy_cls`/`attention_entropy_patches`.
- For per-channel ablation, `profiling_result.json` must also contain `per_channel_mean` and `per_channel_std` for all pre_gelu sites (requires Phase 1 re-run after 2026-08-02).
- The `pre_softmax` intervention reconstructs QKᵀ/√d and injects via `attn.proj.input`. This bypasses the attention dropout (`attn_drop`) — acceptable since dropout is disabled in eval mode.
- Random-zeroing control uses per-batch per-layer %-zeroed from the outlier pass
  as the target fraction — no dependency on Phase 1 `outlier_fractions`.
- Per-channel mode only ablates `pre_gelu`; `pre_softmax` and `residual_stream` are skipped.

---

## 8. Test checklist

- [ ] `pytest -m "not slow" tests/test_ablation.py` — 29 fast tests
- [ ] `pytest -m "slow" tests/test_ablation.py` — 13 slow tests
- [ ] End-to-end smoke test: `python run_phase2_ablation.py --data-dir data --num-images 1024 --sigma-thresholds 3.0 6.0`
- [ ] Full run: `python run_phase2_ablation.py --data-dir data --num-images 50000`