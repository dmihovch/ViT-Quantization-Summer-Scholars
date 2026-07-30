# Next Steps: Implementation Roadmap

> **Citations:** All literature references in this document are catalogued with
> full bibliographic details in [`docs/CITATIONS.md`](CITATIONS.md).

## Current State

**Steps 1–6b are complete. Phase 1 is fully implemented.** 82/91 fast tests pass
(9 failures are pre-existing stubs in Phase 2/3 modules). 22 slow tests require
nnsight trace context. Tested with PyTorch 2.12.1, nnsight 0.7.0, NVIDIA RTX 3070.

| Module | Status | Fast tests | Slow tests |
|--------|--------|-----------|------------|
| `src/utils.py` | ✅ Done | `test_utils.py` 3/3 | — |
| `src/model.py` | ✅ Done | — (weights required) | — |
| `src/data_loader.py` | ✅ Done | `test_data_loader.py` 2/2 | — |
| `src/hooks.py` | ✅ Kept (legacy, LayerStats deleted 2026-07-30) | — (tests deleted) | — |
| `src/profiler.py` — single-pass API | ✅ Done | `test_profiler.py` 11/11 | 13 slow |
| `src/profiler.py` — Welford multi-batch | ✅ Done | `test_profiler.py` 21 fast | 7 slow |
| `src/profiler.py` — `histogram_profile_vit` | ✅ Done (Step 6b) | 2 fast | 2 slow |
| `src/exp1_profiling.py` | ✅ Done | — | — |
| `src/plotting.py` — Phase 1 functions | ✅ Done | `test_plotting.py` 2/2 | — |
| `src/ablation.py` | 🔲 Stub | — | — |
| `src/exp2_ablation.py` | 🔲 Stub | — | — |
| `src/integer_gelu.py` | 🔲 Stub | — | — |
| `src/exp3_integer_gelu.py` | 🔲 Stub | — | — |

### Running tests

```bash
# Default (fast only — works on macOS + Linux):
pytest -m "not slow"

# Full suite including ViT-trace tests (Linux / GPU recommended):
pytest -m slow tests/test_profiler.py
```

## Architecture: profiling modules

### `src/hooks.py` — Welford accumulator pipeline (legacy, 3-site)

Registers raw PyTorch forward hooks on `nn.GELU` and `nn.LayerNorm` modules.
Covers 3 of 5 sites: `pre_gelu`, `post_layernorm_1/2`, and `residual_stream`.
**Cannot** capture `pre_softmax` or `post_softmax` — those logits are computed
inline inside Attention.forward() with no `nn.Module` boundary to intercept.

This module is **retained for reference and for its existing tests** but is
**not used in Phase 1** following the Option C decision below.

### `src/profiler.py` — nnsight pipeline  **Primary for all phases**

Wraps a timm ViT with `nnsight.NNsight` and captures all **6** sites per block,
including both attention sites, by intercepting intermediate proxy tensors
inside the trace context.

All single-pass and Welford multi-batch APIs are implemented:
- Population std (ddof=0) throughout, M3/M4 tracked for exact kurtosis.
- `LayerStats` has `m3`, `n_samples`, `per_channel_std`, `per_channel_sum`, `per_channel_sum_sq`.
- `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`, `_site_n`, `run_profiling_dataset_pass` all implemented.
- `histogram_profile_vit` is the one remaining addition (Step 6b).

Full specification: `docs/EXP1-IMPL.md`.

---

### Architecture decision: Option C (resolved)

Three options were considered for collecting dataset-wide statistics across
all 5 measurement sites:

| Option | Correctness | Coverage | Complexity |
|--------|------------|----------|------------|
| (a) Average per-batch `profile_vit` scalars | ❌ incorrect std / kurtosis | all 5 sites | low |
| (b) `hooks.py` for dataset pass, `profiler.py` for spot-checks | ✅ exact mean/std/outlier fracs; approx kurtosis | 3 sites only | low |
| (c) Welford parallel-merge inside `profiler.py` | ✅ same correctness as (b) | **all 6 sites** | medium |

**Decision: Option C.**  The pre-softmax logit distribution is the most
quantization-hostile site in the network and cannot be omitted from the
Phase 1 summary table without a conspicuous methodological gap.  Option B's
3-site coverage is inadequate.  Option C extends `profiler.py` with a
Welford accumulator that receives per-batch stats from `profile_vit` and
merges them using Chan et al. (1983) parallel formula — the same approach
already used in `hooks.py`, now applied to all 6 sites via nnsight.

**Kurtosis is exact** via Pébay (2008) M3/M4 parallel merge. No approximation. No caveat label needed.

**Step 4b-ii (done):** `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`, `_site_n`, and `run_profiling_dataset_pass` are implemented in `profiler.py`.

**Step 4b-iii (done):** `per_channel_std` support added to `profiler.LayerStats`, `_StatsSavers`, `_register_stat_saves`, `_finalize_stats`, `WelfordAccumulator`, `merge_batch_stats`, and `finalize_accumulator`. Per-channel sums and sums-of-squares are tracked for exact cross-batch merging.

---

## Implementation Order

### Step 1: `src/utils.py` — ✅ DONE

- `seed_everything(seed)`, `get_device()`, `ensure_dir(path)`.

**Tests:** `test_utils.py` (3/3 pass)

---

### Step 2: `src/model.py` — ✅ DONE

- `load_vit(device)` — loads `vit_base_patch16_224`, calls
  `disable_fused_attn()`, sets eval mode, derives transform from timm config.
- `evaluate_accuracy(model, loader, device)` — top-1 / top-5 loop inside
  `torch.no_grad()`.
- `disable_fused_attn(model)` — sets `block.attn.fused_attn = False` on
  every block.  **Must be called before wrapping with NNsight** so that
  PyTorch's SDPA path is bypassed and the raw QKᵀ logit matrix is
  materialised.

**Tests:** no fast unit tests (require pretrained weights).

---

### Step 3: `src/data_loader.py` — ✅ DONE

- `build_val_loader(data_dir, transform, batch_size, num_images, device, shuffle=None)`.
  Auto-shuffles subsets for class diversity; full datasets use deterministic order.
  Randomly samples indices via `torch.randperm` when shuffling a subset, enabling
  cross-seed variance.

**Tests:** `test_data_loader.py` (2/2 pass)

---

### Step 4: `src/profiler.py` — ✅ DONE (single-pass API)

See Step 4b below for the dataset-wide extension.

---

### Step 4b: `src/profiler.py` extension — ✅ DONE (Welford multi-batch API + per-channel std)

nnsight-based profiler.  Collects stats at **six sites per block** in a
single forward pass.

#### Measurement sites

| Site identifier | What is captured | How |
|-----------------|------------------|-----|
| `patch_embed/residual_stream` (block 0 only) | Patch-embed output; residual before block 0 | `blocks[0].norm1.input[0][0]` |
| `blocks.{i-1}/residual_stream` (i > 0) | Residual stream entering block i | `blocks[i].norm1.input[0][0]` |
| `blocks.{i}/post_layernorm_1` | Output of `norm1` (pre-attention LN) | `blocks[i].norm1.output` |
| `blocks.{i}/post_layernorm_2` | Output of `norm2` (pre-MLP LN) | `blocks[i].norm2.output` |
| `blocks.{i}/pre_gelu` | Input to the MLP activation | `blocks[i].mlp.act.input[0][0]` |
| `blocks.{i}/pre_softmax` | Raw QKᵀ/√d logit matrix | Reconstructed from `attn.qkv.output` inside trace |
| `blocks.{i}/post_softmax` | Post-softmax attention weights | `attn.attn_drop.input[0][0]` |

#### `LayerStats` dataclass

```python
@dataclass
class LayerStats:
    site_identifier: str          # e.g. "blocks.3/pre_gelu"
    mean: float
    std: float
    kurtosis: float               # excess kurtosis: E[(x−μ)⁴]/σ⁴ − 3 (exact, Pébay 2008)
    m3: float = 0.0               # Σ(x−μ)³ for cross-batch merge
    outlier_fractions: dict[str, float]
    # keys: "3.0_sigma", "4.0_sigma", "6.0_sigma"
    n_samples: int = 0            # total scalar elements
    per_channel_std: list[float] | None = None  # per-channel population σ
    per_channel_sum: list[float] | None = None   # per-channel sum for merge
    per_channel_sum_sq: list[float] | None = None  # per-channel sum-of-squares for merge
```

#### Statistics computed

- **mean, std** — global over all tensor elements (population, ddof=0).
- **kurtosis** — excess kurtosis; Gaussian ≈ 0, heavy-tailed > 0. Exact via Pébay (2008) M3/M4 parallel merge.
- **m3** — third central moment sum Σ(x−μ)³; used internally for cross-batch kurtosis merge.
- **outlier_fractions** — fraction of |x| > k·σ for k ∈ {3.0, 4.0, 6.0}.
- **per_channel_std** — per-channel population σ (pre_gelu, post_layernorm_1/2 only).
- **n_samples** — total scalar elements processed.

#### Workflow

```python
from nnsight import NNsight
from src.model import load_vit, disable_fused_attn
from src.profiler import profile_vit, save_profiling_result

model, transform = load_vit(device)    # disable_fused_attn called inside
wrapped = NNsight(model)

result = profile_vit(wrapped, image_batch)   # single forward pass
save_profiling_result(result, output_dir / "profiling_result.json")
```

#### nnsight compatibility

- Version: nnsight ≥0.7.0 (tested with 0.7.0).  PyTorch ≥2.5 required.
- Three API changes from nnsight 0.2.x are documented in `EXP1-IMPL.md` §A:
  `.input` returns tensor directly, forward-pass dependency ordering, and
  `.save()` returns concrete tensor.
- `_finalize_stats` uses a `_val()` helper that is compatible with both
  nnsight <0.3 (proxy objects with `.value`) and ≥0.3 (concrete tensors).

**Tests:**
- Fast: `test_profiler.py -m "not slow"` → 82/82 pass (excluding Phase 2/3 stubs)
- Slow: `test_profiler.py -m slow` → 22 tests; require nnsight trace context

#### Step 4b-ii + 4b-iii: Welford multi-batch extension + per-channel std — ✅ DONE

Implemented in `profiler.py`:
- `WelfordAccumulator` — carries `n, mean, M2, M3, M4, outlier_counts, per_channel_sum, per_channel_sum_sq` across batches.
- `_site_n(site_id, B, N, D, D_mlp, num_heads) -> int` — top-level function.
- `merge_batch_stats(acc, batch_stats, batch_n)` — exact Pébay (2008) parallel merge for M2, M3, M4 + per-channel sum accumulation.
- `finalize_accumulator(acc) -> LayerStats` — exact global mean, std, kurtosis, outlier_fractions, per_channel_std.
- `run_profiling_dataset_pass(wrapped_model, loader, device) -> dict[SiteId, LayerStats]` — iterates loader, calls `profile_vit` per batch, merges, finalizes.
- `per_channel_std` support: `LayerStats` gains `per_channel_std`, `per_channel_sum`, `per_channel_sum_sq` fields. `_register_stat_saves` accepts `track_per_channel=True`. `profile_vit` passes `track_per_channel=True` for `pre_gelu`, `post_layernorm_1`, `post_layernorm_2`.

**Tests added** (`test_profiler.py`):
- Fast: `test_welford_accumulator_construction`, `test_merge_batch_stats_single_batch`, `test_finalize_accumulator_two_equal_batches`, `test_merge_batch_stats_exact_kurtosis_known_data`, `test_merge_batch_stats_raises_on_zero_batch_n`, `test_finalize_accumulator_raises_on_zero_n`, `test_site_n_returns_correct_counts`, `test_merge_batch_stats_outlier_accumulation`, `test_per_channel_merge_two_batches` — all pass.
- Slow: `test_slow_run_profiling_dataset_pass_site_coverage`, `test_slow_run_profiling_dataset_pass_exact_n_samples`, `test_slow_run_profiling_dataset_pass_per_channel_std_present`, `test_slow_run_profiling_dataset_pass_per_channel_std_shape` — require PyTorch 2.2.x + nnsight 0.2.21.

**Reference:** Pébay (2008) *Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments*, Sandia SAND2008-6212.

---

### Step 5: `src/exp1_profiling.py` — ✅ DONE

Wire everything together for Phase 1.

```python
def run(config: ProfilingConfig) -> None:
```

#### Architecture decision: Option C (resolved — see Architecture section above)

`exp1_profiling.run()` uses `profiler.run_profiling_dataset_pass` (Step 4b)
to collect exact global statistics across all **6 sites** per block.  Do not
use `hooks.py` or call `profile_vit` directly in the dataset loop.

#### Workflow

1. `model, transform = load_vit(config.device)`.
2. `wrapped = NNsight(model)`.
3. `loader = build_val_loader(config.data_dir, transform, config.batch_size,
   config.num_images, config.device)`.
4. `stats = run_profiling_dataset_pass(wrapped, loader, config.device)`
   inside `torch.no_grad()` (the function handles the loop internally).
5. `ensure_dir(config.output_dir)`.
6. `save_profiling_result(ProfilingResult(stats, num_blocks, batch_shape),
   config.output_dir / "profiling_result.json")`.
7. Generate histograms and heatmaps by calling `plotting.*` functions.

**When complete:** `python run_phase1_profiling.py --num-images 1024` should
produce `outputs/phase1-profiling/profiling_result.json` with entries for
all 6 sites across all 12 blocks.

**Status:** ✅ Implemented. `run()` loads the model, builds the loader, calls
`run_profiling_dataset_pass`, saves `profiling_result.json`, and generates
histograms + per-channel σ heatmap. `batch_shape` is derived from the first
actual batch (not hardcoded).

> **Output filename change:** Phase 1 now writes `profiling_result.json`
> (via `profiler.save_profiling_result`) instead of `layer_stats.json`
> (via `hooks.save_stats`).  Update `AblationConfig.layer_stats_path` default
> in `config.py` and the Phase 2 workflow accordingly.

---

### Step 6: `src/plotting.py` — ✅ Phase 1 functions done

Six figure functions.  Do this after Phase 1 runs so you have real data.

- `plot_activation_histogram(activations, layer_name, site, output_path, log_scale)`.
- `plot_per_channel_std_heatmap(per_channel_stds, layer_names, output_path)`.
- `plot_attention_entropy_heatmap(entropies, layer_names, output_path)`.
- `plot_accuracy_vs_threshold(results, output_path)`.
- `plot_pct_zeroed_per_layer(results, sigma_k, site, output_path)`.
- `plot_lut_vs_fp32(lut, output_path)`.

**Tests:** `test_plotting.py` (add smoke tests per function)

---

### Step 6b: `histogram_profile_vit` + histogram pipeline rewrite — ✅ DONE

Implemented in `profiler.py`:
- `histogram_profile_vit(wrapped_model, input_batch, block_indices)` — runs
  one nnsight trace, saves full activation tensors at all 6 sites for the
  specified blocks.  Per-block `num_heads`, `head_dim`, and `scale` (not
  block 0's).  Returns CPU tensors.
- `_plot_histograms(wrapped, transform, config, output_dir)` — builds a
  shuffled loader sampling from the full dataset, calls
  `histogram_profile_vit`, and saves 18 PNGs (6 sites × 3 blocks).
- `build_val_loader` auto-shuffle: `shuffle=None` auto-selects based on
  whether `num_images` is a subset.

**Tests:**
- Fast: `test_histogram_profile_vit_raises_on_non_4d_input`,
  `test_histogram_profile_vit_raises_on_model_without_blocks`
- Slow: `test_slow_histogram_profile_vit_shapes`,
  `test_slow_pre_softmax_reconstruction_matches_manual`

---

### Step 6c: Documentation fixes — ✅ DONE

- `WelfordAccumulator.outlier_counts` and `finalize_accumulator` docstrings
  document the per-batch σ convention.
- Duplicate `ProfilingError` removed from `src/exceptions.py`.

---

### Step 7: `src/ablation.py` — 🔲

Phase 2 — outlier zeroing.

- `compute_pct_zeroed(tensor, threshold)`.
- `build_zeroing_hook(layer_name, site, threshold, stats)`.
- `patch_model_for_ablation(model, sigma_k, site, layer_stats)`.
- `compute_entropy_delta(stats_before, stats_after)`.
- `save_ablation_results(results, path)`.

**Tests:** `test_ablation.py` (4 tests; extend for `site` param)

---

### Step 8: `src/exp2_ablation.py` — 🔲

Wire Phase 2.  Run three ablation sweeps (one per site: `pre_gelu`,
`pre_softmax`, `residual_stream`).  For `pre_softmax`, also collect
`compute_entropy_delta` across the sweep.

#### Attention-site σ source (updated for Option C)

Phase 1 now produces dataset-wide `pre_softmax` std via
`run_profiling_dataset_pass`.  Load `profiling_result.json` and read
`pre_softmax` std directly — no single-batch estimation needed.
The `attn_profile_num_images` and `attn_profile_seed` fields of
`AblationConfig` are **deprecated** and should be ignored; they remain in
the dataclass for backwards compatibility but log a deprecation warning if
non-default values are passed.

#### Workflow

1. Load model (`load_vit`) and Phase 1 stats
   (`profiler.load_profiling_result(config.layer_stats_path)`).
2. All 6 sites' σ values are available directly from the loaded result.
3. For each site in `{pre_gelu, pre_softmax, residual_stream}`:
   - For each `k` in `config.sigma_thresholds`:
     a. `handles = patch_model_for_ablation(model, k, site, layer_stats)`
     b. Evaluate top-1 / top-5 accuracy over the full val loader.
     c. Record `AblationResult` per layer (pct_zeroed, top1, top5).
     d. `remove_hooks(handles)` before the next iteration.
   - For `pre_softmax` only: also record `compute_entropy_delta` across k.
4. `save_ablation_results(results, config.output_dir / "ablation_results.csv")`.
5. Generate plots via `plotting.*`.

---

### Step 9: `src/integer_gelu.py` — 🔲

Phase 3 — LUT construction.

- `build_lut(scale_in, scale_out)`.
- `apply_lut(tensor, lut)`.
- `compare_lut_vs_fp32(lut, scale_in)`.

**Tests:** `test_integer_gelu.py` (4 tests)

---

### Step 10: `src/exp3_integer_gelu.py` — 🔲

Wire Phase 3.

---

## What to Read (and When)

Read these **as you implement**, not all upfront.

### Before Step 4b and Step 5 (Welford extension + exp1)

- **Pébay (2008) — Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments**
  Sandia Technical Report SAND2008-6212
  > The exact parallel higher-moments formula (Eq. 3.1–3.4) used in `merge_batch_stats` for M2, M3, and M4. Required reading before implementing Step 4b-ii. The Chan et al. (1983) formula for M2 is a special case of this.

- **Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words (ViT)**
  https://arxiv.org/abs/2010.11929
  > Read Section 3.1 (the encoder block diagram).  You are hooking at five
  > sites; this gives you the ground truth for what each site measures.

- **nnsight documentation** — https://nnsight.net/documentation/
  > Read the trace context manager and `.save()` / `.value` sections before
  > debugging any proxy-related issues in `profiler.py`.

- **Sun et al. (2023) — Massive Activations in Large Language Models**
  https://arxiv.org/abs/2402.17762
  > Describes "massive activation" outliers in attention and residual streams.
  > Use their per-channel magnitude analysis as a template.

- **Zhai et al. (2023) — Stabilizing Transformer Training by Preventing
  Attention Entropy Collapse**
  https://arxiv.org/abs/2303.06296
  > Published at ICML 2023. Provides the entropy measurement formula for
  > ``post_softmax`` analysis and defines "entropy collapse" as a
  > diagnostic signal.

### Before Step 7 (ablation)

- **Wei et al. (2022) — Outlier Suppression**
  https://arxiv.org/abs/2209.13325
  > Studies zeroing vs. clamping vs. shifting for transformer outliers.

- **Bondarenko et al. (2023) — Understanding ViT Quantization Challenges**
  https://arxiv.org/abs/2109.12948
  > Studies **ViT specifically** (not LLMs).  Identifies inter-channel
  > variance and the exact quantization failure modes you are solving.  Read
  > before interpreting Phase 1 histograms so you know what to look for.

### Before Step 9 (integer GELU)

- **Kim et al. (2021) — I-BERT**
  https://arxiv.org/abs/2101.01321
  > Primary reference for integer-only GELU (polynomial approximation).
  > Compare against your LUT approach.

- **I-ViT — Integer-only Quantization for Efficient Vision Transformer Inference**
  > Introduces `ShiftGELU`.  Compare against your LUT for Jetson hardware.

### Background (read anytime)

- **Gholami et al. (2021) — A Survey of Quantization Methods**
  https://arxiv.org/abs/2103.13630
  > The canonical quantization survey.  Skim Sections 2–3 for vocabulary.
