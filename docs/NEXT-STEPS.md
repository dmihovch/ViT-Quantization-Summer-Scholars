# Next Steps: Implementation Roadmap

## Current State

**Steps 1–4 are complete.  The profiling framework has been fully refactored
to use nnsight.  48/48 fast tests pass; 13 slow tests (nnsight trace + full
ViT forward) are gated behind `@pytest.mark.slow` for Linux / GPU runs.**

| Module | Status | Fast tests | Slow tests |
|--------|--------|-----------|------------|
| `src/utils.py` | ✅ Done | `test_utils.py` 3/3 | — |
| `src/model.py` | ✅ Done | — (weights required) | — |
| `src/data_loader.py` | ✅ Done | `test_data_loader.py` 2/2 | — |
| `src/hooks.py` | ✅ Kept (legacy Welford pipeline) | `test_hooks.py` 26/26 | — |
| `src/profiler.py` | ✅ Done (nnsight, all 5 sites) | `test_profiler.py` 11/11 | 13 slow tests |
| `src/exp1_profiling.py` | 🔲 Stub | — | — |
| `src/plotting.py` | 🔲 Stub | — | — |
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

### macOS + PyTorch 2.2.x note

Any test that calls an nnsight `model.trace(...)` context from within pytest
on macOS with PyTorch 2.2.x aborts with a C-level signal-handler conflict
(observed in `layer_norm` and `conv2d` C++ kernels).  The same code runs
correctly when called from a plain Python script.  All nnsight trace tests
are marked `@pytest.mark.slow` to exclude them from the default run on
macOS.  They pass on Linux (verified by smoke-testing outside pytest).

---

## Architecture: the two profiling modules

The project now has two profiling modules that serve different purposes:

### `src/hooks.py` — legacy Welford accumulator pipeline

Registers raw PyTorch forward hooks on `nn.GELU` and `nn.LayerNorm` modules.
Accumulates stats online across an **arbitrary number of batches without
storing any raw tensor data**.  Terminates at 3 of 5 sites (`pre_gelu`,
`post_layernorm`, `residual_stream`).

**Use this when:** running a full dataset pass where memory is the constraint
(e.g. profiling all 50 000 ImageNet validation images).

### `src/profiler.py` — nnsight single-pass pipeline ✨ **Primary**

Wraps a timm ViT with `nnsight.NNsight` and collects statistics at **all 5
sites** in a single forward pass, with no raw tensors retained.  Captures
both attention sites (`pre_softmax`, `post_softmax`) by:
- Reconstructing QKᵀ/√d from `attn.qkv.output` inside the trace (pre-softmax).
- Reading `attn.attn_drop.input` (post-softmax attention weights).

**Use this when:** running a single batch for profiling or the exp1 pipeline.

The nnsight approach replaces the raw-hook workarounds entirely.  `hooks.py`
is retained for the multi-batch accumulation use case; `profiler.py` is the
canonical implementation going forward.

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

- `build_val_loader(data_dir, transform, batch_size, num_images, device)`.

> **Implementation note:** `ImageFolder` raises a bare `FileNotFoundError`
> (not our `DataDirectoryError`) when the directory has no class
> subdirectories.  The implementation wraps and re-raises as
> `DataDirectoryError`.

**Tests:** `test_data_loader.py` (2/2 pass)

---

### Step 4: `src/profiler.py` — ✅ DONE (replaces raw-hook approach)

nnsight-based profiler.  Collects stats at **five sites per block** in a
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
    kurtosis: float               # excess kurtosis: E[(x−μ)⁴]/σ⁴ − 3
    outlier_fractions: dict[str, float]
    # keys: "3.0_sigma", "5.0_sigma", "8.0_sigma"
```

#### Statistics computed

- **mean, std** — global over all tensor elements.
- **kurtosis** — excess kurtosis; Gaussian ≈ 0, heavy-tailed > 0.
- **outlier_fractions** — fraction of |x| > k·σ for k ∈ {3.0, 5.0, 8.0}.

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

- Version pinned to `nnsight==0.2.21` (`requirements.txt` / `environment.yml`).
- nnsight pulls `transformers 5.x` as a dependency; that package emits a
  warning about PyTorch < 2.4 but it is benign — nnsight itself works fine
  with PyTorch 2.2.x for non-HuggingFace models.
- `nnsight.NNsight(module)` wraps any `nn.Module`.  Access sub-module
  proxies by the same dotted path: `wrapped.blocks[3].norm1.output`.
- Proxies are only available **inside** a `with wrapped.trace(x):` block.
  Use `.save()` to retain values after the context closes; access via
  `.value` attribute after the context exits.

**Tests:**
- Fast (no trace): `test_profiler.py -m "not slow"` → 11/11 pass
- Slow (full trace): `test_profiler.py -m slow` → 13 tests; run on Linux

---

### Step 5: `src/exp1_profiling.py` — 🔲 NEXT

Wire everything together for Phase 1.

```python
def run(config: ProfilingConfig) -> None:
```

1. `model, transform = load_vit(config.device)`.
2. `loader = build_val_loader(config.data_dir, transform, config.batch_size,
   config.num_images, config.device)`.
3. `wrapped = NNsight(model)` — model already has `fused_attn=False`.
4. For each batch in loader:
   - Call `profile_vit(wrapped, images.to(config.device))`.
   - Accumulate stats by merging each batch's `ProfilingResult` into a
     running aggregate (or collect per-batch and average at the end).
5. `ensure_dir(config.output_dir)`.
6. `save_profiling_result(aggregate_result, config.output_dir / "profiling_result.json")`.
7. Generate histograms and heatmaps by calling `plotting.*` functions.

> **Design decision needed:** `profile_vit` runs one forward pass and
> returns single-pass stats.  For a full dataset run you need either:
> (a) Run `profile_vit` per batch and average the returned scalars — simple
>     but loses distributional accuracy (means of means ≠ global mean for
>     kurtosis).
> (b) Keep `hooks.py` accumulator pipeline for the multi-batch dataset pass,
>     and use `profiler.py` only for single-batch previews.
> (c) Extend `profiler.py` with a multi-batch accumulation loop that calls
>     `profile_vit` per batch and merges accumulators via Welford's
>     parallel-groups formula.
>
> **Recommendation:** Option (b) is cleanest for Phase 1.  Use `hooks.py`
> for the dataset pass (it was built for this), derive the histogram data
> from a separate lightweight sampling hook, and use `profiler.py` for any
> single-batch spot-checks.

**When complete:** `python run_phase1_profiling.py --num-images 128` should
produce `outputs/phase1-profiling/profiling_result.json`.

---

### Step 6: `src/plotting.py` — 🔲

Six figure functions.  Do this after Phase 1 runs so you have real data.

- `plot_activation_histogram(activations, layer_name, site, output_path, log_scale)`.
- `plot_per_channel_std_heatmap(per_channel_stds, layer_names, output_path)`.
- `plot_attention_entropy_heatmap(entropies, layer_names, output_path)`.
- `plot_accuracy_vs_threshold(results, output_path)`.
- `plot_pct_zeroed_per_layer(results, sigma_k, site, output_path)`.
- `plot_lut_vs_fp32(lut, output_path)`.

**Tests:** `test_plotting.py` (add smoke tests per function)

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

### Before Step 5 (exp1 + profiling)

- **Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words (ViT)**
  https://arxiv.org/abs/2010.11929
  > Read Section 3.1 (the encoder block diagram).  You are now hooking at
  > five sites; this gives you the ground truth for what each site measures.

- **nnsight documentation** — https://nnsight.net/documentation/
  > Read the trace context manager and `.save()` / `.value` sections before
  > debugging any proxy-related issues in `profiler.py`.

- **Sun et al. (2023) — Massive Activations in Large Language Models**
  https://arxiv.org/abs/2402.17762
  > Describes "massive activation" outliers in attention and residual streams.
  > Use their per-channel magnitude analysis as a template.

- **Zhai et al. (2023) — Stabilizing Transformer Training by Preventing
  Attention Entropy Collapse**
  https://arxiv.org/abs/2204.09548
  > Provides the entropy measurement formula for `post_softmax` analysis.

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
