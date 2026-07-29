# ⚠️ MISTAKES LEDGER

> **This document is a record of things that went wrong.**
>
> Every entry here represents a mistake that was made, an approach that was
> tried and abandoned, or a design that looked reasonable and turned out to be
> incorrect or harmful.  **Do not implement anything described in this file.**
> The correct implementation is in `NEXT-STEPS.md` and `EXP1-IMPL.md`.
>
> This ledger exists so that:
> 1. The same mistake is not made twice.
> 2. Context-switching agents entering this codebase understand *why* things
>    are the way they are — the current design is not arbitrary, it is the
>    residue of these failures.
> 3. The reasoning behind discarded approaches is preserved, not lost.
>
> **If you are an agent implementing Phase 1:** read this first. Every section
> below has a "What to do instead" note pointing to the correct spec.

---

## 1. Statistical mistakes

### 1.1 — Using sample std (`ddof=1`) instead of population std (`ddof=0`)

**What happened:** The first draft of `_register_stat_saves` called `t.std()`
with no arguments. PyTorch defaults to `correction=1` (Bessel's correction,
sample std). This introduced a systematic negative bias of `(n−1)/n` per batch
into every variance value fed into `merge_batch_stats`. Because the Welford
accumulator tracks population variance, the ddof mismatch was a silent
correctness bug — no error, just subtly wrong numbers in every merged statistic.

**Why it's wrong:** Bessel's correction is appropriate when you are estimating
the variance of an unobserved population from a sample. We are not doing that.
We are measuring a fully-observed finite set of activation values — every
element in the batch tensor is known exactly. The correct statistic is the
population std. There is no statistical justification for Bessel's correction
in this context.

**What to do instead:** Always use `t.std(correction=0)` in `profiler.py`.
This is enforced throughout `_register_stat_saves`, the outlier threshold
computations, and everywhere in the Welford pipeline. See `EXP1-IMPL.md §2.1`.

---

### 1.2 — Approximate kurtosis: accumulating `(x − batch_mean)⁴` per batch

**What happened:** The original plan accumulated kurtosis across batches by
summing `(x − batch_mean)⁴` where `batch_mean` is the local mean of the
current batch, not the global mean. The intent was to avoid storing raw
tensors. The result is a biased approximation with no bounded error guarantee.

**Why it's wrong:** When per-batch means differ (they always do), centring
each batch on its own mean introduces cross-batch error into the accumulated
M4. The error is unbounded — it grows with the variance of per-batch means.
An approximation with no quantifiable error cannot appear in a published
table without a caveat that would undermine the entire measurement.

**What to do instead:** Use the exact Pébay (2008) parallel higher-moments
merge. Track `M3 = Σ(x−μ)³` and `M4 = Σ(x−μ)⁴` as running sums.
`merge_batch_stats` implements the full Pébay merge for M2, M3, and M4.
Kurtosis from `finalize_accumulator` is exact. See `EXP1-IMPL.md §2.2`.

**Reference:** Pébay (2008), *Formulas for Robust, One-Pass Parallel
Computation of Covariances and Arbitrary-Order Statistical Moments*,
Sandia SAND2008-6212, Eq. 3.1–3.4.

---

### 1.3 — Outlier fractions are not relative to global σ

**What happened:** This is not an outright mistake but a subtle definitional
trap. `outlier_fractions` in `LayerStats` and `WelfordAccumulator` are computed
relative to **per-batch σ**, then accumulated. The final fraction is a weighted
average of per-batch outlier rates — it is **not** the fraction of elements
that exceed `k · σ_global`.

**Why it matters:** A reader who does not know this will assume the outlier
fractions are relative to the global std. They are not. If per-batch σ varies
meaningfully across batches, the two statistics can differ noticeably.

**What to do instead:** This is documented in `WelfordAccumulator.outlier_counts`
and `finalize_accumulator` docstrings. Do not change the implementation — the
per-batch convention is well-defined and appropriate for per-layer quantization
calibration. Just be aware of what the number means.

---

## 2. Element count mistakes

### 2.1 — Deriving token count N from `batch_result.batch_shape[2]`

**What happened:** An early draft of `run_profiling_dataset_pass` derived N
(the token sequence length) as `batch_result.batch_shape[2]`. That index is
the image height — 224 pixels — not the number of tokens.

**Why it's wrong:** For ViT-B/16 on 224×224 inputs, the token sequence length
is `(224/16)² + 1 = 197` (196 patch tokens plus the CLS token). Using 224
instead would inflate `batch_n` by a factor of 224/197 ≈ 1.14, corrupting
every Welford merge silently — no shape error, just wrong statistics.

**What to do instead:** Always derive N as:
```python
N = inner_model.patch_embed.num_patches + 1  # 197 for ViT-B/16 on 224×224
```
This is enforced in both `run_profiling_dataset_pass` and `profile_vit` with
an explicit comment. See `EXP1-IMPL.md §3.1`.

---

### 2.2 — Defining `_site_n` as a closure inside the batch loop

**What happened:** A draft defined `_site_n` as a nested function inside the
`for batch_idx` loop, capturing `B`, `N`, `D`, `D_mlp`, `num_heads` by
reference from the enclosing scope.

**Why it's wrong:** Python closures capture variables by reference, not by
value. If the function were called after any of those variables were rebound
(e.g., `B` changes on the last partial batch), it would silently return wrong
counts. It was also wastefully re-defined on every loop iteration.

**What to do instead:** `_site_n` is a top-level module function that receives
all dimensions as explicit arguments. Architecture constants are extracted from
the model once before the loop. See `EXP1-IMPL.md §3.1`.

---

### 2.3 — Re-accessing architecture constants inside the batch loop

**What happened:** A draft accessed `inner_model.blocks[0].attn.num_heads`,
`inner_model.embed_dim`, and `inner_model.blocks[0].mlp.fc1.out_features`
on every iteration of the batch loop.

**Why it's wrong:** These are model constants. They do not change between
batches. Accessing them inside the loop is wasteful and implies they might
change, which misleads readers. It also makes the code harder to audit because
the constants are not visibly extracted before use.

**What to do instead:** Extract `N`, `D`, `num_heads`, `D_mlp` once before
the loop. This is required in both `run_profiling_dataset_pass` and
`profile_vit`. See `EXP1-IMPL.md §3.5`.

---

## 3. Histogram design mistakes

### 3.1 — Synthetic Gaussian histograms: the entire first approach

**What happened:** The first histogram implementation drew `50_000` synthetic
samples from `N(mean, std²)` using `numpy.random.default_rng().normal(...)` and
plotted those. Every histogram title was labelled `[reconstructed N(μ,σ²)]`.

**Why it's wrong:** The entire point of Phase 1 is to characterise heavy-tailed,
non-Gaussian activation distributions. A Gaussian reconstruction drawn from
`N(μ, σ²)` **by construction cannot show heavy tails** — it will always look
Gaussian regardless of what the actual distribution is. Annotating the kurtosis
value on top of a Gaussian bell curve is not a substitute for showing the real
distribution. The spec requires "log-scale histograms per site showing
heavy-tailed distributions." Gaussian reconstructions do not satisfy this.

**This implementation is completely superseded.** Do not implement it.
The stale code templates for the old `_plot_histograms` are preserved in
`EXP1-IMPL.md §8.3` only as a warning — they are marked SUPERSEDED there.

**What to do instead:** Use `histogram_profile_vit` to collect real activation
tensors for a representative batch and generate histograms from those.
See `EXP1-IMPL.md §9`.

---

### 3.2 — Prefix slicing for subsampling: `t.reshape(-1)[:50_000]`

**What happened:** Before the full-tensor approach was settled on, an
intermediate design proposed saving at most 50,000 scalar values per site by
taking a prefix slice of the flattened tensor.

**Why it's wrong:** PyTorch tensors are row-major (C-contiguous). For a
`(B, N, D)` tensor, `reshape(-1)[:50_000]` takes elements in order:
*(batch 0, token 0, channels 0..D−1), (batch 0, token 1, channels 0..D−1), ...*

For `pre_gelu` with `D_mlp = 3072`, 50,000 / 3,072 ≈ 16 tokens. You get the
first 16 tokens from image 0 and nothing from images 1–63. The histogram
represents the activation of one image's early patches, not the batch
distribution. This is a systematic spatial and per-image bias, not a random
sample.

**What to do instead:** Save the full tensor. At B=64, the largest site
(`pre_gelu`) is 155 MB per block. Three blocks is ~1.5 GB — acceptable on
any GPU with ≥8 GB VRAM. No subsampling is needed. See `EXP1-IMPL.md §9`.

---

### 3.3 — The `--spot-batch` CLI flag design

**What happened:** An earlier spec added a `--spot-batch` boolean flag to
`run_phase1_profiling.py` and a `spot_batch: bool = False` field to
`ProfilingConfig`, making the real-activation histogram pass optional.

**Why it's wrong:** The histogram deliverable ("log-scale histograms per site
showing heavy-tailed distributions") is not optional — it is a Phase 1
requirement. Making the only mechanism that produces valid histograms
opt-in behind a flag means Phase 1 could be declared "complete" with only
Gaussian reconstructions in the output directory. The flag also adds config
complexity with no benefit since the histogram pass costs one forward pass
and ~1.5 GB of transient memory.

**This flag was never implemented and should never be added.**

**What to do instead:** The histogram pass is unconditional. `_plot_histograms`
is called at the end of every `exp1_profiling.run()` execution with no guard.
See `EXP1-IMPL.md §9`.

---

### 3.4 — Passing `loader` to `_plot_histograms` instead of `transform`

**What happened:** An intermediate spec gave `_plot_histograms` the signature
`_plot_histograms(wrapped, loader, config)` and retrieved the histogram batch
as `next(iter(loader))` from the existing Welford-pass loader.

**Why it's wrong:** The Welford loader uses `shuffle=False`, so `next(iter(loader))`
returns the very first batch from the unshuffled ImageNet validation set.
ImageNet is stored in alphabetical order by WordNet synset ID. The first 64
images are drawn from the first one or two alphabetical classes (tench and
goldfish — both fish). Histograms generated from 64 fish images may not
represent the full distribution of ViT activations across ImageNet, since the
model routes different semantic content differently.

**What to do instead:** `_plot_histograms` takes `transform` (not `loader`) and
builds its own shuffled loader internally:
```python
histogram_loader = build_val_loader(
    config.data_dir, transform, config.batch_size,
    config.num_images, config.device, shuffle=True,
)
```
`seed_everything(42)` is called in `main()` before `run()`, so the shuffled
draw is deterministic across runs. `build_val_loader` accepts `shuffle: bool = False`
(new parameter added in Step 6b; default preserves existing Welford-pass behaviour).
See `EXP1-IMPL.md §9`.

---

### 3.5 — `tensor.numpy()` without `.detach().cpu()` guard

**What happened:** An intermediate spec showed:
```python
activations = tensor.numpy().ravel().astype(np.float32)
```

**Why it's wrong:** `.numpy()` fails on CUDA tensors with "can't convert CUDA
tensor to numpy." Even if `histogram_profile_vit` returns CPU tensors today,
that is an implementation detail that could change. Relying on it silently
makes the call fragile.

**What to do instead:**
```python
activations = tensor.detach().cpu().numpy().ravel().astype(np.float32)
```
This is explicit about both device and gradient detachment, and is robust to
any future changes in `histogram_profile_vit`'s return convention.

---

### 3.6 — Importing `histogram_profile_vit` inside `_plot_histograms`

**What happened:** A spec draft imported `histogram_profile_vit` inside the
function body:
```python
def _plot_histograms(...):
    from src.profiler import histogram_profile_vit
    ...
```

**Why it's wrong:** Both `_plot_histograms` and `histogram_profile_vit` live
in modules that are already imported at the top of `exp1_profiling.py`. The
internal import is unnecessary, implies a circular import concern that doesn't
exist, and violates the project's convention of module-level imports. A junior
coder seeing this pattern may cargo-cult it in contexts where it actually does
cause circular imports.

**What to do instead:** Import `histogram_profile_vit` at the top of
`exp1_profiling.py` alongside the other `src.profiler` imports. See
`EXP1-IMPL.md §8.1`.

---

## 4. Architecture approach mistakes

### 4.1 — Option A: Averaging per-batch `profile_vit` scalar outputs

**What happened:** The first proposed approach for dataset-wide statistics was
to run `profile_vit` on each batch, collect the returned `LayerStats`, and
average the scalar outputs (mean, std, kurtosis) across batches.

**Why it's wrong:** You cannot recover global statistics by averaging per-batch
statistics. Specifically:
- **Mean:** batch means average correctly only if all batches are the same size.
  With a partial last batch this is wrong.
- **Std:** the average of per-batch stds is not the global std. It ignores
  between-batch variance entirely.
- **Kurtosis:** the average of per-batch kurtosis values has no relationship
  to global kurtosis. They are different statistics.

This approach would have produced silently incorrect numbers with no error signal.

**What to do instead:** Use the Welford parallel-merge accumulator (Option C).
See `NEXT-STEPS.md` and `EXP1-IMPL.md §3`.

---

### 4.2 — Option B: `hooks.py` for dataset pass, `profiler.py` for attention

**What happened:** Option B proposed using `hooks.py` (PyTorch raw forward
hooks) for the dataset-wide pass to get exact statistics for 3 sites, and
`profile_vit` (nnsight) for a single representative batch to estimate
`pre_softmax` std.

**Why it's wrong:**
1. **Coverage gap:** `hooks.py` can only intercept at `nn.Module` boundaries.
   The `pre_softmax` QKᵀ/√d logit matrix is computed inline inside
   `Attention.forward()` with no module boundary. `hooks.py` cannot see it.
   Covering only 3 of 6 sites is an inadequate Phase 1 characterisation —
   the pre-softmax logit distribution is the most quantization-hostile site.
2. **Single-batch σ estimate for `pre_softmax`:** Using one batch to estimate
   σ for a threshold that is later applied dataset-wide is methodologically
   inconsistent. The estimate is batch-dependent and must carry a caveat.
3. **Two separate pipelines:** Maintaining both a hooks pipeline and an nnsight
   pipeline for the same experiment adds complexity with no benefit once Option
   C is available.

**What to do instead:** Option C — Welford merge inside `profiler.py` (nnsight)
covering all 6 sites exactly. `hooks.py` is retained for reference and its
existing tests but is not used in any phase. See `NEXT-STEPS.md` architecture
decision section.

---

## 5. Test infrastructure mistakes

### 5.1 — `TensorDataset` constructed without labels

**What happened:** An early slow test created a `TensorDataset` with only
images and no labels, then tried to unpack batches as `for images, _ in loader`.

```python
dataset = TensorDataset(torch.randn(4, 3, 224, 224))  # WRONG
```

**Why it's wrong:** `TensorDataset` with a single tensor yields single-element
tuples. Unpacking with `images, _` raises `ValueError: not enough values to
unpack`.

**What to do instead:**
```python
images = torch.randn(4, 3, 224, 224)
labels = torch.zeros(4, dtype=torch.long)
dataset = TensorDataset(images, labels)
```
All slow tests in `test_profiler.py` now use this pattern.

---

### 5.2 — Tests not updated when `_register_stat_saves` signature changed

**What happened:** When `n_samples: int` was added as a required third
argument to `_register_stat_saves`, two existing slow tests continued calling
it with the old 2-argument signature. No `TypeError` was raised at import time
because the function is only called inside an nnsight trace context — the error
would only surface at runtime during the trace.

**Why it's dangerous:** Silent signature drift in trace-time calls is
particularly hard to catch. The function appears to be called correctly at the
call site; the error only materialises when the trace executes.

**What to do instead:** When adding required arguments to any function called
inside a trace context, grep for all call sites immediately and update them.
Assert the new field in at least one test.

---

## 6. Documentation mistakes

### 6.1 — `batch_shape` hardcoded as `(config.batch_size, 3, 224, 224)`

**What happened:** An `exp1_profiling.run()` draft hardcoded the batch shape
metadata as `(config.batch_size, 3, 224, 224)`.

**Why it's wrong:** When `num_images % batch_size != 0`, the final batch is
smaller than `config.batch_size`. The stored metadata would misrepresent the
actual input shape. Any code that reads `batch_shape` to infer sizes would get
a wrong answer for an uneven final batch.

**What to do instead:**
```python
first_images, _ = next(iter(loader))
actual_batch_shape = tuple(first_images.shape)
```
`build_val_loader` uses `shuffle=False` for the Welford pass, so the first
batch is deterministic.

---

### 6.2 — "Five sites" used everywhere instead of "six sites"

**What happened:** The project's framework spec (`vit_profiling_framework.md`)
defines five *conceptual* measurement targets. This led to "five sites" being
written into many docstrings, comments, and test descriptions. In fact,
`profile_vit` instruments **six** distinct site IDs per block because
Post-LayerNorm is split into two separate module outputs (`norm1.output` =
`post_layernorm_1` and `norm2.output` = `post_layernorm_2`).

**Why it matters:** An agent told to instrument "five sites" will miss one.

**Definitive list of six site IDs per block:**
1. `{scope}/residual_stream`
2. `blocks.{i}/post_layernorm_1`
3. `blocks.{i}/post_layernorm_2`
4. `blocks.{i}/pre_gelu`
5. `blocks.{i}/pre_softmax`
6. `blocks.{i}/post_softmax`

---

### 6.3 — `ProfilingError` defined twice in `exceptions.py`

**What happened:** `src/exceptions.py` had two class definitions for
`ProfilingError` with slightly different docstrings. The second silently
shadowed the first. Python uses the last definition; the first was dead code.

**Resolution:** Already fixed. Only one definition remains. Do not add another.

---

### 6.4 — `AblationConfig.layer_stats_path` docstring references old filename

**What happened:** `AblationConfig.layer_stats_path` was documented as
expecting `layer_stats.json` (from `hooks.save_stats`). Phase 1 now produces
`profiling_result.json` via `profiler.save_profiling_result`.

**Current status:** The docstring fix and `exp2_ablation.py` update are deferred
to Phase 2 implementation. When implementing Phase 2, update the docstring to
reference `profiling_result.json` and call `profiler.load_profiling_result`
instead of `hooks.load_stats`. See `open-issues.md §7.1`.

---

### 6.5 — `EXP1-IMPL.md §8.3` left stale after histogram redesign

**What happened:** After the histogram pipeline was redesigned (§3.1–3.6 above),
`EXP1-IMPL.md §8.3` still contained a complete, correct-looking implementation
of the old synthetic Gaussian `_plot_histograms`. A coding agent reading the
file would encounter the stale template *before* encountering the replacement
in §9 and might implement the wrong one.

**Resolution:** §8.3 is now marked **SUPERSEDED** with an explicit warning and
a pointer to §9. The stale code was removed from §8.3. Do not restore it.

---

## 7. `EXP1-IMPL.md §10b` — already-applied fixes listed as TODO

**What happened:** `EXP1-IMPL.md §10b` was added as a list of doc fixes
required before Phase 1 completion. Several items in that list have since been
applied (outlier fraction convention documented in source, six-vs-five fixed,
`ProfilingError` duplicate removed). The section was written before those fixes
landed and was not subsequently updated.

**Current status:** The fixes listed in §10b have all been applied. The section
is now historical record. `NEXT-STEPS.md §6c` status reflects this (items
marked ✅ DONE). Do not re-apply any of them.
