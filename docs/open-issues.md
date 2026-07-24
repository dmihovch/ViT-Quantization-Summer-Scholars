# Open Issues

Tracks decisions made, bugs fixed, and items that still require action before
Phase 1 is complete. Ordered by category, then by resolution status.

> **Last updated:** after the Step 4b-i review and fix session.
> **Test status at time of writing:** 48/48 fast tests pass. 2 slow tests
> updated. 0 new regressions.

---

## 1. Statistical correctness

### 1.1 — ddof convention — ✅ RESOLVED

**Issue:** `_register_stat_saves` called `t.std()` which uses PyTorch's default
of `correction=1` (Bessel's correction, sample std). This introduced a
systematic negative bias of `(n-1)/n` per batch into every variance value that
would be fed into `merge_batch_stats`. The Welford accumulator tracks population
variance, so the mismatch was a silent correctness bug.

**Resolution:** All std computations in `profiler.py` now use
`t.std(correction=0)` (population std, ddof=0). This is correct because we are
profiling a fully-observed finite set of activation values — we are not
estimating an unobserved population parameter. Bessel's correction has no
statistical justification in this context.

**Files changed:** `src/profiler.py` (`_register_stat_saves`, outlier threshold
computations). Documented in `EXP1-IMPL.md` Section 2.1 as a non-negotiable
convention.

---

### 1.2 — Kurtosis: approximation replaced with exact formula — ✅ RESOLVED

**Issue:** The original plan accumulated kurtosis by summing `(x − batch_mean)⁴`
per batch, where `batch_mean` is the local batch mean rather than the global
mean. This is an approximation with no bounded error guarantee — its accuracy
degrades when per-batch means vary significantly, and it can never be published
without a caveat that we cannot quantify.

**Resolution:** Replaced with the exact Pébay (2008) parallel higher-moments
formula. `WelfordAccumulator` now tracks `M3 = Σ(x−μ)³` and `M4 = Σ(x−μ)⁴`
as exact running sums. `merge_batch_stats` implements the full Pébay merge for
M2, M3, and M4. Kurtosis from `finalize_accumulator` is exact.

**Implementation cost:** One extra proxy save per site per block (M3). Two
extra float fields on `WelfordAccumulator` (M3, M4). ~20 lines in
`merge_batch_stats`.

**Reference:** Pébay (2008), *Formulas for Robust, One-Pass Parallel
Computation of Covariances and Arbitrary-Order Statistical Moments*, Sandia
SAND2008-6212, Eq. 3.1–3.4.

**Files changed:** `src/profiler.py` (`LayerStats`, `_StatsSavers`,
`_register_stat_saves` saves M3; `EXP1-IMPL.md` Section 2.2 has the full
formula). `WelfordAccumulator` and `merge_batch_stats` in Step 4b-ii spec.

**Remaining:** `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`
are specified in `EXP1-IMPL.md` Section 3 but not yet written. See Issue 5.1.

---

### 1.3 — `n_samples` absent from `LayerStats` — ✅ RESOLVED

**Issue:** `LayerStats` stored no element count, making it impossible to verify
whether outlier fractions were computed over 1 000 or 1 000 000 elements, or
to detect if a batch was accidentally processed twice.

**Resolution:** `LayerStats` gains `n_samples: int = 0`. `_finalize_stats`
populates it from `_StatsSavers.n_samples`, which is set from the `n_samples`
argument passed to `_register_stat_saves`. `finalize_accumulator` (Step 4b-ii)
will set `n_samples = acc.n` (total elements across all batches).

**Files changed:** `src/profiler.py`, `tests/test_profiler.py` (two slow tests
now assert `stats.n_samples`).

---

## 2. Correctness of element-count derivation

### 2.1 — Wrong N: image height used instead of token count — ✅ RESOLVED

**Issue:** The draft `run_profiling_dataset_pass` derived the token sequence
length N as `batch_result.batch_shape[2]`, which is the image height (224),
not the number of tokens (197 for ViT-B/16 on 224×224 inputs). This would have
produced `batch_n` values that are ~14% too large, corrupting every Welford
merge.

**Resolution:** N is now derived as `inner_model.patch_embed.num_patches + 1`
(number of patch embeddings plus the CLS token). For ViT-B/16 on 224×224:
`(224/16)² + 1 = 196 + 1 = 197`. This is computed once before the batch loop
in `run_profiling_dataset_pass` and asserted in the `_site_n` docstring.

**Files changed:** `src/profiler.py` (`profile_vit` now also sets `N` this way
before the block loop). `EXP1-IMPL.md` `_site_n` docstring.

---

### 2.2 — `_site_n` defined as a closure inside the batch loop — ✅ RESOLVED

**Issue:** The draft defined `_site_n` as a nested function inside the `for
batch_idx` loop, capturing `B`, `N`, `D`, etc. by reference. Python closures
capture variables by reference, not value — if the function were ever called
after the loop variable was rebound it would silently use the wrong values.
It was also re-defined on every iteration.

**Resolution:** `_site_n` is now a **top-level module function** that receives
all dimensions as explicit arguments. Architecture constants (`N, D, num_heads,
D_mlp`) are extracted from the model once before the loop.

**Files changed:** `EXP1-IMPL.md` Section 3.1 (spec). Applied in `profile_vit`
constant extraction in `src/profiler.py`.

---

### 2.3 — Architecture constants re-accessed per batch — ✅ RESOLVED

**Issue:** The draft accessed `inner_model.blocks[0].attn.num_heads`,
`inner_model.embed_dim`, and `inner_model.blocks[0].mlp.fc1.out_features`
inside the batch loop. These are model constants that never change.

**Resolution:** All four constants (`N`, `D`, `num_heads`, `D_mlp`) are
extracted once before the loop in `run_profiling_dataset_pass` and in
`profile_vit`.

**Files changed:** `src/profiler.py`, `EXP1-IMPL.md` Section 3.5.

---

## 3. `ProfilingResult.batch_shape` metadata inaccuracy

### 3.1 — batch_shape hardcoded, last batch may be smaller — ⚠️ NEEDS ACTION

**Issue:** In the `exp1_profiling.run()` plan, `batch_shape` is hardcoded as
`(config.batch_size, 3, 224, 224)`. When `num_images % batch_size != 0`, the
final batch is smaller. The stored metadata would misrepresent the actual batch
size.

**Status:** Specified as resolved in `EXP1-IMPL.md` Section 10: peek at the
first actual batch from the loader and use `tuple(first_images.shape)` instead.
However this code is in `exp1_profiling.py` which is still a stub — the fix
needs to be applied when Step 5 is implemented.

**Action required:** Implement the first-batch peek in `exp1_profiling.run()`
per `EXP1-IMPL.md` Section 10. Note that two `next(iter(loader))` calls are
needed if `--spot-batch` also peeks at the first batch — ensure the loader is
reset or shared correctly.

---

## 4. Test infrastructure

### 4.1 — Broken slow test: TensorDataset label unpacking — ✅ RESOLVED

**Issue:** The original slow test spec used `TensorDataset(torch.randn(4, 3,
224, 224))` (no labels), then tried to unpack `for images, _ in loader`. This
raises `ValueError: too many values to unpack` at runtime.

**Resolution:** The slow tests in `EXP1-IMPL.md` Section 4.2 now use:
```python
images = torch.randn(4, 3, 224, 224)
labels = torch.zeros(4, dtype=torch.long)
dataset = TensorDataset(images, labels)
```
The `(images, labels)` unpack now works correctly.

---

### 4.2 — Slow tests updated for new `_register_stat_saves` signature — ✅ RESOLVED

**Issue:** Two existing slow tests called `_register_stat_saves(proxy, site_id)`
with the old 2-argument signature. Adding `n_samples` as a required third
argument would have broken them silently (wrong call, no TypeError at import
time because the function is called inside a trace context).

**Resolution:** Both tests in `tests/test_profiler.py` are updated to pass
`n_samples` and now also assert `stats.n_samples == n_samples`.

---

### 4.3 — `_canned_result()` missing new `LayerStats` fields — ✅ RESOLVED

**Issue:** The `_canned_result()` helper in `test_profiler.py` constructs
`LayerStats` by keyword. Adding `m3` and `n_samples` as new fields (with
defaults) means existing constructions still work, but the helper's explicit
construction was stale and misleading.

**Resolution:** `_canned_result()` updated to pass `m3=0.0, n_samples=0`
explicitly on all `LayerStats` constructions.

---

## 5. Pending implementation (Step 4b-ii)

All items below are **specified** in `EXP1-IMPL.md` but not yet written into
production code. They must be completed before `exp1_profiling.py` can run.

### 5.1 — `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`, `_site_n`, `run_profiling_dataset_pass` — ✅ DONE

**Implemented in `src/profiler.py`.** Five additions plus per-channel std support
(Step 4b-iii). 9 new fast tests + 4 new slow tests in `test_profiler.py`. All
fast tests pass. Slow tests require PyTorch 2.2.x + nnsight 0.2.21 (known
incompatibility with PyTorch 2.12).

**Key design constraints preserved:**
- `WelfordAccumulator` tracks `M3` and `M4` for exact kurtosis.
- `merge_batch_stats` implements the Pébay (2008) Eq. 3.1–3.4.
- `_site_n` is a top-level function; N comes from `patch_embed.num_patches + 1`.
- Architecture constants extracted before the batch loop.
- `run_profiling_dataset_pass` handles the empty-loader edge case with a
  `RuntimeError`.

---

### 5.2 — `per_channel_std` in `profiler.LayerStats` — ✅ DONE

**Implemented.** `LayerStats` gains `per_channel_std`, `per_channel_sum`,
`per_channel_sum_sq` fields. `_register_stat_saves` accepts
`track_per_channel=True` and saves per-channel sum and sum-of-squares proxies.
`_finalize_stats` computes `per_channel_std` from the sums. `profile_vit` passes
`track_per_channel=True` for `pre_gelu`, `post_layernorm_1`, `post_layernorm_2`.
`WelfordAccumulator` stores `per_channel_sum` and `per_channel_sum_sq` for exact
cross-batch merging. `merge_batch_stats` accumulates them directly (no merge
formula needed — sums are additive). `finalize_accumulator` computes
`per_channel_std` from the accumulated sums.

**Unblocks:** `plot_per_channel_std_heatmap` and the heatmap deliverable.

---

## 6. Histogram quality

### 6.1 — Reconstructed Gaussian histograms mask heavy tails — ⚠️ KNOWN LIMITATION

**Issue:** The Welford pipeline correctly discards raw activation tensors to
remain memory-efficient. Histograms in `_plot_histograms` are therefore drawn
from synthetic `N(mean, std²)` samples, not real activations. The entire
purpose of Phase 1 is to characterise heavy-tailed non-Gaussian distributions.
A Gaussian reconstruction cannot show the tails, regardless of how the kurtosis
value is annotated on the chart.

The spec (`vit_profiling_framework.md`) requires "log-scale histograms per site
showing heavy-tailed distributions." A Gaussian reconstruction does not satisfy
this requirement.

**Current mitigation:** Every histogram title contains the literal string
`[reconstructed N(μ,σ²)]`. This is mandatory — it must never be removed.

**Full resolution requires `--spot-batch` (Issue 6.2).**

---

### 6.2 — `--spot-batch` real-activation histograms — 🔲 PARTIALLY SPECIFIED

**Issue:** `--spot-batch` is specified in `EXP1-IMPL.md` Section 9 and the
`run_phase1_profiling.py` argument added. However the implementation note
explicitly marks the function body as a placeholder: `profile_vit` discards
raw activation tensors, so `_run_spot_batch` cannot currently produce real
data.

**What is missing:**
- A `spot_profile_vit(wrapped_model, batch, max_samples_per_site)` function in
  `profiler.py` that, for each site, saves a capped raw sample
  (`t[:max_samples].save()` after `reshape(-1)`) rather than only scalar stats.
- The `--spot-batch` flag added to `run_phase1_profiling.py` and a
  `spot_batch: bool = False` field added to `ProfilingConfig`.
- `_run_spot_batch` in `exp1_profiling.py` calling `spot_profile_vit` and
  passing the resulting raw arrays directly to `plot_activation_histogram`.

**Priority:** This is a Phase 1 deliverable blocker if the spec's histogram
requirement is taken seriously. Recommend implementing `spot_profile_vit`
alongside the Step 4b-ii Welford additions, since it requires the same
nnsight trace familiarity.

---

## 7. Phase 2 config migration

### 7.1 — `AblationConfig.layer_stats_path` points to old filename — ⚠️ NEEDS ACTION

**Issue:** `AblationConfig.layer_stats_path` in `src/config.py` was written to
expect `layer_stats.json` (from `hooks.save_stats`). Phase 1 now writes
`profiling_result.json` (from `profiler.save_profiling_result`).

**Resolution (not yet applied):** Update the `layer_stats_path` docstring in
`AblationConfig` to state it expects `profiling_result.json`. Update
`exp2_ablation.py` (when implemented) to call `profiler.load_profiling_result`
instead of `hooks.load_stats`.

**Also:** `attn_profile_num_images` and `attn_profile_seed` in `AblationConfig`
are now unused — Phase 1 produces dataset-wide `pre_softmax` std directly.
Keep the fields for backwards compatibility but emit `logger.warning` if they
are set to non-default values.

---

## 8. `exceptions.py` duplicate class

### 8.1 — `ProfilingError` defined twice — ⚠️ MINOR, NEEDS CLEANUP

**Issue:** `src/exceptions.py` defines `ProfilingError` twice (lines 42 and 51)
with slightly different docstrings. The second definition silently shadows the
first. Python will use the second definition; the first is dead code.

**Resolution:** Remove the duplicate. Keep the second definition (it has the
more complete docstring). Not urgent but should be cleaned up before the
codebase is shared.

---

## Summary table

| # | Issue | Status | Blocks |
|---|-------|--------|--------|
| 1.1 | ddof convention (population vs sample std) | ✅ Resolved | — |
| 1.2 | Kurtosis approximation → exact Pébay formula | ✅ Resolved | — |
| 1.3 | `n_samples` absent from `LayerStats` | ✅ Resolved | — |
| 2.1 | Wrong N: image height vs token count | ✅ Resolved | — |
| 2.2 | `_site_n` closure bug | ✅ Resolved | — |
| 2.3 | Architecture constants re-accessed per batch | ✅ Resolved | — |
| 3.1 | `batch_shape` hardcoded (last batch smaller) | ✅ Done | — |
| 4.1 | TensorDataset label unpack bug in slow tests | ✅ Resolved | — |
| 4.2 | Slow tests broken by new `n_samples` arg | ✅ Resolved | — |
| 4.3 | `_canned_result()` stale field construction | ✅ Resolved | — |
| 5.1 | Welford classes not yet written | ✅ Done | — |
| 5.2 | `per_channel_std` gap in `profiler.LayerStats` | ✅ Done | — |
| 6.1 | Reconstructed Gaussian histograms mask tails | ⚠️ Known limitation | Publication |
| 6.2 | `--spot-batch` real-activation histograms | 🔲 Partially spec'd | Histogram deliverable |
| 7.1 | `AblationConfig` filename + deprecated fields | ⚠️ Spec'd, not coded | Step 8 |
| 8.1 | `ProfilingError` defined twice in exceptions.py | ⚠️ Minor cleanup | — |
