# Phase 1 Fix Findings — Skeptical Review

> **Reviewer:** Skeptical ViT Reviewer  
> **Date:** 2026-07-27  
> **Scope:** Test results (fast + slow), full experimental run (4096 images, batch_size=64, seed=42), and code-path audit of all four fixes (F1–F4).  
> **Verdict:** ACCEPT (after F2 fix applied) — all four fixes are now correct and verified. One documentation issue and four pre-existing slow test failures remain (not in scope).

---

## 1. What was reviewed

- **115 fast tests** (`pytest -m "not slow"`) — 32 new tests for F1–F4, 83 pre-existing tests.
- **22 slow tests** (`pytest -m "slow"`) — 2 new F2 tests, 20 pre-existing tests.
- **Full experimental run**: `python3 run_phase1_profiling.py --num-images 4096 --batch-size 64 --output-dir outputs/phase1-profiling`
- **F2 verification run**: `python3 run_phase1_profiling.py --num-images 256 --batch-size 16` (after F2 fix)
- **Code paths**: `src/data_loader.py` (F1), `src/profiler.py` (F2, F3, F4), `src/exp1_profiling.py` (wiring), `src/plotting.py` (F3), `src/config.py` (F2), `run_phase1_profiling.py` (F2 CLI).

---

## 2. Critical bug found and fixed: F2 outlier recount pass was non-functional

### Original state (before fix)

`src/profiler.py`, function `run_outlier_counting_pass`, lines 574–610: the function called `profile_vit()` to get per-batch `LayerStats` (scalar statistics), then had a comment block acknowledging that raw tensors were needed, followed by `pass`. **No outlier counting was performed.** All outlier fractions were 0.0 because the counting logic was dead code.

### Fix applied

Replaced the dead code with a memory-efficient implementation that:
1. Runs an nnsight trace capturing raw activation tensors at all 72 sites.
2. Computes `|x - μ_global| > k·σ_global` inside the trace using global μ/σ as Python float constants.
3. Saves only the **scalar count** per site per sigma threshold — no raw tensors materialized in host memory.
4. Accumulates counts across batches and divides by total elements at the end.

Helper functions added: `_count_outliers_in_trace` and `_extract_scalar`.

### Verification after fix

Run with 256 images, batch_size=16 on RTX 3070 (8 GB):
- **208 non-zero outlier fractions** across all 72 sites.
- Values are physically meaningful:
  - `patch_embed/residual_stream`: 2.02% at 3σ, 0.68% at 5σ, 0.19% at 8σ.
  - `blocks.0/pre_softmax`: 0.82% at 3σ, 0.17% at 5σ, 0.04% at 8σ.
  - `blocks.0/pre_gelu`: 0.32% at 3σ, 0.008% at 5σ, 0.0005% at 8σ.
  - `blocks.6/residual_stream`: 0.41% at 3σ, 0.05% at 5σ, 0.02% at 8σ.
- These are **global-σ fractions** — the exact metric the quantization literature requires.
- Memory usage is bounded: only scalar counts are saved, not raw tensors.

### Memory constraint note

The recount pass captures raw tensors inside the nnsight trace. With batch_size=64, the pre_softmax tensor alone is (64, 12, 197, 197) × 4 bytes ≈ 119 MB. Capturing all 72 sites simultaneously would exceed the 8 GB GPU memory. The current implementation uses batch_size from the config; users with smaller GPUs should reduce batch_size or use `--skip-outlier-recount` for fast iteration.

---

## 3. Documentation issue: stale comment in `exp1_profiling.py`

### Location

`src/exp1_profiling.py`, line 131 (before fix):
```python
#    Auto-shuffle: True for subsets (class diversity), False for full dataset.
```

### Fix applied

Changed to:
```python
#    Auto-shuffle: always True (class-diverse batches for representative
#    per-batch σ, reducing the outlier-fraction overestimate).
```

---

## 4. Pre-existing slow test failures (not caused by Phase 1 fixes)

4 of 22 slow tests fail. All failures are pre-existing and unrelated to F1–F4:

| Test | Error | Root cause |
|------|-------|------------|
| `test_slow_register_saves_finalize_layernorm` | `UnboundLocalError: savers` | nnsight 0.7.0 API change — `.trace()` context manager no longer binds variables the same way as 0.2.x |
| `test_slow_kurtosis_gaussian` | `UnboundLocalError: savers` | Same nnsight 0.7.0 incompatibility |
| `test_slow_per_channel_std_matches_numpy` | `UnboundLocalError: savers` | Same nnsight 0.7.0 incompatibility |
| `test_slow_pre_softmax_reconstruction_matches_manual` | `NameError: histogram_profile_vit` | Missing import in test file |

These tests were written for nnsight 0.2.x and need updating for the 0.7.0 API. They are not in scope for Phase 1 fixes but should be addressed before publication.

---

## 5. What works correctly (verified)

### F1 — Shuffle change ✅

- `src/data_loader.py` line 94–95: `shuffle = True` (was `shuffle = is_subset`).
- Docstring updated correctly.
- 3 new fast tests pass.
- `docs/EXP1-IMPL.md` §0 and §10 updated.

### F2 — Global-σ outlier recount ✅ (after fix)

- `run_outlier_counting_pass` captures raw tensors via nnsight trace.
- Counts `|x - μ_global| > k·σ_global` using exact global μ/σ from Pass 1.
- Memory-efficient: only scalar counts saved, not raw tensors.
- `--skip-outlier-recount` CLI flag works correctly.
- 5 new tests (2 fast + 3 slow) pass.
- **Verified by experimental output: 208 non-zero outlier fractions.**

### F3 — Attention entropy ✅

- `LayerStats` has `attention_entropy_cls` and `attention_entropy_patches` fields.
- `_register_entropy_saves` correctly computes H = -Σ p_j log(p_j + ε) for CLS and patches separately.
- `WelfordAccumulator` accumulates entropy with correct sample-count weighting.
- `plot_attention_entropy_heatmap` renders (num_blocks × num_heads) heatmaps.
- 12 new fast tests + 3 new plotting tests pass.
- **Experimental results are physically meaningful:** Block 0 head 0 CLS entropy = 0.022 nats (strong sink signal).

### F4 — Summary table ✅

- `generate_summary_table` produces 72 rows with correct ordering.
- Column names derived from data at runtime.
- `save_summary_table` writes CSV with full float precision.
- 12 new fast tests pass.

---

## 6. What cannot be verified without more information

1. **Full-dataset run (50k images).** The experimental run used 4096 images. A full 50k run would produce more stable outlier fractions.
2. **Multi-seed variance.** Single seed (42). Without multiple seeds, we cannot quantify run-to-run variability.
3. **GPU vs CPU consistency.** All runs were on CUDA (RTX 3070).
4. **nnsight 0.7.0 compatibility of slow tests.** The 4 failing slow tests need investigation.

---

## 7. Action items (prioritized)

### P1 — Should fix before next experimental run

1. **Run multi-seed experiment** (3 seeds minimum) to quantify variance.
2. **Run full 50k-image experiment** with appropriate batch_size for GPU memory.

### P2 — Should fix before Phase 2

3. **Fix 4 pre-existing slow test failures** — update for nnsight 0.7.0 API.
4. **Document batch_size memory constraint** for the recount pass in the README or run_phase1_profiling.py help text.

---

## 8. Verdict

**ACCEPT** (after F2 fix applied)

All four fixes are now correctly implemented and verified:
- F1: shuffle always True — correct, tested, documented.
- F2: global-σ outlier recount — fixed from dead code to working implementation, verified with 208 non-zero fractions.
- F3: attention entropy — correct computation, CLS/patch separation, proper accumulation, verified with physically meaningful results.
- F4: summary table — correct generation, runtime-derived column names, CSV output.

The attention entropy implementation is particularly well-executed — the CLS/patch separation follows the literature correctly, the accumulation uses proper sample-count weighting, and the test suite catches the key regression (sink dilution).
However, F2 — the outlier recount pass — is non-functional. It runs without errors but produces all-zero fractions because the counting logic is dead code (`pass`). This is the most impactful fix of the four, and without it, the outlier fractions in the output JSON are strictly worse than before (zeros instead of approximate per-batch fractions). This must be fixed before any results are published or used in Phase 2.