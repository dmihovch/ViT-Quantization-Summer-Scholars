# Open Issues

Tracks items that still require action. Historical resolved items are in
`docs/mistakes-ledger.md`. Phase 1 is complete; remaining items are deferred
to Phase 2/3 or are follow-up improvements.

> **Last updated:** after Step 6b implementation completion.
> **Test status:** 82/91 fast tests pass (9 failures are pre-existing Phase 2/3 stubs).
> 22 slow tests require nnsight trace context.

---

## Active — Phase 1 follow-up

### 10.1 — Outlier fractions are per-batch, not global-σ — ⚠️ Known limitation

The outlier fractions in `profiling_result.json` are computed as weighted averages
of per-batch outlier rates (threshold = k·σ_batch), not fractions relative to
global σ. Per-batch σ < global σ (by the law of total variance), so the
threshold is lower and outlier fractions are overestimated by ~5–10%.

**Why this matters:** The standard practice in the quantization literature
(Bondarenko et al. 2023; Dettmers et al. 2022; Xiao et al. 2023; Wei et al. 2022)
is to report outlier fractions relative to global σ, computed in a two-pass
manner (first pass: global μ, σ; second pass: count |x − μ_global| > k·σ_global).

**Impact on Phase 2/3:** None. Phase 2 uses the correct global σ for thresholding
(the JSON stores exact global σ from the Pébay merge). Only the *reported*
outlier fractions in the JSON are approximate. Phase 3 uses moments (mean, std,
kurtosis), not outlier fractions.

**Resolution options:**
1. Add a post-hoc counting pass (`run_outlier_counting_pass`) that uses global
   μ and σ to count outliers correctly. This adds ~25 min to the full 50k run.
2. Accept the ~5–10% overestimate as a diagnostic and document the limitation.
   **Recommendation: Option 2 for now, Option 1 before publication.**

---

## Deferred — Phase 2

### 7.1 — `AblationConfig.layer_stats_path` docstring is stale

`AblationConfig.layer_stats_path` in `src/config.py` references `layer_stats.json`
(the old hooks pipeline). Phase 1 produces `profiling_result.json`.

When implementing Phase 2:
- Update the `layer_stats_path` docstring to reference `profiling_result.json`.
- Call `profiler.load_profiling_result` instead of `hooks.load_stats`.
- `attn_profile_num_images` and `attn_profile_seed` are unused (Phase 1 now
  provides dataset-wide `pre_softmax` σ directly). Emit `logger.warning` if
  they are set to non-default values.

---

## Summary

| # | Issue | Status | Blocks |
|---|-------|--------|--------|
| 10.1 | Outlier fractions are per-batch, not global-σ | ⚠️ Known limitation | Publication |
| 7.1 | `AblationConfig` filename + deprecated fields | ⚠️ Deferred | Phase 2 |