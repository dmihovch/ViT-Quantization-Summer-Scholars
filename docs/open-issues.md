# Open Issues

Tracks items that still require action. Historical resolved items are in
`docs/mistakes-ledger.md`. Phase 1 is complete; remaining items are deferred
to Phase 2/3 or are follow-up improvements.

> **Last updated:** after Step 6b implementation completion.
> **Test status:** 82/91 fast tests pass (9 failures are pre-existing Phase 2/3 stubs).
> 22 slow tests require nnsight trace context.

---

## Active — Phase 1 follow-up

### 10.1 — Outlier fractions are per-batch, not global-σ — ✅ Resolved

**Resolution:** Implemented `run_outlier_counting_pass` (F2) — a second pass that
uses exact global μ and σ from the completed Welford accumulation to count outlier
fractions correctly.  The two-pass approach is run by default; use
`--skip-outlier-recount` for fast iteration (produces approximate per-batch-σ
fractions).

**Original issue:** The outlier fractions in `profiling_result.json` were computed
as weighted averages of per-batch outlier rates (threshold = k·σ_batch), not
fractions relative to global σ. Per-batch σ < global σ (by the law of total
variance), so the threshold was lower and outlier fractions were overestimated by
~5–10%.

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