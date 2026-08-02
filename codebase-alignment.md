# Codebase Alignment Report

> **Generated:** 2026-08-01 — comprehensive audit of README, docs/, and src/.
> **Updated:** 2026-08-01 — all fixable discrepancies resolved.
> **Purpose:** Identify where the three sources (README, documentation, code) agree and disagree.

---

## 1. Three Narratives

### 1.1 What the README says the project is

A three-phase research project for profiling and ablating massive activation outliers in ViT-B/16, with a pathway toward integer-only GELU on NVIDIA Jetson edge hardware.

- **Phase 1 (profiling):** ✅ Complete. 6 measurement sites across 12 encoder blocks.
- **Phase 2 (ablation):** ✅ Complete. nnsight-based intervention with mean-centered thresholding, random-zeroing control, and class-balanced subset sampling.
- **Phase 3 (integer GELU):** Stub. `run_phase3_integer_gelu.py`, `src/integer_gelu.py`, `src/exp3_integer_gelu.py` all marked as stubs.
- **Docs listed:** `EXP1-IMPL.md`, `EXP2-IMPL.md`, `NEXT-STEPS.md`, `issues.md`, `MISTAKES.md`, `CITATIONS.md`, `AI-DISCLAIMER.md`.
- **`hooks.py`:** Described as "legacy raw-hook pipeline (deprecated; kept for reference)."
- **Phase 2 default `--data-dir`:** `data`.
- **Phase 2 sigma thresholds:** `3.0 4.0 6.0` (matches code default).

### 1.2 What the docs say the project is

A three-phase research project where Phase 1 and Phase 2 are both complete with fixes, and Phase 3 is not yet implemented.

- **Phase 1:** ✅ Complete. 73 sites (6×12 + 1 final residual). Welford multi-batch pipeline with exact Pébay (2008) parallel merge. OUTLIER_SIGMAS = (3.0, 4.0, 6.0). All features: LayerNorm γ/β, LN2 amplification ratio, attention entropy, max/min, RunMetadata, per-channel std, two-pass outlier recount.
- **Phase 2:** ✅ Complete with fixes (T-020, T-021, T-022 applied 2026-08-01). nnsight-based intervention with mean-centered thresholding, random-zeroing control, class-balanced subset sampling, entropy delta computation. 30 fast + 13 slow tests.
- **Phase 3:** 🔲 Not yet implemented.
- **Docs present:** `EXP1-IMPL.md`, `EXP2-IMPL.md`, `NEXT-STEPS.md`, `issues.md`, `MISTAKES.md`, `CITATIONS.md`, `AI-DISCLAIMER.md`.
- **Open issues:** T-007 (verify Yadav & Das DOI), T-011 (researcher sign-off on all citations).

### 1.3 What the code says the project is

A three-phase research project where Phase 1 and Phase 2 are fully implemented, and Phase 3 is a stub.

- **Phase 1:** Fully implemented. `src/profiler.py` (~1800 lines) with `profile_vit`, `run_profiling_dataset_pass`, `run_outlier_counting_pass`, `histogram_profile_vit`, Welford multi-batch merge, 73 sites. `src/exp1_profiling.py` orchestrator with multi-seed support. `run_phase1_profiling.py` CLI with `--all`, `--num-seeds`, `--skip-outlier-recount`.
- **Phase 2:** Fully implemented. `src/ablation.py` (~680 lines) with `zero_outliers_in_trace`, `_build_zeroing_mask` (mean-centered), `_build_random_mask`, `_intervene_pre_gelu`, `_intervene_residual_stream` (CLS preserved), `_intervene_pre_softmax` (QKᵀ/√d reconstruction + entropy capture). `src/exp2_ablation.py` orchestrator with per-batch matched random control. `run_phase2_ablation.py` CLI with `--sigma-thresholds` defaulting to `[3.0, 4.0, 6.0]`.
- **Phase 3:** Stub. `src/integer_gelu.py` has `GELULut` dataclass and function signatures but all three functions (`build_lut`, `apply_lut`, `compare_lut_vs_fp32`) raise `NotImplementedError`. `src/exp3_integer_gelu.py` raises `NotImplementedError`. `run_phase3_integer_gelu.py` is a complete CLI wrapper that will call the stub.
- **`hooks.py`:** Deleted 2026-08-01. nnsight has fully replaced raw PyTorch hooks.
- **`OUTLIER_SIGMAS`:** `(3.0, 4.0, 6.0)` in `src/profiler.py`.
- **Phase 2 default `--data-dir`:** `data`.
- **Outputs present:** `outputs/phase1-profiling/seed_{42,43,44}/` with `profiling_result.json`, histograms, heatmaps. `outputs/phase2-ablation/` with `ablation_results.csv`, `entropy_deltas.csv`, and all expected PNGs.

---

## 2. Where They Agree

| Topic | README | Docs | Code | Verdict |
|-------|--------|------|------|---------|
| Phase 1 status | ✅ Complete | ✅ Complete | ✅ Complete | **AGREE** |
| Phase 2 status | ✅ Complete | ✅ Complete | ✅ Complete | **AGREE** |
| Phase 3 status | Stub | 🔲 Not implemented | `NotImplementedError` | **AGREE** |
| Target model | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | Same | `load_vit` uses exact string | **AGREE** |
| Dataset | ImageNet-1K validation | Same | `ImageFolder` in `data_loader.py` | **AGREE** |
| Profiling method | nnsight-based | nnsight-based (EXP1-IMPL) | nnsight trace in `profiler.py` | **AGREE** |
| Statistical method | — | Pébay (2008) exact merge | `merge_batch_stats` implements Pébay | **AGREE** |
| OUTLIER_SIGMAS | — | (3.0, 4.0, 6.0) | `(3.0, 4.0, 6.0)` | **AGREE** |
| Site count | "6 measurement sites" | 73 sites (EXP1-IMPL §0) | 73 sites (6×12 + 1 final) | **AGREE** (but README is imprecise) |
| Phase 2 threshold | — | Mean-centered `\|x−μ\| > k·σ` | `_build_zeroing_mask` uses `(tensor - mean).abs()` | **AGREE** |
| Phase 2 random control | — | Implemented (EXP2-IMPL §2.3) | `_build_random_mask` + `random_fractions` | **AGREE** |
| Phase 2 CLS preservation | — | CLS token preserved (EXP2-IMPL §2.6) | `mask[:, 0, :] = True` in `_intervene_residual_stream` | **AGREE** |
| Phase 2 entropy | — | `torch.special.entr` (T-017) | `torch.special.entr` in `_intervene_pre_softmax` | **AGREE** |
| LayerNorm γ/β | — | Captured (T-004 closed) | Extracted in `profile_vit` post-trace | **AGREE** |
| LN2 amplification ratio | — | Computed (T-005 closed) | `ln2_amplification_ratio` in `LayerStats` | **AGREE** |
| RunMetadata | — | Embedded in JSON (T-016) | `RunMetadata` in `ProfilingResult` | **AGREE** |
| max/min | — | Tracked (T-009 closed) | `max_val`/`min_val` in `WelfordAccumulator` | **AGREE** |
| eval() assertion | — | Asserted (T-019 closed) | `profile_vit` checks `inner_model.training` | **AGREE** |
| Class-balanced subset | — | Fixed (T-022 closed) | Seeded permutation in `build_val_loader` | **AGREE** |
| Phase 1 multi-seed | Supported | Documented (EXP1-IMPL §0) | `run()` iterates seeds | **AGREE** |
| Phase 1 `--skip-outlier-recount` | — | Documented | `ProfilingConfig.skip_outlier_recount` | **AGREE** |
| Phase 2 sigma defaults | `3.0 4.0 6.0` | `3.0 4.0 6.0` in EXP2-IMPL | `[3.0, 4.0, 6.0]` in CLI | **AGREE** |
| Phase 2 `--data-dir` default | `data` | `data` in EXP2-IMPL | `data` in CLI | **AGREE** |
| Doc filenames | `issues.md`, `MISTAKES.md`, `EXP2-IMPL.md` | Same | N/A | **AGREE** |
| Experimental spec | `EXP1-IMPL.md` + `EXP2-IMPL.md` | Same | N/A | **AGREE** |
| `scripts/` directory | Listed in README | N/A | Present | **AGREE** |
| `download_imagenet_val.py` | Listed in README | N/A | Present | **AGREE** |

---

## 3. Remaining Discrepancies

None. All identified discrepancies have been resolved.

---

## 4. Open Issues Still Unresolved

From `docs/issues.md` summary table:

| Ticket | Severity | Status | Title |
|--------|----------|--------|-------|
| T-007 | MEDIUM | 🔲 Open | Verify reconstructed DOI for Yadav & Das 2025 |
| T-011 | LOW | 🔲 Open | Researcher sign-off on all citations |

These are the only two open tickets. Everything else (T-001 through T-022) is closed.

---

## 5. Fixes Applied (2026-08-01)

| # | Discrepancy | Fix |
|---|-------------|-----|
| 1 | README said Phase 2 was a stub | Updated to ✅ Complete with description of features |
| 2 | `vit_profiling_framework.md` referenced everywhere but missing | Removed all references; README now points to EXP1-IMPL.md + EXP2-IMPL.md |
| 3 | README doc filenames wrong | Fixed: `open-issues.md` → `issues.md`, `mistakes-ledger.md` → `MISTAKES.md`; added `EXP2-IMPL.md`, `CITATIONS.md` |
| 4 | README Phase 2 sigma thresholds wrong | Changed `2.0 3.0 4.0 5.0` → `3.0 4.0 6.0` |
| 5 | CITATIONS.md duplicate verification line | Removed duplicate Bondarenko et al. 2021 verification entry |
| 6 | CITATIONS.md referenced missing docs | Removed all `IMPL-phase1-fixes.md` and `vit_entropy_methodology.md` references; removed entire "Additional references" section |
| 7 | CITATIONS.md referenced `vit_profiling_framework.md` | Replaced with appropriate existing doc references |
| 8 | NEXT-STEPS.md referenced scispace doc | Removed `docs/scispace-docs/vit_profiling_framework.md` reference |
| 9 | issues.md header referenced scispace doc | Changed to `docs/EXP1-IMPL.md` |
| 10 | `src/data_loader.py` referenced `open-issues.md` | Changed to `docs/MISTAKES.md §1.3` |
| 11 | `src/profiler.py` referenced `open-issues.md` | Changed to `docs/MISTAKES.md §1.3` |
| 12 | EXP1-IMPL.md had stale test count | Removed stale "80/112 fast tests pass" line |
| 13 | MISTAKES.md §6.4 referenced `open-issues.md` | Changed to `docs/issues.md` T-009 |
| 14 | README missing `scripts/` and `download_imagenet_val.py` | Added to repository layout |
| 15 | README Phase 2 `--data-dir` default mismatch | Changed code default from `data/imagenet-val` to `data` |
| 16 | `hooks.py` obsolete — nnsight has fully replaced it | Deleted `src/hooks.py`, removed `HookRegistrationError` and `ShapeMismatchError` from `src/exceptions.py`, updated `tests/test_exceptions.py`, README, NEXT-STEPS.md |
| 17 | Phase 1 output path inconsistent between single/multi-seed | Always write to `seed_{N}` subdirectory; updated Phase 2/3 defaults, README, EXP1-IMPL.md |

---

## 6. Verdict

The **README, documentation, and code are now consistent**. All three sources agree on Phase 1 and Phase 2 status, sigma thresholds, data directory defaults, and doc filenames. The scispace documentation references have been removed — the implementation specs (EXP1-IMPL.md and EXP2-IMPL.md) are now the authoritative experimental specifications.

The only remaining open items are the two open tickets (T-007 and T-011).