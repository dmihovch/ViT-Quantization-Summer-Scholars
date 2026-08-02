# Codebase Alignment Report

> **Generated:** 2026-08-01 — comprehensive audit of README, docs/, and src/.
> **Purpose:** Identify where the three sources (README, documentation, code) agree and disagree.

---

## 1. Three Narratives

### 1.1 What the README says the project is

A three-phase research project for profiling and ablating massive activation outliers in ViT-B/16, with a pathway toward integer-only GELU on NVIDIA Jetson edge hardware.

- **Phase 1 (profiling):** ✅ Complete. 6 measurement sites across 12 encoder blocks.
- **Phase 2 (ablation):** Stub. `run_phase2_ablation.py`, `src/ablation.py`, `src/exp2_ablation.py` all marked as stubs.
- **Phase 3 (integer GELU):** Stub. `run_phase3_integer_gelu.py`, `src/integer_gelu.py`, `src/exp3_integer_gelu.py` all marked as stubs.
- **Docs listed:** `EXP1-IMPL.md`, `vit_profiling_framework.md`, `NEXT-STEPS.md`, `open-issues.md`, `mistakes-ledger.md`, `AI-DISCLAIMER.md`.
- **`hooks.py`:** Described as "legacy, LayerStats deleted 2026-07-30."
- **Phase 2 default `--data-dir`:** `data`.

### 1.2 What the docs say the project is

A three-phase research project where Phase 1 and Phase 2 are both complete with fixes, and Phase 3 is not yet implemented.

- **Phase 1:** ✅ Complete. 73 sites (6×12 + 1 final residual). Welford multi-batch pipeline with exact Pébay (2008) parallel merge. OUTLIER_SIGMAS = (3.0, 4.0, 6.0). All features: LayerNorm γ/β, LN2 amplification ratio, attention entropy, max/min, RunMetadata, per-channel std, two-pass outlier recount.
- **Phase 2:** ✅ Complete with fixes (T-020, T-021, T-022 applied 2026-08-01). nnsight-based intervention with mean-centered thresholding, random-zeroing control, class-balanced subset sampling, entropy delta computation. 30 fast + 13 slow tests.
- **Phase 3:** 🔲 Not yet implemented.
- **Docs present:** `EXP1-IMPL.md`, `EXP2-IMPL.md`, `NEXT-STEPS.md`, `issues.md`, `MISTAKES.md`, `CITATIONS.md`, `AI-DISCLAIMER.md`.
- **Docs referenced but missing:** `vit_profiling_framework.md` (referenced by README, CITATIONS.md, MISTAKES.md, issues.md, NEXT-STEPS.md, EXP1-IMPL.md — all point to `docs/scispace-docs/vit_profiling_framework.md` which does not exist in the repo). Also missing: `docs/IMPL-phase1-fixes.md`, `docs/vit_entropy_methodology.md` (referenced by CITATIONS.md).
- **Open issues:** T-007 (verify Yadav & Das DOI), T-011 (researcher sign-off on all citations).

### 1.3 What the code says the project is

A three-phase research project where Phase 1 and Phase 2 are fully implemented, and Phase 3 is a stub.

- **Phase 1:** Fully implemented. `src/profiler.py` (~1800 lines) with `profile_vit`, `run_profiling_dataset_pass`, `run_outlier_counting_pass`, `histogram_profile_vit`, Welford multi-batch merge, 73 sites. `src/exp1_profiling.py` orchestrator with multi-seed support. `run_phase1_profiling.py` CLI with `--all`, `--num-seeds`, `--skip-outlier-recount`.
- **Phase 2:** Fully implemented. `src/ablation.py` (~680 lines) with `zero_outliers_in_trace`, `_build_zeroing_mask` (mean-centered), `_build_random_mask`, `_intervene_pre_gelu`, `_intervene_residual_stream` (CLS preserved), `_intervene_pre_softmax` (QKᵀ/√d reconstruction + entropy capture). `src/exp2_ablation.py` orchestrator with per-batch matched random control. `run_phase2_ablation.py` CLI with `--sigma-thresholds` defaulting to `[3.0, 4.0, 6.0]`.
- **Phase 3:** Stub. `src/integer_gelu.py` has `GELULut` dataclass and function signatures but all three functions (`build_lut`, `apply_lut`, `compare_lut_vs_fp32`) raise `NotImplementedError`. `src/exp3_integer_gelu.py` raises `NotImplementedError`. `run_phase3_integer_gelu.py` is a complete CLI wrapper that will call the stub.
- **`hooks.py`:** Still exists as a full module (~558 lines) with `HookHandle`, `_SiteAccumulator`, `register_profiling_hooks`, `remove_hooks`, `save_stats`, `load_stats`. `LayerStats` was deleted but the rest of the legacy pipeline remains.
- **`OUTLIER_SIGMAS`:** `(3.0, 4.0, 6.0)` in `src/profiler.py`.
- **Phase 2 default `--data-dir`:** `data`.
- **Outputs present:** `outputs/phase1-profiling/seed_{42,43,44}/` with `profiling_result.json`, histograms, heatmaps. `outputs/phase2-ablation/` with `ablation_results.csv`, `entropy_deltas.csv`, and all expected PNGs.

---

## 2. Where They Agree

| Topic | README | Docs | Code | Verdict |
|-------|--------|------|------|---------|
| Phase 1 status | ✅ Complete | ✅ Complete | ✅ Complete | **AGREE** |
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
| Phase 2 sigma defaults | `2.0 3.0 4.0 5.0` in README | `3.0 4.0 6.0` in EXP2-IMPL | `[3.0, 4.0, 6.0]` in CLI | **CODE & DOCS AGREE; README DISAGREES** |

---

## 3. Where They Disagree

### 3.1 CRITICAL: README says Phase 2 is a stub — code and docs say it's complete

**README:**
```
├── run_phase2_ablation.py       # Phase 2 entry point (stub)
├── src/ablation.py              # outlier zeroing, % zeroed, AblationResult (stub)
├── src/exp2_ablation.py         # Phase 2 orchestrator (stub)
```

**Reality:** `src/ablation.py` is ~680 lines with 11 functions, `src/exp2_ablation.py` is ~374 lines with a full orchestrator including per-batch matched random control, `run_phase2_ablation.py` is a complete CLI. Phase 2 has 30 fast + 13 slow tests. Outputs exist in `outputs/phase2-ablation/`.

**Impact:** Anyone reading only the README would think Phase 2 hasn't been started. This is the single biggest documentation-code mismatch in the project.

### 3.2 CRITICAL: `docs/vit_profiling_framework.md` does not exist

**README** references it as the experimental spec:
```
**Experimental spec:** [`docs/vit_profiling_framework.md`](docs/vit_profiling_framework.md)
```

**Multiple docs reference it** as `docs/scispace-docs/vit_profiling_framework.md`:
- `CITATIONS.md` — Pébay 2008, Li et al. 2023, Sun et al. 2024, I-ViT entries
- `MISTAKES.md` — §6.2, §9.1, §10.1, §11.1, §12.1
- `issues.md` — T-002, T-004, T-006, T-012
- `NEXT-STEPS.md` — §Before Step 9
- `EXP1-IMPL.md` — §2.3 (references it for sigma thresholds)

**Reality:** Neither `docs/vit_profiling_framework.md` nor `docs/scispace-docs/vit_profiling_framework.md` exists in the repository. The `docs/` directory contains only: `AI-DISCLAIMER.md`, `CITATIONS.md`, `EXP1-IMPL.md`, `EXP2-IMPL.md`, `MISTAKES.md`, `NEXT-STEPS.md`, `issues.md`.

**Impact:** The "source of truth" experimental spec is missing. All cross-references to it are dead links. The README's claim that this is the authoritative spec is misleading.

### 3.3 HIGH: README doc filenames don't match actual files

| README name | Actual name | Status |
|-------------|-------------|--------|
| `docs/open-issues.md` | `docs/issues.md` | Renamed |
| `docs/mistakes-ledger.md` | `docs/MISTAKES.md` | Renamed |
| `docs/vit_profiling_framework.md` | **MISSING** | Does not exist |
| `docs/EXP2-IMPL.md` | Not listed in README | Exists but omitted |

### 3.4 HIGH: README Phase 2 default sigma thresholds are wrong

**README:**
```sh
python run_phase2_ablation.py --sigma-thresholds 2.0 3.0 4.0 5.0
```

**Code** (`run_phase2_ablation.py` L63):
```python
default=[3.0, 4.0, 6.0],
```

**Docs** (EXP2-IMPL.md §0):
```bash
--sigma-thresholds 3.0 6.0
```

The README suggests `2.0 3.0 4.0 5.0` — none of these match the actual default `[3.0, 4.0, 6.0]` or the literature-standard thresholds.

### 3.5 MEDIUM: Missing docs referenced by CITATIONS.md

`CITATIONS.md` references these files that don't exist:
- `docs/IMPL-phase1-fixes.md` — referenced by Maisonnave et al. 2025, Mali 2025, Lee & Kim 2025, Yadav & Das 2025 entries
- `docs/vit_entropy_methodology.md` — referenced by Maisonnave et al. 2025, Mali 2025, Lee & Kim 2025, Yadav & Das 2025 entries, and has its own §Additional references section

### 3.6 MEDIUM: CITATIONS.md has a duplicate verification line

Lines 123-124 for Bondarenko et al. 2021:
```
- **Verification:** ⚠️ arXiv preprint (not peer-reviewed).
- **☐ Researcher sign-off: Not yet reviewed**
- **Verification:** ✅ arXiv:2109.12948. Preprint (not peer-reviewed).
```
Two `Verification` lines with contradictory symbols (⚠️ vs ✅).

### 3.7 MEDIUM: README says hooks.py "LayerStats deleted" but file still exists

**README:**
```
├── hooks.py  # forward hook machinery (legacy, LayerStats deleted 2026-07-30)
```

**Reality:** `src/hooks.py` is 558 lines with `HookHandle`, `_SiteAccumulator`, `register_profiling_hooks`, `remove_hooks`, `save_stats`, `load_stats`. Only `LayerStats` was deleted — the rest of the legacy pipeline remains. The README implies the file is mostly dead code, but it still contains functional (if deprecated) profiling machinery.

### 3.8 MEDIUM: EXP1-IMPL.md test count may be stale

**EXP1-IMPL.md** L4:
```
80/112 fast tests pass (32 slow tests require nnsight trace context).
```

This count hasn't been updated since many tests were added (γ/β, entropy, LN2 ratio, max/min, summary table, outlier recount, etc.). The test file now has ~137 test functions. The actual pass count is unverified.

### 3.9 LOW: Phase 2 `--layer-stats` default path assumes single-seed layout

**Code** (`run_phase2_ablation.py` L70):
```python
default=Path("outputs/phase1-profiling/profiling_result.json"),
```

But with `--num-seeds 3`, the actual path would be `outputs/phase1-profiling/seed_42/profiling_result.json`. The default only works for single-seed runs.

### 3.10 LOW: `scripts/` directory not documented in README

The `scripts/` directory exists with three utility scripts (`regenerate_plots.py`, `smoke_test_nnsight_intervention.py`, `verify_pre_softmax_fidelity.py`) but is not mentioned anywhere in the README's repository layout.

### 3.11 LOW: `download_imagenet_val.py` not in README layout

The file exists at the project root but is not listed in the README's repository layout section.

---

## 4. Open Issues Still Unresolved

From `docs/issues.md` summary table:

| Ticket | Severity | Status | Title |
|--------|----------|--------|-------|
| T-007 | MEDIUM | 🔲 Open | Verify reconstructed DOI for Yadav & Das 2025 |
| T-011 | LOW | 🔲 Open | Researcher sign-off on all citations |

These are the only two open tickets. Everything else (T-001 through T-022) is closed.

---

## 5. Summary of Required Fixes

### Critical (fix immediately)

1. **Update README Phase 2 status** — Change all "(stub)" markers to "✅ Complete" for `run_phase2_ablation.py`, `src/ablation.py`, `src/exp2_ablation.py`. Update the Phase 2 description to mention mean-centered thresholding, random-zeroing control, and class-balanced sampling.

2. **Locate or recreate `vit_profiling_framework.md`** — This is referenced by 6+ files as the authoritative experimental spec. Either restore it from version control or acknowledge its removal and update all cross-references.

### High (fix soon)

3. **Fix README doc filenames** — Change `open-issues.md` → `issues.md`, `mistakes-ledger.md` → `MISTAKES.md`. Add `EXP2-IMPL.md` to the docs listing. Remove or fix the `vit_profiling_framework.md` reference.

4. **Fix README Phase 2 sigma thresholds** — Change `2.0 3.0 4.0 5.0` to `3.0 4.0 6.0` to match the code default and literature standards.

### Medium (fix when convenient)

5. **Fix CITATIONS.md duplicate verification line** — Remove the duplicate `Verification` entry for Bondarenko et al. 2021 (line 124).

6. **Update or remove stale CITATIONS.md references** — `docs/IMPL-phase1-fixes.md` and `docs/vit_entropy_methodology.md` don't exist. Either create them or remove the references.

7. **Clarify hooks.py status in README** — The file still contains functional legacy code. Either delete it entirely or document what remains and why.

8. **Update EXP1-IMPL.md test count** — The "80/112 fast tests pass" line is stale.

### Low (nice to have)

9. **Fix Phase 2 `--layer-stats` default** — Consider documenting that multi-seed runs need an explicit path.

10. **Add `scripts/` to README layout** — Document the utility scripts.

11. **Add `download_imagenet_val.py` to README layout** — It's a project file that should be documented.

---

## 6. Verdict

The **code and documentation are largely consistent** for Phase 1 and Phase 2. The implementation matches what EXP1-IMPL.md and EXP2-IMPL.md describe. The issue tracker (issues.md) and mistakes ledger (MISTAKES.md) accurately reflect the current state of fixes.

The **README is significantly out of date**. It describes Phase 2 as a stub when it's fully implemented, references files that don't exist (`vit_profiling_framework.md`), uses wrong filenames for existing docs, and has incorrect default values for Phase 2 commands. The README appears to have been written early in the project and not updated as Phase 2 was completed.

The **missing `vit_profiling_framework.md`** is the most concerning finding. Six different documentation files reference it as the authoritative experimental specification, but it doesn't exist in the repository. This needs to be resolved before any external reader can understand the project's research design.