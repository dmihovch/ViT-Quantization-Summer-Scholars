# Next Steps: Implementation Roadmap

> **Last updated:** 2026-08-02 — Phase 1 per-channel mean added + Phase 2 per-channel ablation complete.

---

## Current State

- **Phase 1 (profiling):** ✅ Complete.  ``src/profiler.py`` produces dataset-wide
  statistics via exact Pébay (2008) parallel merge.  Six measurement sites per
  encoder block.  ``profiling_result.json`` includes global μ, σ, kurtosis,
  outlier fractions, per-channel σ and μ, attention entropy, LayerNorm γ/β, LN2
  amplification ratio, max/min, and ``RunMetadata``.
- **Phase 2 (ablation):** ✅ Complete.  Outlier zeroing uses
  mean-centered threshold (``|x − μ| > k·σ``) consistent with Phase 1.
  Random-zeroing control condition included.  Per-channel ablation
  (``--granularity per_channel``) tests whether outlier concentration in
  high-variance channels drives accuracy degradation.  Subset evaluation uses
  class-balanced sampling.  Entropy deltas computed against Phase 1 baselines.
- **Phase 3 (integer GELU):** 🔲 Not yet implemented.

### Key Phase 2 results (50k images)

| k | Global | Per-channel | Δ |
|---|--------|-------------|---|
| 3.0 | 43.24% | 47.00% | +3.76% |
| 4.0 | 75.12% | 75.54% | +0.42% |
| 6.0 | 84.58% | 84.11% | −0.47% |

Baseline: 85.03%.  Per-channel thresholds preserve significantly more accuracy
at aggressive thresholds (k=3), suggesting per-channel quantization of the MLP
hidden dimension would reduce the accuracy penalty of INT8 range clipping.

### Running tests

```bash
# Fast tests only
pytest -m "not slow" tests/

# All tests (requires GPU + nnsight)
pytest tests/

# Ablation-specific
pytest tests/test_ablation.py -v
```

---

## Architecture: profiling modules

### `src/profiler.py` — nnsight pipeline  **Primary for all phases**

**Status:** ✅ Complete.  Single-pass ``profile_vit`` + multi-batch
``run_profiling_dataset_pass`` + two-pass ``run_outlier_counting_pass``.

---

## Implementation Order

### Step 1: `src/utils.py` — ✅ DONE
### Step 2: `src/model.py` — ✅ DONE
### Step 3: `src/data_loader.py` — ✅ DONE (class-imbalance fix applied 2026-08-01)
### Step 4: `src/profiler.py` — ✅ DONE (single-pass API)
### Step 4b: `src/profiler.py` extension — ✅ DONE (Welford multi-batch API + per-channel std)
### Step 5: `src/exp1_profiling.py` — ✅ DONE
### Step 6: `src/plotting.py` — ✅ Phase 1 + Phase 2 functions done
### Step 6b: `histogram_profile_vit` + histogram pipeline rewrite — ✅ DONE
### Step 6c: Documentation fixes — ✅ DONE
### Step 7: `src/ablation.py` — ✅ DONE (nnsight-based intervention, mean-centered, random control)
### Step 7b: Phase 2 fixes (2026-08-01) — ✅ DONE
  - Mean-centered thresholding (T-020)
  - Random-zeroing control (T-021)
  - Class-imbalance fix in subset mode (T-022)
### Step 8: `src/exp2_ablation.py` — ✅ DONE (nnsight-based orchestrator with random control)
### Step 8b: Phase 2 per-channel ablation (2026-08-02) — ✅ DONE
  - Per-channel mean serialized in Phase 1 (``LayerStats.per_channel_mean``)
  - ``_build_per_channel_zeroing_mask`` in ``ablation.py``
  - ``--granularity per_channel`` CLI flag
  - Per-channel mode: pre_gelu only, no random control
  - Phase 1 re-run with ``--all --seed 42``
  - Full 50k-image global + per-channel ablation runs complete
### Step 9: `src/integer_gelu.py` — 🔲
### Step 10: `src/exp3_integer_gelu.py` — 🔲

---

## What to Read (and When)

### Before Step 9 (integer GELU)
- Kim et al. (2021), "I-BERT," ICML 2021 — integer-only GELU via polynomial approx.
- Li & Gu (2023), "I-ViT," ICCV 2023 — ShiftGELU for ViT.
- `src/profiler.py` — ``LayerStats`` fields available for quantization scale derivation.

### Background (read anytime)
- Pébay (2008), SAND2008-6212 — parallel higher-moments merge.
- Bondarenko et al. (2021), arXiv:2109.12948 — transformer quantization challenges.
- Dettmers et al. (2022), "LLM.int8()," NeurIPS 2022 — outlier handling.
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight) — outlier suppression.
- Xiao et al. (2023), "SmoothQuant," ICML 2023 — per-channel scaling.
- Zhai et al. (2023), ICML, arXiv:2303.06296 — attention entropy collapse.
- Maisonnave et al. (2025), arXiv:2508.16311 — CLS/patch entropy separation.