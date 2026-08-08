# Next Steps: Implementation Roadmap

> **Last updated:** 2026-08-08 — 5-seed multi-condition full run complete; RQ2 and RQ4 resolved.

---

## Current State

- **Phase 1 (profiling):** ✅ Complete.  ``src/profiler.py`` produces dataset-wide
  statistics via exact Pébay (2008) parallel merge.  Six measurement sites per
  encoder block.  ``profiling_result.json`` includes global μ, σ, kurtosis,
  outlier fractions, per-channel σ and μ, attention entropy, LayerNorm γ/β, LN2
  amplification ratio, max/min, and ``RunMetadata``.

- **Phase 2 (ablation):** ✅ Complete; expansion ongoing.  Outlier zeroing uses
  mean-centered threshold (``|x − μ| > k·σ``) consistent with Phase 1.
  Random-zeroing control condition included.  Per-channel ablation with
  ``--granularity per_channel``, ``--ablation-mode {outlier,mean_only,var_only}``,
  ``--layer-range START END``, and ``--per-channel-sites SITE [SITE ...]``.
  Per-channel mode now covers all four channel-structured sites (pre_gelu,
  post_layernorm_1, post_layernorm_2, residual_stream) via a single generic
  intervention function.

- **Phase 3 (integer GELU):** ❌ Deleted 2026-08-03.  Focus shifted to Phase 2
  expansion.  See `docs/phase2-expansion.md`.

- **Plotting:** ✅ Complete.  Two-tier architecture:
  - `src/plotting.py` — workhorse plots for research iteration (~17 functions).
  - `src/plotting_poster.py` — poster-quality plots for presentations (~7 functions).
  - `scripts/regenerate_plots.py` — regenerates all workhorse plots from data files.
  - `scripts/generate_poster_plots.py` — generates all poster plots from data files.
  - `scripts/analyze_ablation_results.py` — post-hoc analysis with CIs.
  - `scripts/analyze_layernorm_gamma.py` — LN γ correlation analysis.
  - `scripts/analyze_effective_gain.py` — fc1.weight ⊙ γ effective gain analysis.

- **Tests:** 252 total (192 fast + 60 slow, model-dependent).  All pass.
  Expanded with 10 new per-channel site tests (2026-08-05).

### Key Phase 2 results (50k images, 5 seeds)

| k | Global | Per-channel | Δ | 95% CI |
|---|--------|-------------|---|--------|
| 3.0 | 43.24% | 47.00% | +3.76% | [3.12%, 4.36%] |
| 4.0 | 75.12% | 75.54% | +0.42% | [−0.11%, 0.96%] |
| 6.0 | 84.58% | 84.11% | −0.47% | [−0.93%, −0.03%] |

Baseline: 85.03%.  Per-channel thresholds preserve significantly more accuracy
at aggressive thresholds (k=3).  Ablation is deterministic given fixed Phase 1
stats, so all 5 seeds produce identical accuracies.

### Per-channel ablation decomposition (k=3, 5 seeds)

| Condition | top-1 |
|-----------|-------|
| Baseline | 85.03% |
| Global outlier | 43.24% |
| Per-channel outlier | 47.00% |
| Per-channel mean_only | **63.32%** |
| Per-channel var_only | 6.56% |

**Key finding:** Mean correction is the dominant mechanism (recovering 20 pp
over global).  Variance correction alone is catastrophic — per-channel σ_c
without μ_c zeros the wrong activations.

### Effective gain analysis (2026-08-03)

Pearson r(LN2 γ, pre-GELU σ_c) ≈ 0.0003 — no correlation (different dimensionalities).
The proper analysis computes ``‖fc1.weight[c, :] ⊙ γ‖`` per channel (both 3072-dim):

| Block | r(gain, σ_c) |
|-------|-------------|
| 0–7   | −0.13 to +0.21 |
| 8     | **+0.7550** |
| 9     | **+0.7747** |
| 10    | **+0.6496** |
| 11    | **+0.7674** |

Mean r across all blocks: **+0.3241**.  The strong correlation in late blocks
(8–11) confirms the SmoothQuant hypothesis: the per-channel variance pattern is
architectural — encoded in the interaction of fc1.weight and LN2 γ.  See
`scripts/analyze_effective_gain.py`.

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
### Step 8b2: Per-channel site expansion (2026-08-05) — ✅ DONE
  - Added ``track_per_channel=True`` to residual stream saves in Phase 1
  - Created ``_intervene_per_channel_generic()`` — single function for any (B,N,D) site
  - Expanded per-channel site list to all four channel-structured sites
  - Added ``--per-channel-sites`` CLI flag (defaults to all four)
  - Enabled random-zeroing control for per-channel mode
  - Added 10 new tests for per-channel zeroing at new sites
  - Updated ``scripts/run_full_experiment.sh`` to pass ``--per-channel-sites pre_gelu`` for RQ2 runs
### Step 8c: Phase 2 expansion (2026-08-03) — ✅ DONE
  - Phase 3 deleted (code, tests, config, plotting references)
  - ``--ablation-mode {outlier,mean_only,var_only}`` CLI flag
  - ``--layer-range START END`` CLI flag
  - ``scripts/analyze_layernorm_gamma.py`` (LN γ correlation analysis)
  - ``docs/phase2-expansion.md`` (research questions & experiment plan)
### Step 8d: Plotting architecture (2026-08-03) — ✅ DONE
  - ``--approximate-outliers`` renamed from ``--skip-outlier-recount`` for clarity
  - Workhorse plotting decoupled from experiment runs
  - ``src/plotting.py`` expanded: 10 new Phase 1 + Phase 2 plot functions
  - ``scripts/regenerate_plots.py`` rewritten as universal workhorse regenerator
  - ``scripts/analyze_ablation_results.py`` uses ``src/plotting.py`` for all figures
  - Poster-quality plotting: ``src/plotting_poster.py`` + ``scripts/generate_poster_plots.py``
  - 7 poster plot types: activation overlay, site grid, ridgeline, streamgraph, Hinton,
    accuracy-vs-sparsity, ablation waterfall
  - 252 total test functions (192 fast + 60 slow, model-dependent).  All pass.
### Step 8e: Phase 2 expansion experiments — 🔲 (partial)
  - mean_only + var_only at k=3 (RQ2) — ✅ Complete (2026-08-08)
  - Multi-seed variance at k=3 (RQ4) — ✅ Complete (2026-08-08)
  - Layer-group ablation at k=3 (RQ3) — 🔲 Open
  - Finer k-sweep [2.5..3.5] (RQ5) — 🔲 Open

---

## Quick Reference — Plotting Commands

### One-command regeneration
```sh
bash scripts/regenerate_all.sh --run-dir outputs/full-run-2026-8-4
```

### Workhorse plots
```sh
# Phase 1
python scripts/regenerate_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/phase1-profiling/seed_42/

# Phase 2 single run
python scripts/regenerate_plots.py \
    --csv outputs/phase2-ablation/ablation_results.csv \
    --output-dir outputs/phase2-ablation/

# Phase 2 comparison
python scripts/regenerate_plots.py \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/phase2-comparison/

# Convenience: auto-discover from run directory
python scripts/regenerate_plots.py \
    --run-dir outputs/full-run-2026-8-4 \
    --output-dir outputs/full-run-2026-8-4/plots/
```

### Poster plots
```sh
# Phase 1 + Phase 2
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/poster-plots

# Full suite with activation overlay (needs GPU)
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/poster-plots \
    --histogram-data-dir data

# Merge mean_only + var_only + outlier for waterfall
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-global/seed_42/ablation_results.csv \
    --csv-b outputs/phase2-per-channel/seed_42/ablation_results.csv \
    --csv-b outputs/phase2-per-channel-mean-only/seed_42/ablation_results.csv \
    --csv-b outputs/phase2-per-channel-var-only/seed_42/ablation_results.csv \
    --output-dir outputs/poster-plots
```

### Post-hoc analysis
```sh
# 95% CI + effective channels + degradation efficiency
python scripts/analyze_ablation_results.py \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/ablation-analysis

# LN γ correlation
python scripts/analyze_layernorm_gamma.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/layernorm-gamma-analysis

# Effective gain correlation
python scripts/analyze_effective_gain.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/effective-gain-analysis
```

---

## What to Read (and When)

### Before running experiments
- `docs/EXP1-IMPL.md` — Phase 1 profiling specification (statistical conventions, site naming).
- `docs/EXP2-IMPL.md` — Phase 2 ablation specification (intervention logic, threshold definitions).
- `docs/phase2-expansion.md` — research questions and experiment plan for per-channel deep dive.

### Understanding the codebase
- `docs/NEXT-STEPS.md` — this file: implementation roadmap and current state.
- `docs/issues.md` — open and closed tickets (T-001 through T-033).
- `docs/MISTAKES.md` — historical wrong approaches and lessons learned.
- `docs/CITATIONS.md` — verified bibliography with usage context.

### Background reading
- Xiao et al. (2023), "SmoothQuant," ICML 2023 — per-channel scaling and LN γ analysis.
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight) — outlier suppression.
- Pébay (2008), SAND2008-6212 — parallel higher-moments merge.
- Bondarenko et al. (2021), arXiv:2109.12948 — transformer quantization challenges.
- Dettmers et al. (2022), "LLM.int8()," NeurIPS 2022 — outlier handling.
- Zhai et al. (2023), ICML, arXiv:2303.06296 — attention entropy collapse.
- Maisonnave et al. (2025), arXiv:2508.16311 — CLS/patch entropy separation.
