# ViT Quantization & Outlier Profiling

Research codebase for profiling and ablating massive activation outliers in a
Vision Transformer, with a pathway toward integer-only inference on NVIDIA
Jetson edge hardware.

**Target model:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` via [`timm`](https://github.com/huggingface/pytorch-image-models)
**Dataset:** ImageNet-1K validation split
**Experimental specs:** [`docs/EXP1-IMPL.md`](docs/EXP1-IMPL.md) (Phase 1) and [`docs/EXP2-IMPL.md`](docs/EXP2-IMPL.md) (Phase 2)
**Phase 2 expansion:** [`docs/phase2-expansion.md`](docs/phase2-expansion.md)

---

## Research Summary

Pre-GELU activations and attention logits in ViTs exhibit heavy-tailed distributions
dominated by a small number of massive outliers. This project investigates two
questions:

1. **Where are the outliers?** We profile activation statistics (mean, std, kurtosis,
   outlier fractions, per-channel σ and μ) at **6 measurement sites** across all 12 encoder
   blocks (Phase 1). ✅ Complete.
2. **How much do they matter, and why?** We zero out values beyond k·σ and measure the
   accuracy degradation curve. A random-zeroing control condition isolates
   the effect of outliers specifically. Per-channel ablation decomposes the effect
   into mean-correction and variance-correction components across all four
   channel-structured sites (pre_gelu, post_layernorm_1, post_layernorm_2,
   residual_stream). Layer-group ablation isolates which blocks drive the
   degradation (Phase 2). ✅ Complete; expansion complete.

> **Note:** Phase 3 (integer GELU LUTs) was deferred on 2026-08-03 in favor of
> deeper per-channel ablation analysis.  See `docs/phase2-expansion.md`.

---

## Key Results (50,000 images, 5 seeds: 42–46)

### Phase 1: Activation Profiling

| Metric | Finding |
|--------|---------|
| Sites profiled | 73 (6 per block + final residual stream) |
| Pre-GELU block 10 | μ = −28.33, σ = 11.20, kurtosis = 0.60 |
| Per-channel σ spread (block 10) | 2.06 - 25.54 (12× range) |
| LN2 γ vs σ_c correlation | r ≈ 0.0003 (no correlation; per-channel variance emerges from fc1.weight interaction) |
| Attention entropy | CLS entropy collapses in later blocks (entropy sink phenomenon) |

### Phase 2: Outlier Ablation

**Baseline top-1: 85.03%**, baseline top-5: 97.52%

| k | Global (top-1) | Per-channel (top-1) | Δ | 95% CI |
|---|---|---|---|---|
| 3.0 | 43.24% | 47.00% | **+3.76%** | [3.12%, 4.36%] |
| 4.0 | 75.12% | 75.54% | +0.42% | [−0.11%, 0.96%] |
| 6.0 | 84.58% | 84.11% | −0.47% | [−0.93%, −0.03%] |

**Key finding:** Per-channel thresholds preserve 3.76% more accuracy at k=3
(95% CI: [3.12%, 4.36%], which is statistically significant). The effect vanishes
by k=4. The random-zeroing control confirms that accuracy degradation is caused
by outliers specifically, as opposed to general activation sparsity.

### Post-Hoc Analysis

**Degradation efficiency (accuracy loss per 1% sparsity):**

| k | Global | Per-channel | Efficiency ratio |
|---|---|---|---|
| 3.0 | 100.97 pp/% | 53.43 pp/% | **1.89×** |
| 4.0 | 23.94 pp/% | 13.33 pp/% | 1.80× |
| 6.0 | 1.07 pp/% | 1.28 pp/% | 0.83× |

Per-channel thresholds are **1.89× more efficient** at k=3. Each 1% of zeroed
elements costs half as much accuracy. This indicates that per-channel thresholds
selectively preserve channels that carry more classification signal.

**Effective channels preserved (of 36,864 total across 12 blocks × 3,072):**

| k | Global | Per-channel | Δ |
|---|---|---|---|
| 3.0 | 36,711 | 36,602 | −110 |
| 4.0 | 36,844 | 36,822 | −21 |
| 6.0 | 36,862 | 36,855 | −8 |

Per-channel ablation achieves 3.76% *higher* accuracy while preserving slightly
*fewer* total channels (110 fewer at k=3). This confirms that per-channel thresholds
redistribute the zeroing budget from low-importance channels to high-importance
channels.

### Per-Channel Ablation Decomposition (k=3)

| Condition | top-1 | Δ vs global |
|-----------|-------|-------------|
| Baseline | 85.03% | — |
| Global outlier | 43.24% | — |
| Per-channel outlier | 47.00% | +3.76 pp |
| Per-channel mean_only | **63.32%** | **+20.08 pp** |
| Per-channel var_only | 6.56% | −36.68 pp |

**Key finding:** Mean correction dominates.  Per-channel μ_c recovers 20 pp over
the global condition by correcting for shifted channel means (μ_c ∈ [−71.18, 26.01]
at Block 10).  Variance correction alone (per-channel σ_c with global μ) is
catastrophic — it applies narrow thresholds to channels with negative means,
zeroing activations that are genuinely within-channel normal.  See
``docs/phase2-expansion.md`` for full analysis.

---

## Repository Layout

```
.
├── run_phase1_profiling.py      # Phase 1 entry point (--all, --num-seeds, --seed, --approximate-outliers)
├── run_phase2_ablation.py       # Phase 2 entry point (--sigma-thresholds, --granularity, --ablation-mode, --layer-range, --per-channel-sites)
│
├── src/
│   ├── config.py                # frozen dataclasses for all experiment configs
│   ├── model.py                 # load ViT-B/16, evaluate top-1/top-5 accuracy
│   ├── data_loader.py           # ImageFolder DataLoader with auto-shuffle
│   ├── profiler.py              # nnsight-based profiler (6-site + Welford multi-batch + outlier recount)
│   ├── ablation.py              # nnsight-based outlier zeroing with random control + per-channel
│   ├── plotting.py              # workhorse figure generation (headless matplotlib)
│   ├── plotting_poster.py       # poster-quality figures (custom palettes, annotation-driven)
│   ├── utils.py                 # seed_everything, get_device, ensure_dir, log_system_info
│   ├── exceptions.py            # DataDirectoryError, ProfilingError
│   ├── exp1_profiling.py        # Phase 1 orchestrator (✅ complete)
│   └── exp2_ablation.py         # Phase 2 orchestrator (✅ complete)
│
├── tests/
│   ├── conftest.py
│   ├── test_ablation.py
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_exceptions.py
│   ├── test_model.py
│   ├── test_plotting.py
│   ├── test_plotting_poster.py
│   ├── test_profiler.py
│   └── test_utils.py
│
├── scripts/
│   ├── regenerate_all.sh                  # unified plot regeneration from a run directory
│   ├── run_full_experiment.sh             # full experiment orchestration (Phase 1 + Phase 2)
│   ├── generate_poster_plots.py          # poster-quality figure generation
│   ├── regenerate_plots.py               # workhorse plot regeneration from data files
│   ├── analyze_ablation_results.py       # CI, sufficiency, degradation efficiency
│   ├── analyze_layernorm_gamma.py        # LN γ vs per-channel σ correlation
│   ├── analyze_effective_gain.py         # fc1.weight ⊙ γ effective gain analysis
│   ├── smoke_test_nnsight_intervention.py
│   └── verify_pre_softmax_fidelity.py
│
├── docs/
│   ├── EXP1-IMPL.md                  # Phase 1 implementation spec (authoritative)
│   ├── EXP2-IMPL.md                  # Phase 2 implementation spec (authoritative)
│   ├── EXP2b-PLAN.md                 # per-channel ablation planning (historical)
│   ├── phase2-expansion.md           # Phase 2 expansion: research questions & experiments
│   ├── NEXT-STEPS.md                 # implementation roadmap
│   ├── issues.md                     # active issues & known limitations
│   ├── MISTAKES.md                   # historical wrong approaches
│   ├── CITATIONS.md                  # verified bibliography
│   └── AI-DISCLAIMER.md
│
├── outputs/                     # written by runners (git-ignored)
│   ├── phase1-profiling/
│   ├── phase2-ablation/
│   ├── layernorm-gamma-analysis/
│   ├── ablation-analysis/
│   └── poster-plots/
│
├── data/                        # ImageNet val images (git-ignored)
├── download_imagenet_val.py     # helper to download ImageNet-1K val split
├── environment.yml
└── pytest.ini
```

---

## Setup

```sh
# Create the environment (once).
conda env create -f environment.yml
conda activate vit-quant
```

> **Requirements:** PyTorch ≥2.5, nnsight ≥0.7, timm ≥1.0, CUDA-capable GPU
> with ≥8 GB VRAM recommended. Tested on Python 3.13, PyTorch 2.12.1, nnsight 0.7.0,
> NVIDIA RTX 3070 (8 GB).

---

## Getting the Data

The experiments need ImageNet-1K validation images under `data/` in ImageFolder
layout (`data/<class_name>/<image>.JPEG`). The dataset should contain 50,000
images across 1,000 classes (50 images per class).

Use `download_imagenet_val.py` to download the validation split automatically.

---

## Methodological Note

This is a **mechanistic interpretability and ablation study**, not a
generalisation claim.  Phase 1 calibrates per-layer outlier thresholds (μ, σ)
on the ImageNet-1K validation set.  Phase 2 then applies those thresholds to
the **same** validation set to measure the accuracy impact of zeroing
elements that exceed them.

This is not a train/test leak in the traditional ML sense. No model
parameters are updated, no hyperparameters are tuned, and no optimisation
occurs on the validation set. The thresholds are purely descriptive
population statistics of the activation distributions. Using a separate
calibration set (e.g., the ImageNet training split) would produce different
thresholds and would answer a *different question*: "what happens when
you zero elements based on activation statistics from a disjoint set of
images?" That is a relevant follow-up, not the question this study
sets out to answer.

The same validation set is used for three distinct purposes:
1. **Profiling** (Phase 1): computing population μ and σ of activation tensors.
2. **Ablation** (Phase 2): zeroing elements beyond k·σ and measuring accuracy.
3. **Accuracy evaluation** (Phase 2): computing top-1/top-5 on the zeroed model.

All three operate on the same 50,000 images. This is acceptable for a
descriptive mechanistic study. We are characterising the *observed*
activation statistics and measuring what happens when we surgically remove
outliers from those specific statistics on those specific images.

**Disclosure for reviewers:** A more rigorous separation would profile on
the ImageNet-1K training set (1.28M images) and ablate/evaluate on the
validation set.  This would eliminate any concern about distributional
overlap between calibration and evaluation, at the cost of ~12 additional
GPU-hours.  We disclose this design decision explicitly and welcome reviewer
guidance on whether the additional compute is warranted for the claims made.

---

## Running the Experiments

### Full orchestration (recommended)

```sh
# Full run (50k images, 3 seeds, all ablation modes).
bash scripts/run_full_experiment.sh

# Smoke test (128 images, 1 seed, ~2 min).
bash scripts/run_full_experiment.sh --smoke

# Named run with custom settings.
bash scripts/run_full_experiment.sh --name my-run --num-seeds 5 --batch-size 128

# Show all options.
bash scripts/run_full_experiment.sh --help
```

### Individual experiment runs

### Phase 1 — Activation Profiling

> **⚠️ Important:** Phase 1 MUST be re-run if the profiler has changed since the
> last profiling output was produced (e.g., new ``track_per_channel`` sites added,
> ``LayerStats`` fields changed). Per-channel Phase 2 ablation will refuse to run
> with stale profiling data. See ``_check_residual_stream_per_channel_stats`` in
> ``src/exp2_ablation.py``.

```sh
# Quick run (1,024 images).
python run_phase1_profiling.py --num-images 1024

# Multi-seed (3 independent runs).
python run_phase1_profiling.py --num-images 1024 --num-seeds 3 --seed 42

# Full dataset profiling.
python run_phase1_profiling.py --all

# Fast iteration (skip the second-pass global-σ outlier recount).
python run_phase1_profiling.py --num-images 1024 --approximate-outliers
```

### Phase 2 — Outlier Ablation

```sh
# Global ablation sweep (default: 50k images, k ∈ {3, 4, 6}).
python run_phase2_ablation.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json

# Custom sigma thresholds.
python run_phase2_ablation.py --num-images 1024 \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --sigma-thresholds 3.0 6.0

# Per-channel ablation (all four channel-structured sites).
python run_phase2_ablation.py --num-images 50000 --granularity per_channel \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json

# Per-channel ablation restricted to pre_gelu only (e.g. for RQ2).
python run_phase2_ablation.py --num-images 50000 --granularity per_channel \
    --per-channel-sites pre_gelu \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json

# Mean-only per-channel (isolates mean-correction component).
python run_phase2_ablation.py --num-images 50000 \
    --granularity per_channel --ablation-mode mean_only \
    --sigma-thresholds 3.0

# Var-only per-channel (isolates variance-correction component).
python run_phase2_ablation.py --num-images 50000 \
    --granularity per_channel --ablation-mode var_only \
    --sigma-thresholds 3.0

# Layer-group ablation (block 10 only).
python run_phase2_ablation.py --num-images 50000 \
    --granularity per_channel --layer-range 10 10 \
    --sigma-thresholds 3.0
```

---

## Generating Plots

Plots are **not** generated during experiment runs (except activation histograms
in Phase 1, which need the live model).  All visualisation is done offline from
data files via dedicated scripts.

### One-command regeneration (recommended)

Regenerate everything from a single run directory:

```sh
bash scripts/regenerate_all.sh --run-dir outputs/full-run-2026-8-4
```

This auto-discovers all data files and produces:

| Subdirectory | Contents |
|-------------|----------|
| `plots/phase1/` | Phase 1 profiling plots (heatmaps, histograms) |
| `plots/phase2/` | Phase 2 single-run and comparison plots |
| `plots/poster/` | Poster-quality figures |
| `plots/analysis/` | Ablation analysis, effective gain, layernorm gamma |

### Workhorse plots (researcher iteration)

Regenerate standard plots from existing data. This is fast, functional, and requires no GPU
(except `--histograms`):

```sh
# Phase 1 plots from profiling_result.json.
python scripts/regenerate_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/phase1-profiling/seed_42/

# Phase 2 plots from a single ablation CSV.
python scripts/regenerate_plots.py \
    --csv outputs/phase2-ablation/ablation_results.csv \
    --output-dir outputs/phase2-ablation/

# Phase 2 comparison (global vs per-channel overlay).
python scripts/regenerate_plots.py \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/phase2-comparison/

# Full suite with activation histograms (needs model + GPU).
python scripts/regenerate_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/all-plots/ \
    --histograms --data-dir data

# Convenience: auto-discover from a run directory.
python scripts/regenerate_plots.py \
    --run-dir outputs/full-run-2026-8-4 \
    --output-dir outputs/full-run-2026-8-4/plots/
```

### Poster-quality plots (presentation / publication)

Generate polished figures suitable for posters and papers. These use custom colour
palettes, direct annotation, ≥14 pt fonts, and have no chartjunk:

```sh
# Phase 1 only (profiling stats).
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/poster-plots

# Phase 1 + Phase 2 comparison.
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/poster-plots

# Full suite including activation distribution overlay (needs model + GPU).
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/poster-plots \
    --histogram-data-dir data

# Merge mean_only + var_only + outlier CSVs for the waterfall chart.
python scripts/generate_poster_plots.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/phase2-global/seed_42/ablation_results.csv \
    --csv-b outputs/phase2-per-channel/seed_42/ablation_results.csv \
    --csv-b outputs/phase2-per-channel-mean-only/seed_42/ablation_results.csv \
    --csv-b outputs/phase2-per-channel-var-only/seed_42/ablation_results.csv \
    --output-dir outputs/poster-plots

# Convenience: auto-discover from a run directory.
python scripts/generate_poster_plots.py \
    --run-dir outputs/full-run-2026-8-4 \
    --output-dir outputs/full-run-2026-8-4/plots/poster
```

**Poster plots generated:**

| Plot | Description |
|------|-------------|
| `poster_outlier_grid_*.png` | 12x6 tile grid showing outlier fractions across all sites at a glance. |
| `poster_sigma_ridgeline.png` | Per-channel σ distributions as overlapping density curves. Block depth maps to colour. |
| `poster_entropy_streamgraph.png` | CLS attention entropy stream. Collapse is visible as narrowing bands. |
| `poster_mean_hinton_blk10.png` | Hinton diagram of per-channel μ (area is proportional to |μ|, colour is sign). |
| `poster_accuracy_vs_sparsity.png` | Connected scatter plot of accuracy vs %-zeroed, color-coded by condition. |
| `poster_ablation_waterfall.png` | Waterfall chart decomposing the per-channel accuracy benefit. |
| `poster_activation_overlay_blk10.png` | Hero figure showing an activation histogram with global vs per-channel threshold bands. |

---

## Post-Hoc Analysis Scripts

```sh
# 95% CI on global vs per-channel accuracy delta.
python scripts/analyze_ablation_results.py \
    --csv-a outputs/phase2-ablation-global-50k/ablation_results.csv \
    --csv-b outputs/phase2-ablation-per-channel-50k/ablation_results.csv \
    --output-dir outputs/ablation-analysis

# LayerNorm γ vs per-channel σ correlation.
python scripts/analyze_layernorm_gamma.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/layernorm-gamma-analysis

# Effective per-channel gain (‖fc1⊙γ‖) vs per-channel σ correlation.
python scripts/analyze_effective_gain.py \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/effective-gain-analysis

# Convenience: auto-discover from a run directory.
python scripts/analyze_ablation_results.py \
    --run-dir outputs/full-run-2026-8-4 \
    --output-dir outputs/full-run-2026-8-4/plots/analysis/ablation
```

---

## Running the Tests

```sh
pytest -m "not slow"      # Fast suite (192 tests), no model download required
pytest                     # Full suite (252 tests, includes model-dependent slow tests)
```
