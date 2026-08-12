# ViT Quantization & Outlier Profiling

Research codebase for profiling and ablating activation outliers in a Vision Transformer, with a view toward integer-only inference on edge hardware.

**Target model:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` via [`timm`](https://github.com/huggingface/pytorch-image-models)
**Dataset:** ImageNet-1K validation split

---

## Research Summary

This project profiles activation statistics at 73 measurement sites across all 12 encoder blocks of a ViT-B/16, then measures the accuracy impact of zeroing activations that exceed per-layer outlier thresholds.

**Phase 1** collects mean, std, kurtosis, outlier fractions, and per-channel statistics (σ_c, μ_c) at six sites per block over 50,000 images using exact Pébay parallel-merge statistics.

**Phase 2** zeros activations beyond k·σ thresholds and records top-1/top-5 accuracy. A random-zeroing control runs at matched sparsity to separate the effect of outlier removal from the effect of sparsity alone. Per-channel ablation decomposes the accuracy difference into mean-correction and variance-correction components.

---

## Key Results (50,000 images, 5 seeds: 42-46)

### Phase 1: Activation Profiling

| Metric | Value |
|--------|-------|
| Sites profiled | 73 (6 per block + final residual stream) |
| Pre-GELU block 10 mean | -28.33 |
| Pre-GELU block 10 std | 11.20 |
| Pre-GELU block 10 kurtosis | 0.60 |
| Per-channel σ range (block 10) | 2.06 - 25.54 (12.4x spread) |
| Per-channel μ range (block 10) | -71.18 - 26.01 |
| Outlier fraction at 3σ (block 10) | 0.39% |
| LN2 γ vs σ_c Pearson r | 0.0003 |

### Phase 2: Outlier Ablation

**Baseline top-1: 85.03%, top-5: 97.52%**

| k | Global top-1 | Per-channel top-1 | Delta | 95% CI |
|---|---|---|---|---|
| 3.0 | 43.24% | 47.00% | +3.76 pp | [3.12, 4.36] |
| 4.0 | 75.12% | 75.54% | +0.42 pp | [-0.11, 0.96] |
| 6.0 | 84.58% | 84.11% | -0.47 pp | [-0.93, -0.03] |

At k=3, per-channel thresholds yield 3.76 pp higher top-1 accuracy than global thresholds (95% CI: [3.12, 4.36] pp, two-proportion z-interval, N=50,000). The difference is within the confidence interval at k=4 and k=6.

The random-zeroing control preserves accuracy within 0.1 pp of baseline at all sparsity levels, confirming that the degradation under outlier zeroing is driven by the specific values removed, not by sparsity alone.

### Per-Channel Ablation Decomposition (k=3)

| Condition | top-1 | Delta vs global |
|-----------|-------|-----------------|
| Baseline | 85.03% | -- |
| Global outlier | 43.24% | -- |
| Per-channel outlier | 47.00% | +3.76 pp |
| Per-channel mean_only | 63.32% | +20.08 pp |
| Per-channel var_only | 6.56% | -36.68 pp |

Using per-channel μ_c with global σ (mean_only) recovers 20.08 pp over the global condition. Using per-channel σ_c with global μ (var_only) reduces accuracy to 6.56%. The mean_only condition corrects for shifted channel means (μ_c ranges from -71.18 to 26.01 at block 10). The var_only condition applies narrow per-channel thresholds centered on the global mean, which falls outside the true center of many channels, causing over-zeroing.

### Post-Hoc Analysis

**Degradation efficiency (accuracy loss per 1% of activations zeroed):**

| k | Global | Per-channel | Ratio |
|---|---|---|---|
| 3.0 | 100.97 pp/% | 53.43 pp/% | 1.89x |
| 4.0 | 23.94 pp/% | 13.33 pp/% | 1.80x |
| 6.0 | 1.07 pp/% | 1.28 pp/% | 0.83x |

**Effective channels preserved (12 blocks x 3,072 = 36,864 total):**

| k | Global | Per-channel | Delta |
|---|---|---|---|
| 3.0 | 36,711 | 36,602 | -110 |
| 4.0 | 36,844 | 36,822 | -21 |
| 6.0 | 36,862 | 36,855 | -8 |

**Effective gain correlation (Pearson r between ||fc1.weight[c,:] x γ||_2 and per-channel σ_c):**

| Block | r |
|-------|---|
| 0-7 | -0.13 to +0.21 |
| 8 | +0.755 |
| 9 | +0.775 |
| 10 | +0.650 |
| 11 | +0.767 |

---

## Repository Layout

```
.
├── run_phase1_profiling.py      # Phase 1 entry point
├── run_phase2_ablation.py       # Phase 2 entry point
│
├── src/
│   ├── config.py                # frozen dataclasses for experiment configs
│   ├── model.py                 # ViT-B/16 loading and top-1/top-5 evaluation
│   ├── data_loader.py           # ImageNet-1K DataLoader
│   ├── profiler.py              # nnsight-based activation profiler (Welford pipeline)
│   ├── ablation.py              # outlier zeroing with random control and per-channel modes
│   ├── plotting.py              # workhorse figure generation
│   ├── plotting_poster.py       # poster-quality figures
│   ├── plotting_utils.py        # shared colour palettes and label formatting
│   ├── utils.py                 # seed_everything, get_device, system metadata
│   ├── exceptions.py            # custom exception types
│   ├── exp1_profiling.py        # Phase 1 orchestrator
│   └── exp2_ablation.py         # Phase 2 orchestrator
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
│   ├── run_full_experiment.sh             # Phase 1 + Phase 2 orchestration
│   ├── regenerate_all.sh                  # regenerate all plots from a run directory
│   ├── regenerate_plots.py                # workhorse plot regeneration from data files
│   ├── generate_poster_plots.py           # poster figure generation
│   ├── generate_all_plots.py              # batch plot generation
│   ├── generate_gain_sigma_scatter.py     # effective gain vs per-channel σ scatter plots
│   ├── generate_gain_sigma_scatter_vertical.py
│   ├── generate_final_report_plots.py
│   ├── analyze_ablation_results.py        # CI and degradation efficiency
│   ├── analyze_layernorm_gamma.py         # LN γ vs per-channel σ correlation
│   ├── analyze_effective_gain.py          # fc1.weight x γ effective gain analysis
│   ├── validate_gain_correlation.py
│   ├── smoke_test_nnsight_intervention.py
│   └── verify_pre_softmax_fidelity.py
│
├── docs/
│   └── CITATIONS.md
│
├── plots/                       # figures (committed)
│   ├── phase1/
│   ├── phase2/
│   ├── poster/
│   └── analysis/
│
├── outputs/                     # experiment outputs (git-ignored)
├── data/                        # ImageNet val images (git-ignored)
├── download_imagenet_val.py
├── environment.yml
└── pytest.ini
```

---

## Setup

```sh
conda env create -f environment.yml
conda activate vit-quant
```

Requirements: PyTorch >= 2.5, nnsight >= 0.7, timm >= 1.0, CUDA-capable GPU with >= 8 GB VRAM. Tested on Python 3.13, PyTorch 2.12.1, nnsight 0.7.0, NVIDIA RTX 3070 (8 GB).

---

## Getting the Data

The experiments require ImageNet-1K validation images under `data/` in ImageFolder layout (`data/<class_name>/<image>.JPEG`), 50,000 images across 1,000 classes.

```sh
python download_imagenet_val.py
```

---

## Methodological Note

Phase 1 calibrates per-layer outlier thresholds (μ, σ) on the ImageNet-1K validation set. Phase 2 applies those thresholds to the same validation set to measure the accuracy impact of zeroing elements that exceed them.

No model parameters are updated, no hyperparameters are tuned, and no optimisation occurs on the validation set. The thresholds are descriptive population statistics of the activation distributions. The study characterises the observed activation statistics on these specific images and measures what happens when outlier-valued activations are zeroed.

The same 50,000 images are used for three purposes:
1. **Profiling** (Phase 1): computing population μ and σ of activation tensors.
2. **Ablation** (Phase 2): zeroing elements beyond k·σ.
3. **Evaluation** (Phase 2): computing top-1/top-5 on the zeroed model.

A more rigorous design would profile on the ImageNet-1K training set and evaluate on the validation set. This would eliminate calibration/evaluation overlap at the cost of roughly 12 additional GPU-hours.

---

## Running the Experiments

### Full run

```sh
# Full run (50k images, 5 seeds, all ablation modes).
bash scripts/run_full_experiment.sh

# Smoke test (128 images, 1 seed).
bash scripts/run_full_experiment.sh --smoke

# Custom settings.
bash scripts/run_full_experiment.sh --name my-run --num-seeds 5 --batch-size 128

# Show all options.
bash scripts/run_full_experiment.sh --help
```

### Phase 1 - Activation Profiling

> **Note:** Phase 2 per-channel ablation requires Phase 1 output. Re-run Phase 1 if `LayerStats` fields have changed since the last profiling run. See `_check_residual_stream_per_channel_stats` in `src/exp2_ablation.py`.

```sh
# Quick run (1,024 images).
python run_phase1_profiling.py --num-images 1024

# Multi-seed run.
python run_phase1_profiling.py --num-images 1024 --num-seeds 3 --seed 42

# Full dataset.
python run_phase1_profiling.py --all

# Skip the second-pass outlier recount.
python run_phase1_profiling.py --num-images 1024 --approximate-outliers
```

### Phase 2 - Outlier Ablation

```sh
# Global ablation sweep (k in {3, 4, 6}).
python run_phase2_ablation.py \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json

# Custom sigma thresholds.
python run_phase2_ablation.py --num-images 1024 \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json \
    --sigma-thresholds 3.0 6.0

# Per-channel ablation (all four channel-structured sites).
python run_phase2_ablation.py --num-images 50000 --granularity per_channel \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json

# Mean-only per-channel (mean-correction component only).
python run_phase2_ablation.py --num-images 50000 \
    --granularity per_channel --ablation-mode mean_only \
    --sigma-thresholds 3.0

# Var-only per-channel (variance-correction component only).
python run_phase2_ablation.py --num-images 50000 \
    --granularity per_channel --ablation-mode var_only \
    --sigma-thresholds 3.0
```

---

## Generating Plots

Plots are generated offline from saved data files. Experiment runs produce JSON and CSV output; plotting scripts consume those files separately.

### One-command regeneration

```sh
bash scripts/regenerate_all.sh --run-dir outputs/5-seed-full-run-2026-08-05
```

Outputs to `plots/` with subdirectories `phase1/`, `phase2/`, `poster/`, and `analysis/`.

### Workhorse plots

```sh
# Phase 1 plots.
python scripts/regenerate_plots.py \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/

# Phase 2 single-condition plots.
python scripts/regenerate_plots.py \
    --csv outputs/5-seed-full-run-2026-08-05/phase2-global/seed_42/ablation_results.csv \
    --output-dir outputs/5-seed-full-run-2026-08-05/phase2-global/seed_42/

# Phase 2 global vs per-channel comparison.
python scripts/regenerate_plots.py \
    --csv-a outputs/5-seed-full-run-2026-08-05/phase2-global/seed_42/ablation_results.csv \
    --csv-b outputs/5-seed-full-run-2026-08-05/phase2-per-channel/seed_42/ablation_results.csv \
    --output-dir outputs/phase2-comparison/
```

### Poster figures

```sh
python scripts/generate_poster_plots.py \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json \
    --csv-a outputs/5-seed-full-run-2026-08-05/phase2-global/seed_42/ablation_results.csv \
    --csv-b outputs/5-seed-full-run-2026-08-05/phase2-per-channel/seed_42/ablation_results.csv \
    --csv-b outputs/5-seed-full-run-2026-08-05/phase2-per-channel-mean-only/seed_42/ablation_results.csv \
    --csv-b outputs/5-seed-full-run-2026-08-05/phase2-per-channel-var-only/seed_42/ablation_results.csv \
    --output-dir outputs/poster-plots
```

---

## Post-Hoc Analysis Scripts

```sh
# 95% CI on global vs per-channel accuracy delta.
python scripts/analyze_ablation_results.py \
    --csv-a outputs/5-seed-full-run-2026-08-05/phase2-global/seed_42/ablation_results.csv \
    --csv-b outputs/5-seed-full-run-2026-08-05/phase2-per-channel/seed_42/ablation_results.csv \
    --output-dir outputs/ablation-analysis

# LayerNorm γ vs per-channel σ correlation.
python scripts/analyze_layernorm_gamma.py \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/layernorm-gamma-analysis

# Effective per-channel gain (||fc1 x γ||) vs per-channel σ correlation.
python scripts/analyze_effective_gain.py \
    --layer-stats outputs/5-seed-full-run-2026-08-05/phase1-profiling/seed_42/profiling_result.json \
    --output-dir outputs/effective-gain-analysis
```

---

## Running the Tests

```sh
pytest -m "not slow"   # fast suite, no model download required
pytest                  # full suite, includes GPU-dependent tests
```
