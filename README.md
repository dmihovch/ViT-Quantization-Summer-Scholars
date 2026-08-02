# ViT Quantization & Outlier Profiling

Research codebase for profiling and ablating massive activation outliers in a
Vision Transformer, with a pathway toward integer-only GELU execution on NVIDIA
Jetson edge hardware.

**Target model:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` via [`timm`](https://github.com/huggingface/pytorch-image-models)
**Dataset:** ImageNet-1K validation split
**Experimental spec:** [`docs/EXP1-IMPL.md`](docs/EXP1-IMPL.md) (Phase 1) and [`docs/EXP2-IMPL.md`](docs/EXP2-IMPL.md) (Phase 2)

---

## Research Summary

Pre-GELU activations and attention logits in ViTs exhibit heavy-tailed distributions
dominated by a small number of massive outliers. This project answers three questions:

1. **Where are the outliers?** — Profile activation statistics (mean, std, kurtosis,
   outlier fractions, per-channel σ and μ) at **6 measurement sites** across all 12 encoder
   blocks (Phase 1). ✅ Complete.
2. **How much do they matter?** — Zero out values beyond k·σ and measure the
   accuracy degradation curve, with a random-zeroing control condition to isolate
   the effect of outliers specifically.  Includes per-channel ablation to test
   whether outlier concentration in high-variance channels drives degradation (Phase 2). ✅ Complete.
3. **Can GELU run on integers?** — Build per-layer INT8 LUTs that approximate
   GELU without any FP32 dequantization (Phase 3). 🔲 Not yet implemented.

---

## Key Results (50,000 images, seed=42)

### Phase 1 — Activation Profiling

| Metric | Finding |
|--------|---------|
| Sites profiled | 73 (6 per block + final residual stream) |
| Pre-GELU block 10 | μ = −28.33, σ = 11.20, kurtosis = 0.60 |
| Per-channel σ spread (block 10) | 2.06 – 25.54 (12× range) |
| Attention entropy | CLS entropy collapses in later blocks (entropy sink phenomenon) |

### Phase 2 — Outlier Ablation

**Baseline top-1: 85.03%**

| k | Global (top-1) | Per-channel (top-1) | Δ |
|---|---|---|---|
| 3.0 | 43.24% | 47.00% | **+3.76%** |
| 4.0 | 75.12% | 75.54% | +0.42% |
| 6.0 | 84.58% | 84.11% | −0.47% |

**Key finding:** Per-channel thresholds preserve 3.76% more accuracy at k=3
by redistributing the zeroing budget away from high-variance channels that carry
signal.  At k≥4 the difference vanishes because both thresholds are wide enough
to preserve nearly all elements.  The random-zeroing control confirms that
accuracy degradation at k=3 is due to outliers specifically, not activation
sparsity in general (random zeroing at matched fractions preserves baseline
accuracy).

---

## Repository Layout

```
.
├── run_phase1_profiling.py      # Phase 1 entry point (--all, --num-seeds, --seed)
├── run_phase2_ablation.py       # Phase 2 entry point (--sigma-thresholds, --granularity)
├── run_phase3_integer_gelu.py   # Phase 3 entry point (stub)
│
├── src/
│   ├── config.py                # frozen dataclasses for all experiment configs
│   ├── model.py                 # load ViT-B/16, evaluate top-1/top-5 accuracy
│   ├── data_loader.py           # ImageFolder DataLoader with auto-shuffle
│   ├── profiler.py              # nnsight-based profiler (primary, 6-site + Welford multi-batch)
│   ├── ablation.py              # nnsight-based outlier zeroing with random control + per-channel
│   ├── integer_gelu.py          # LUT construction + FP32 comparison (stub)
│   ├── plotting.py              # all figure generation (headless matplotlib)
│   ├── utils.py                 # seed_everything, get_device, ensure_dir, log_system_info
│   ├── exceptions.py            # DataDirectoryError, ProfilingError, …
│   ├── exp1_profiling.py        # Phase 1 orchestrator (✅ complete)
│   ├── exp2_ablation.py         # Phase 2 orchestrator (✅ complete)
│   └── exp3_integer_gelu.py     # Phase 3 orchestrator (stub)
│
├── tests/
│   ├── conftest.py
│   ├── test_exceptions.py
│   ├── test_utils.py
│   ├── test_config.py
│   ├── test_profiler.py
│   ├── test_ablation.py
│   ├── test_integer_gelu.py
│   ├── test_data_loader.py
│   └── test_plotting.py
│
├── scripts/
│   ├── regenerate_plots.py
│   ├── smoke_test_nnsight_intervention.py
│   └── verify_pre_softmax_fidelity.py
│
├── docs/
│   ├── EXP1-IMPL.md                  # Phase 1 implementation spec (authoritative)
│   ├── EXP2-IMPL.md                  # Phase 2 implementation spec (authoritative)
│   ├── NEXT-STEPS.md                 # implementation roadmap
│   ├── issues.md                     # active issues & known limitations
│   ├── MISTAKES.md                   # historical wrong approaches
│   ├── CITATIONS.md                  # verified bibliography
│   └── AI-DISCLAIMER.md
│
├── outputs/                     # written by runners (git-ignored)
│   ├── phase1-profiling/
│   ├── phase2-ablation/
│   └── phase3-integer-gelu/
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

## Running the Experiments

```sh
# Phase 1 — profile activation distributions across 6 sites.
# 1,024 images (subset, auto-shuffled for class diversity).
python run_phase1_profiling.py --num-images 1024

# Multi-seed run for variance estimation.
python run_phase1_profiling.py --num-images 1024 --num-seeds 3 --seed 42

# Full 50k-image dataset.
python run_phase1_profiling.py --all

# Phase 2 — outlier ablation sweep with random-zeroing control.
# Default sigma thresholds match Phase 1 OUTLIER_SIGMAS (3, 4, 6).
python run_phase2_ablation.py --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json

# Phase 2 with custom thresholds and a subset for smoke-testing.
python run_phase2_ablation.py --num-images 1024 \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --sigma-thresholds 3.0 6.0

# Phase 2 — per-channel ablation (pre_gelu only, uses per-channel μ_c and σ_c).
python run_phase2_ablation.py --num-images 50000 \
    --layer-stats outputs/phase1-profiling/seed_42/profiling_result.json \
    --granularity per_channel

# Phase 3 — integer GELU LUT construction and comparison (stub).
python run_phase3_integer_gelu.py
```

Outputs land in `outputs/phase{1,2,3}-*/`.

---

## Running the Tests

```sh
# Fast suite — no model download required.
pytest -m "not slow"

# Full suite (includes slow tests requiring nnsight trace context).
pytest
```

> **Note:** 5 Phase 3 stub tests fail with `NotImplementedError` — this is
> expected until Phase 3 is implemented.  All 147 other tests pass.