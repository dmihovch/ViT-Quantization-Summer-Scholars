# ViT Quantization & Outlier Profiling

Research codebase for profiling and ablating massive activation outliers in a
Vision Transformer, with a pathway toward integer-only GELU execution on NVIDIA
Jetson edge hardware.

**Target model:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` via [`timm`](https://github.com/huggingface/pytorch-image-models)
**Dataset:** ImageNet-1K validation split
**Experimental spec:** [`docs/vit_profiling_framework.md`](docs/vit_profiling_framework.md)

---

## Research Summary

Pre-GELU activations and attention logits in ViTs exhibit heavy-tailed distributions
dominated by a small number of massive outliers. This project answers three questions:

1. **Where are the outliers?** — Profile activation statistics (mean, std, kurtosis,
   outlier fractions, per-channel σ) at **6 measurement sites** across all 12 encoder
   blocks (Phase 1). ✅ Complete.
2. **How much do they matter?** — Zero out values beyond k·σ and measure the
   accuracy degradation curve (Phase 2).
3. **Can GELU run on integers?** — Build per-layer INT8 LUTs that approximate
   GELU without any FP32 dequantization (Phase 3).

---

## Repository Layout

```
.
├── run_phase1_profiling.py      # Phase 1 entry point (-all, --num-seeds, --seed)
├── run_phase2_ablation.py       # Phase 2 entry point (stub)
├── run_phase3_integer_gelu.py   # Phase 3 entry point (stub)
│
├── src/
│   ├── config.py                # frozen dataclasses for all experiment configs
│   ├── model.py                 # load ViT-B/16, evaluate top-1/top-5 accuracy
│   ├── data_loader.py           # ImageFolder DataLoader with auto-shuffle
│   ├── hooks.py                 # forward hook machinery (legacy, retained for reference)
│   ├── profiler.py              # nnsight-based profiler (primary, 6-site + Welford multi-batch)
│   ├── ablation.py              # outlier zeroing, % zeroed, AblationResult (stub)
│   ├── integer_gelu.py          # LUT construction + FP32 comparison (stub)
│   ├── plotting.py              # all figure generation (headless matplotlib)
│   ├── utils.py                 # seed_everything, get_device, ensure_dir, log_system_info
│   ├── exceptions.py            # DataDirectoryError, ProfilingError, …
│   ├── exp1_profiling.py        # Phase 1 orchestrator (✅ complete)
│   ├── exp2_ablation.py         # Phase 2 orchestrator (stub)
│   └── exp3_integer_gelu.py     # Phase 3 orchestrator (stub)
│
├── tests/
│   ├── conftest.py
│   ├── test_exceptions.py
│   ├── test_utils.py
│   ├── test_config.py
│   ├── test_hooks.py
│   ├── test_profiler.py         # 82 fast + 22 slow tests
│   ├── test_ablation.py
│   ├── test_integer_gelu.py
│   ├── test_data_loader.py
│   └── test_plotting.py
│
├── docs/
│   ├── EXP1-IMPL.md                  # Phase 1 implementation spec (authoritative)
│   ├── vit_profiling_framework.md    # experimental spec (source of truth)
│   ├── NEXT-STEPS.md                 # implementation roadmap
│   ├── open-issues.md                # active issues & known limitations
│   ├── mistakes-ledger.md            # historical wrong approaches
│   └── AI-DISCLAIMER.md
│
├── outputs/                     # written by runners (git-ignored)
│   ├── phase1-profiling/
│   ├── phase2-ablation/
│   └── phase3-integer-gelu/
│
├── data/                        # ImageNet val images (git-ignored)
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

# Phase 2 — outlier ablation sweep (needs full val set).
python run_phase2_ablation.py --sigma-thresholds 2.0 3.0 4.0 5.0

# Phase 3 — integer GELU LUT construction and comparison.
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