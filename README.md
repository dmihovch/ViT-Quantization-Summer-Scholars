# ViT Quantization & Outlier Profiling

Research codebase for profiling and ablating massive activation outliers in a
Vision Transformer, with a pathway toward integer-only GELU execution on NVIDIA
Jetson edge hardware.

**Target model:** `google/vit-base-patch16-224` via [`timm`](https://github.com/huggingface/pytorch-image-models)
**Dataset:** ImageNet-1K validation split
**Experimental spec:** [`docs/vit_profiling_framework.md`](docs/vit_profiling_framework.md)

---

## Research Summary

Pre-GELU activations in ViTs exhibit heavy-tailed distributions dominated by a
small number of massive outliers. This project answers three questions:

1. **Where are the outliers?** — Profile the max/min/std of pre-GELU tensors
   across all 12 encoder blocks (Phase 1).
2. **How much do they matter?** — Zero out values beyond k·σ and measure the
   accuracy degradation curve (Phase 2).
3. **Can GELU run on integers?** — Build per-layer INT8 LUTs that approximate
   GELU without any FP32 dequantization (Phase 3).

---

## Repository Layout

```
.
├── run_phase1_profiling.py      # Phase 1 entry point
├── run_phase2_ablation.py       # Phase 2 entry point
├── run_phase3_integer_gelu.py   # Phase 3 entry point
├── download_imagenet_val.py     # streams val images from Hugging Face
│
├── src/
│   ├── config.py                # frozen dataclasses for all experiment configs
│   ├── model.py                 # load ViT-B/16, evaluate top-1/top-5 accuracy
│   ├── data_loader.py           # ImageFolder-based DataLoader
│   ├── hooks.py                 # forward hook machinery + LayerStats
│   ├── ablation.py              # outlier zeroing, % zeroed, AblationResult
│   ├── integer_gelu.py          # LUT construction + FP32 comparison
│   ├── plotting.py              # all figure generation (headless matplotlib)
│   ├── utils.py                 # seed_everything, get_device, ensure_dir
│   ├── exceptions.py            # DataDirectoryError, HookRegistrationError, …
│   ├── exp1_profiling.py        # Phase 1 orchestrator
│   ├── exp2_ablation.py         # Phase 2 orchestrator
│   └── exp3_integer_gelu.py     # Phase 3 orchestrator
│
├── tests/
│   ├── conftest.py              # shared fixtures (temp dirs, dummy tensors)
│   ├── test_exceptions.py
│   ├── test_utils.py
│   ├── test_config.py
│   ├── test_hooks.py
│   ├── test_ablation.py
│   ├── test_integer_gelu.py
│   ├── test_data_loader.py
│   └── test_plotting.py
│
├── docs/
│   ├── vit_profiling_framework.md   # experimental spec (source of truth)
│   ├── NEXT-STEPS.md                # implementation roadmap + reading list
│   └── AI-DISCLAIMER.md
│
├── outputs/                     # written by runners (git-ignored)
│   ├── phase1-profiling/
│   ├── phase2-ablation/
│   └── phase3-integer-gelu/
│
├── data/                        # ImageNet val images (git-ignored)
├── docs/                        # project documentation
├── environment.yml
└── pytest.ini
```

---

## Setup

```sh
# Create the environment (once).
conda env create -f environment.yml
conda activate vitquant
```

> **macOS note:** conda numpy and PyTorch may ship conflicting OpenMP runtimes.
> If you see `OMP: Error #15` or `Abort trap: 6`, prefix commands with
> `KMP_DUPLICATE_LIB_OK=TRUE`:
> ```sh
> KMP_DUPLICATE_LIB_OK=TRUE python run_phase1_profiling.py
> ```
> This is harmless — it tells the two OpenMP copies to coexist.

---

## Getting the Data

The experiments need ImageNet-1K validation images under `data/imagenet-val/`.
The downloader streams them from Hugging Face — pull only what you need:

```sh
# 1 024 calibration images for Phase 1 profiling.
python download_imagenet_val.py --num-images 1024

# Full validation set for Phase 2 accuracy sweeps.
python download_imagenet_val.py --num-images 50000
```

The dataset (`ILSVRC/imagenet-1k`) is **gated**: create a free Hugging Face
account, accept the dataset terms, and run `hf auth login` once.

---

## Running the Experiments

```sh
# Phase 1 — profile pre-GELU distributions (needs ~1 024 images).
python run_phase1_profiling.py --num-images 1024

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
# macOS may need the KMP_DUPLICATE_LIB_OK workaround (see Setup above).
KMP_DUPLICATE_LIB_OK=TRUE pytest -m "not slow"

# Full suite.
KMP_DUPLICATE_LIB_OK=TRUE pytest
```
