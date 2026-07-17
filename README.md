# ViT-Quantization-Summer-Scholars

Research codebase for **integer-only edge deployment of Vision Transformers** via
pre-GELU activation clipping. See [`OVERVIEW.md`](OVERVIEW.md) for the full
experimental protocol.

**Target model:** `vit_base_patch16_224` via [`timm`](https://github.com/huggingface/pytorch-image-models)  
**Dataset:** ImageNet-1K validation split

---

## Research Summary

This project investigates the non-linear activation bottleneck for ultra-low-bit
edge deployment of ViTs. The core hypothesis (Dr. Yang's method) is that
**stripping pre-GELU outliers to zero** is a necessary preprocessing step before
replacing the floating-point GELU with an integer-only polynomial approximation,
and that this co-optimization recovers accuracy lost from integer quantization
alone.

The four-phase protocol:

1. **Profiling** – Instrument pre-GELU tensors across all FFN blocks; collect
   distribution statistics (max, mean, variance, kurtosis).
2. **Characterization** – Visualize distributions with log-scale histograms;
   identify dominant outlier channels and σ-based thresholds.
3. **Clipping Ablation** – Sweep four clipping strategies (baseline, 3σ, 2σ,
   zero-strip) and record Top-1/Top-5 accuracy at each.
4. **Integer GELU Integration** – Replace `nn.GELU` with an INT8/INT16
   polynomial approximation; measure the accuracy synergy between clipping and
   integer execution.

---

## Repository Layout

```
.
├── OVERVIEW.md                  # experimental protocol (source of truth)
├── download_imagenet_val.py     # streams validation images from Hugging Face
├── src/
│   ├── model_utils.py           # load ViT-B/16, evaluate Top-1/Top-5 accuracy
│   └── data_loader.py           # image dataset + DataLoader (labeled & unlabeled)
├── tests/
│   ├── conftest.py              # shared fixtures
│   ├── test_model_utils.py      # accuracy evaluation helper tests
│   └── test_data_loader.py      # data loading tests
├── pytest.ini                   # test configuration
└── environment.yml              # conda environment
```

---

## Setup

```sh
conda env update -f environment.yml
conda activate vitquant
```

---

## Getting the data

The experiments need ImageNet-1K validation images under `data/`. The downloader
streams them from Hugging Face so you only pull as many images as you need:

```sh
# Download 1,024 calibration images (stratified by class).
python download_imagenet_val.py --num-images 1024

# Download the full validation set for ablation sweeps.
python download_imagenet_val.py --num-images 50000
```

The dataset (`ILSVRC/imagenet-1k`) is **gated**: create a free Hugging Face
account, accept the dataset terms, and run `hf auth login` once. Images are
written in `ImageFolder` layout (`data/class_<label>/val_<n>.jpeg`).

---

## Running the tests

```sh
# All fast unit tests (no model download required).
pytest -m "not slow"

# Full suite including any slow integration tests.
pytest
```
