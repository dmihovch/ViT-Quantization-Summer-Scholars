# ViT-Quantization-Summer-Scholars

Research codebase for studying **post-training quantization (PTQ)** of a
Vision Transformer (`ViT-B/16`) for edge deployment. The first experiment maps
**activation outliers** across every linear layer to decide which layers are
suitable for mixed-precision (`LLM.int8()`-style) routing and which are not.

The reusable toolchain lives in `src/`; each experiment has a lightweight driver
script at the repository root.

```
.
├── download_imagenet_val.py     # streams validation images from Hugging Face
├── run_experiment1_mapping.py   # Experiment 1 driver (CLI)
├── src/
│   ├── model_utils.py           # load ViT-B/16, tag & iterate layers
│   ├── hooks.py                 # on-the-fly outlier statistics (dataclasses + hooks)
│   ├── data_loader.py           # label-free image dataset + DataLoader
│   └── visualizer.py            # per-layer bar charts
├── tests/                       # the test suite (see below)
├── pytest.ini                   # test configuration
└── environment.yml              # conda environment
```

---

## Setup

```sh
# Create / update the conda environment (installs torch, torchvision,
# matplotlib, pytest, ...).
conda env update -f environment.yml
conda activate vitquant
```

---

## Getting the data

Experiment 1 needs ImageNet-1K validation images under `data/`. The downloader
*streams* them from Hugging Face, so you only pull as many as you ask for
(rather than the full ~6.7 GB validation split):

```sh
# Download 4,096 validation images into data/ (one folder per class).
python download_imagenet_val.py --num-images 4096
```

The official dataset (`ILSVRC/imagenet-1k`) is **gated**: create a free Hugging
Face account, accept the dataset's terms on its page, and run `hf auth login`
once. If access fails, the script prints these exact steps. To use a different
(e.g. non-gated) mirror, pass `--dataset <id>`. See
`python download_imagenet_val.py --help` for all options.

Images are written in `ImageFolder` layout (`data/class_<label>/val_<n>.jpeg`),
which preserves class labels for the later accuracy experiments while still
being picked up by Experiment 1's recursive image search.

---

## Running Experiment 1

How many images to process is a command-line argument, so you can switch
between workflows without editing code:

```sh
# "Debugging" run  - seconds; just checks nothing crashes.
python run_experiment1_mapping.py --num-images 128

# "Characterization" run - the default; generates the per-layer heatmaps.
python run_experiment1_mapping.py            # == --num-images 4096

# "Thesis print" run - the final, full-validation pass.
python run_experiment1_mapping.py --num-images 50000 --batch-size 128
```

Place your images (any nested folder structure) under `data/`. Results are
written to `outputs/exp1_outlier_maps/` (a JSON file plus PNG charts). Run
`python run_experiment1_mapping.py --help` for every option.

---

## The Test Suite

The suite verifies the experiment's logic **without needing real ImageNet data**
and, for most tests, **without downloading the model**. It is built on
[pytest](https://docs.pytest.org/) and lives entirely in `tests/`.

### Why these tests exist

The scientific result depends on the outlier math being correct. Rather than
eyeballing numbers, each metric is checked against a tiny tensor whose answer we
worked out by hand. The suite also guards the "plumbing" (layer tagging, hook
wiring, data loading, CLI parsing) so that a refactor can't silently break the
pipeline.

### Layout

| File | What it covers | Needs the model? |
|------|----------------|------------------|
| `tests/conftest.py` | Shared **fixtures** (synthetic activations, a temp image folder). Not tests themselves. | no |
| `tests/test_model_utils.py` | Layer tagging: attention vs. MLP vs. other. | no |
| `tests/test_hooks.py` | The three outlier metrics, activation extraction, and hook/collector wiring. | no |
| `tests/test_data_loader.py` | Image discovery, dataset shapes, `max_images`, batching, empty-folder error. | no |
| `tests/test_config.py` | Command-line parsing into the immutable `ExperimentConfig`. | no |
| `tests/test_integration.py` | End-to-end: real ViT-B/16, every layer receives data. **Marked `slow`.** | yes |

### How the fast tests stay fast

Most tests build small, hand-constructed tensors with **known** answers. The
central example (defined once in `conftest.py` and reused via the
`synthetic_activations` fixture) is a `[2, 4, 5]` tensor where:

- feature **channel 2** is a persistent outlier (value `10.0` in all 8 tokens), and
- a single spike of `100.0` fixes the maximum magnitude.

That gives exact, assertable answers — e.g. the fixed-threshold outlier density
must be exactly `9 / 40`, and concentrating outliers in one channel must produce
a strictly higher channel-persistence variance than scattering them. No GPU and
no model download are involved, so the whole fast suite runs in ~2 seconds.

### The `slow` marker

`tests/test_integration.py` loads the real model, which downloads ~330 MB of
pretrained weights on the first run. It is tagged `@pytest.mark.slow` so you can
exclude it during quick iteration. The marker is registered in `pytest.ini`
(which also puts the project root on the import path and points pytest at
`tests/`).

### Running the tests

Run all of these from the project root with the `vitquant` environment active.

```sh
# Everything (fast tests + the slow integration test).
pytest

# Quick loop while coding - skip the model download.
pytest -m "not slow"

# Only the slow integration test.
pytest -m "slow"

# Verbose: show each test name and result.
pytest -v

# A single file, or a single test.
pytest tests/test_hooks.py
pytest tests/test_hooks.py::test_fixed_outlier_density_is_exact

# Stop at the first failure and drop into a short traceback.
pytest -x
```

Expected output for the quick loop:

```
26 passed, 1 deselected in ~1.6s
```

### Adding your own tests

1. Create `tests/test_<topic>.py`.
2. Write functions named `test_*`; use `assert` for checks. Annotate parameters
   and return `-> None` to match the project's strict-typing style.
3. Reuse a fixture by adding it as a parameter (e.g.
   `def test_x(synthetic_activations: Tensor) -> None:`). Define new shared
   fixtures in `tests/conftest.py`.
4. Tag anything that loads the model or is otherwise expensive with
   `@pytest.mark.slow`.
5. For exact comparisons of floating-point results, use
   `pytest.approx(expected)`.
