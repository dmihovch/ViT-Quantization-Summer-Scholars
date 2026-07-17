# Repository Structure

This document is the canonical reference for how this codebase is organized and
how new code should be added. Follow it precisely — consistency matters more than
personal preference.

---

## Directory Layout

```
.
├── experiment1-profiling.py          # Phase 1 runner
├── experiment2-characterization.py   # Phase 2 runner
├── experiment3-clipping-ablation.py  # Phase 3 runner
├── experiment4-integer-gelu.py       # Phase 4 runner
├── download_imagenet_val.py          # data download utility (not an experiment)
│
├── src/
│   ├── __init__.py
│   ├── model.py                      # model loading
│   ├── data_loader.py                # datasets and DataLoaders
│   ├── plotting.py                   # all figure generation
│   ├── hooks.py                      # PyTorch forward hook utilities
│   ├── clipping.py                   # pre-GELU clipping strategies
│   ├── integer_gelu.py               # integer polynomial GELU approximation
│   ├── utils.py                      # universal helpers (seeding, logging)
│   ├── exceptions.py                 # custom exception classes
│   │
│   ├── exp1_profiling.py             # Phase 1 logic
│   ├── exp2_characterization.py      # Phase 2 logic
│   ├── exp3_clipping.py              # Phase 3 logic
│   └── exp4_integer_gelu.py          # Phase 4 logic
│
├── tests/
│   ├── conftest.py                   # shared fixtures
│   ├── test_model.py
│   ├── test_data_loader.py
│   ├── test_plotting.py
│   ├── test_hooks.py
│   ├── test_clipping.py
│   ├── test_integer_gelu.py
│   ├── test_utils.py
│   ├── test_exp1.py
│   ├── test_exp2.py
│   ├── test_exp3.py
│   └── test_exp4.py
│
├── outputs/                          # written by experiment runners (git-ignored)
│   ├── exp1-profiling/
│   ├── exp2-characterization/
│   ├── exp3-clipping-ablation/
│   └── exp4-integer-gelu/
│
├── data/                             # ImageNet validation images (git-ignored)
│
├── OVERVIEW.md                       # experimental protocol (source of truth)
├── REPO-STRUCTURE.md                 # this file
├── README.md                         # project summary and quickstart
├── AI-DISCLAIMER.md
├── environment.yml
├── pyrightconfig.json
└── pytest.ini
```

---

## The Golden Rule: Separation of Concerns

```
Root script           →  parses args, builds config, calls src/expN_*.py
src/expN_*.py         →  orchestrates the experiment using shared modules
src/<module>.py       →  does exactly one job; knows nothing about experiments
```

A root script contains **no logic**. It is a thin shell: parse the command line,
construct the config object, hand it to the experiment module, done. If you find
yourself writing a conditional or a loop in a root script, that code belongs in
`src/` instead.

---

## Root-Level Experiment Scripts

**Naming:** `experiment<N>-<experiment-name>.py` where `N` is the phase number
from `OVERVIEW.md` and the name matches the phase title in lowercase with
hyphens.

```
experiment1-profiling.py
experiment2-characterization.py
experiment3-clipping-ablation.py
experiment4-integer-gelu.py
```

**What they do (and only do):**

1. Import `argparse` and the matching `src/expN_*.py` module.
2. Define and call a `parse_config() -> <ExperimentConfig>` function that reads
   `sys.argv` and returns a frozen dataclass.
3. Call the experiment module's `run(config)` entry point.
4. Configure the root logger (`logging.basicConfig`). This is the **only**
   place in the codebase that touches logger configuration.

**What they never do:**

- No `torch`, `timm`, or ML imports.
- No data loading, model loading, or tensor math.
- No `if __name__ == "__main__"` guard required — the script *is* the entry point.

**Template:**

```python
"""
experiment1-profiling.py
========================

Thin runner for Phase 1: pre-GELU activation profiling.
See OVERVIEW.md §Phase 1 for the experimental protocol.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from src import exp1_profiling

BATCH_SIZE_DEFAULT: int = 64
NUM_IMAGES_DEFAULT: int = 1024


@dataclass(frozen=True)
class ProfilingConfig:
    data_dir: Path
    output_dir: Path
    num_images: int
    batch_size: int
    device: torch.device


def parse_config() -> ProfilingConfig:
    parser = argparse.ArgumentParser(
        description="Phase 1: profile pre-GELU activation distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1-profiling"))
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ProfilingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=device,
    )


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
config = parse_config()
exp1_profiling.run(config)
```

---

## `src/` Module Responsibilities

### `src/model.py` — Model Loading

Owns everything required to get a ready-to-use ViT-B/16 instance.

**Exports:**
- `load_vit_b_16(device: torch.device) -> tuple[VisionTransformer, ImageTransform]`
- `evaluate_top1_top5_accuracy(model, loader, device) -> tuple[float, float]`

**Rules:**
- No experiment logic, no hooks, no clipping.
- `load_vit_b_16` returns the model in eval mode, already on `device`.
- `evaluate_top1_top5_accuracy` always wraps forward passes in `torch.no_grad()`.

---

### `src/data_loader.py` — Datasets and DataLoaders

Owns image discovery, the unlabeled dataset (Phases 1–2), and the labeled
ImageFolder loader (Phases 3–4).

**Exports:**
- `build_calibration_loader(image_dir, transform, batch_size, num_images) -> DataLoader`
- `build_validation_loader(data_dir, batch_size, num_images | None) -> DataLoader`

**Rules:**
- Datasets return raw (or transformed) samples only — no labels in `build_calibration_loader`.
- No model imports; `ImageTransform` is the only cross-module type.
- `build_calibration_loader` must be deterministic (no shuffle) so multi-pass
  experiments see identical inputs.

---

### `src/plotting.py` — Figure Generation

Owns every matplotlib figure in the project. Nothing else calls `plt` directly.

**Exports:** one focused function per figure type, e.g.:
- `plot_activation_histogram(data, layer_name, output_path, log_scale=True) -> None`
- `plot_channel_variance_map(variances, layer_name, output_path) -> None`
- `plot_accuracy_ablation(results, output_path) -> None`

**Rules:**
- All functions save to `output_path` and return `None`. They never show windows.
- Always call `matplotlib.use("Agg")` before importing `pyplot` (headless server
  compatibility).
- Always call `plt.close(figure)` after saving to release memory.
- Accepts typed dataclasses or sequences, never raw `dict` objects.

---

### `src/hooks.py` — PyTorch Forward Hook Utilities

Owns the machinery for intercepting pre-GELU tensors via `register_forward_hook`.

**Exports:**
- `register_gelu_hooks(model, recorder) -> list[RemovableHandle]`
- Statistic accumulator classes (e.g. `ActivationStatsAccumulator`)
- Result dataclasses (e.g. `LayerStats`)

**Rules:**
- Hooks must not store raw activation tensors — reduce to scalars on the fly to
  avoid OOM on long runs.
- Always return `RemovableHandle` objects so callers can detach hooks cleanly.
- Hooks target `nn.GELU` modules, not `nn.Linear` (see `OVERVIEW.md §Phase 1`).

---

### `src/clipping.py` — Pre-GELU Clipping Strategies

Owns the four clipping strategies from Phase 3.

**Exports:**
- A `ClipStrategy` enum: `NONE`, `SIGMA_3`, `SIGMA_2`, `ZERO_STRIP`
- `apply_clipping(tensor, strategy, threshold) -> Tensor`
- `ClippedGELU(nn.Module)` — a drop-in replacement that clips then activates
- `patch_model_with_clipping(model, strategy, stats) -> list[RemovableHandle]`

**Rules:**
- `apply_clipping` is a pure function; it never mutates the input tensor.
- `patch_model_with_clipping` returns handles so the caller can unpatch cleanly.
- σ-based strategies require pre-computed `LayerStats` (from `src/hooks.py`);
  never recompute statistics inside this module.

---

### `src/integer_gelu.py` — Integer Polynomial GELU Approximation

Owns the INT8/INT16 piecewise polynomial approximation of GELU (Phase 4).

**Exports:**
- `IntegerGELU(nn.Module)` — drop-in replacement for `nn.GELU`
- `patch_model_with_integer_gelu(model) -> list[nn.Module]` — swaps all GELU
  modules, returns originals for restoration
- `restore_gelu_modules(model, originals) -> None`

**Rules:**
- All arithmetic inside `IntegerGELU.forward` must be expressible in INT8 or INT16.
  Use integer tensor dtypes or explicit bit-shift operations; document the
  precision contract in the docstring.
- No dependency on `src/clipping.py` — the two modules compose at the experiment
  level, not internally.

---

### `src/utils.py` — Universal Helpers

Miscellaneous utilities that don't belong to any single module.

**Exports:**
- `seed_everything(seed: int) -> None` — sets Python, NumPy, and PyTorch seeds
- `get_device() -> torch.device` — returns CUDA if available, else CPU

**Rules:**
- No ML-domain logic; only generic Python/PyTorch utilities.
- `seed_everything` must cover `random`, `numpy.random`, `torch.manual_seed`,
  and `torch.cuda.manual_seed_all`.

---

### `src/exceptions.py` — Custom Exceptions

**Exports custom exception classes used across the project**, e.g.:
- `ShapeMismatchError(ValueError)`
- `DataDirectoryError(FileNotFoundError)`
- `UnsupportedStrategyError(ValueError)`

**Rules:**
- Only define an exception here if it crosses module boundaries or adds
  meaningful semantic clarity over a built-in.
- Every class gets a one-line docstring explaining when it is raised.

---

### `src/expN_*.py` — Experiment-Specific Logic

One file per phase. These are the only modules that import from multiple `src/`
modules and combine them into a pipeline.

**Exports:** a single `run(config) -> None` entry point (plus any supporting
dataclasses used by both the runner script and the module).

**Template structure:**

```python
"""
exp1_profiling.py
=================

Orchestration for Phase 1: pre-GELU activation profiling.
"""

from __future__ import annotations

import logging
from pathlib import Path

# imports from shared src/ modules
from src.model import load_vit_b_16
from src.data_loader import build_calibration_loader
from src.hooks import register_gelu_hooks, ActivationStatsAccumulator
from src import plotting

logger = logging.getLogger(__name__)


def run(config: ProfilingConfig) -> None:  # ProfilingConfig defined in the runner
    """Execute Phase 1 end-to-end."""
    ...
```

**Rules:**
- Never accept raw `dict` as config — always a typed frozen dataclass.
- Use the module-level `logger = logging.getLogger(__name__)` pattern; never
  call `logging.basicConfig` here.
- `run()` must be idempotent: calling it twice with the same config should
  produce the same output directory without crashing.

---

## Config Objects

Every experiment defines its config as a `@dataclass(frozen=True)` in the
corresponding root script. Config objects are:

- Constructed **once** at the entry point.
- Passed through unchanged — no function modifies or re-derives a config.
- Fully typed — every field has an explicit type annotation.
- Free of defaults that hide important decisions. If a hyperparameter changes
  the scientific result, require it explicitly or document the default's rationale.

```python
@dataclass(frozen=True)
class ClippingAblationConfig:
    data_dir: Path
    output_dir: Path
    num_images: int
    batch_size: int
    device: torch.device
    strategies: tuple[ClipStrategy, ...]  # which strategies to sweep
```

---

## Outputs Directory Convention

Each experiment writes its outputs to `outputs/exp<N>-<name>/`. The runner
creates this directory; no module inside `src/` creates directories on its own.

Standard artifacts per experiment:

| Experiment | Artifacts |
|---|---|
| 1 — Profiling | `layer_stats.json`, per-layer histograms (`*.png`) |
| 2 — Characterization | `channel_variance_maps/`, `dominant_channels.json` |
| 3 — Clipping Ablation | `accuracy_results.csv`, `accuracy_ablation.png` |
| 4 — Integer GELU | `accuracy_results.csv`, `cooptimization_delta.json` |

The `outputs/` directory is git-ignored. Commit results by archiving the
relevant subdirectory and documenting it in a report.

---

## Testing Conventions

- One test file per `src/` module: `tests/test_<module>.py`.
- Tests use small, hand-constructed tensors with known answers — never real
  ImageNet data or the downloaded model unless marked `@pytest.mark.slow`.
- Shared fixtures live in `tests/conftest.py`. Keep fixtures minimal.
- Tag any test that downloads the model or processes real data with
  `@pytest.mark.slow`.
- All test functions are fully typed and return `-> None`.

Run the fast suite (no model download):

```sh
pytest -m "not slow"
```

---

## Code Style Checklist

Before committing any file in `src/` or a root script, verify:

- [ ] `from __future__ import annotations` at the top
- [ ] Every function parameter and return value is typed
- [ ] No bare `dict` or `list` without type parameters
- [ ] Config/result structs are dataclasses or TypedDict, not raw dicts
- [ ] All functions have Google-style docstrings
- [ ] Early exits guard all failure modes before any real work
- [ ] `logging` used (not `print`) for all runtime output
- [ ] `torch.no_grad()` explicit in every evaluation/inference path
- [ ] Non-trivial tensor shapes annotated in comments: `# (B, T, C)`
- [ ] `pathlib.Path` used instead of `os.path` strings
- [ ] Formatted with `black`, imports sorted with `isort --profile black`
