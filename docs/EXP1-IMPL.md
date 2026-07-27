# EXP1-IMPL: Experiment 1 — Baseline Activation Profiling

> **Status:** Phase 1 complete. All steps implemented and tested.
> 82/91 fast tests pass (9 failures are pre-existing stubs in Phase 2/3).
> 22 slow tests (marked `@pytest.mark.slow`) require nnsight trace context.
> Tested with PyTorch 2.12.1, nnsight 0.7.0, CUDA 13.0, NVIDIA RTX 3070 (8 GB).

---

## 0. What "done" looks like

```bash
# Single run with 1024 images (auto-shuffled for class diversity)
python run_phase1_profiling.py --num-images 1024

# Multi-seed run for variance estimation
python run_phase1_profiling.py --num-images 1024 --num-seeds 3 --seed 42

# Full dataset (50k images, shuffled for class-diverse batches)
python run_phase1_profiling.py --all
```

produces (single seed):

```
outputs/phase1-profiling/
├── profiling_result.json          # all 6 sites × 12 blocks (+ patch_embed)
├── summary_table.csv              # 73 rows × (5 + num_sigma_thresholds) columns
├── histograms/
│   ├── blocks.0_pre_gelu.png      # real activations — blocks 0, 5, 11 only
│   ├── blocks.0_pre_softmax.png
│   ├── blocks.5_pre_gelu.png
│   ├── blocks.5_pre_softmax.png
│   ├── blocks.11_pre_gelu.png
│   ├── blocks.11_pre_softmax.png
│   └── ...                        # one PNG per (selected block, site) — 18 total
├── per_channel_std_heatmap_d768.png   # layernorm sites (D=768)
├── per_channel_std_heatmap_d3072.png  # pre_gelu sites (D=3072)
├── attention_entropy_cls_heatmap.png     # CLS query: 12 blocks × 12 heads
└── attention_entropy_patches_heatmap.png  # patch queries: 12 blocks × 12 heads
```

With `--num-seeds 3`, output is organised as:

```
outputs/phase1-profiling/
├── seed_42/
│   ├── profiling_result.json
│   ├── histograms/ ...
│   ├── per_channel_std_heatmap_d768.png
│   └── per_channel_std_heatmap_d3072.png
├── seed_43/ ...
└── seed_44/ ...
```

All fast tests (`pytest -m "not slow"`) pass with no regressions.

---

## 1. Files and their status

| Step | File | Status |
|------|------|--------|
| 4b-i | `src/profiler.py` | ✅ Done — `_register_stat_saves` (ddof=0, M3, n_samples) |
| 4b-ii | `src/profiler.py` | ✅ Done — `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`, `_site_n`, `run_profiling_dataset_pass` |
| 4b-ii | `tests/test_profiler.py` | ✅ Done — 21 fast + 7 slow Welford tests |
| 4b-iii | `src/profiler.py` | ✅ Done — `per_channel_std` in `LayerStats`, `WelfordAccumulator`, merge pipeline |
| 5 | `src/plotting.py` | ✅ Done — `plot_activation_histogram`, `plot_per_channel_std_heatmap` |
| 5 | `src/exp1_profiling.py` | ✅ Done — `run(config)`, `_plot_per_channel_heatmap` (grouped by channel dim) |
| 5 | `tests/test_plotting.py` | ✅ Done — smoke tests pass |
| 6b | `src/data_loader.py` | ✅ Done — `shuffle: bool \| None = None` with auto-select |
| 6b | `src/profiler.py` | ✅ Done — `histogram_profile_vit` (nnsight 0.7.0 compatible) |
| 6b | `src/exp1_profiling.py` | ✅ Done — `_plot_histograms(wrapped, transform, config, output_dir)` |
| 6b | `tests/test_profiler.py` | ✅ Done — `test_slow_histogram_profile_vit_shapes` |
| 7 | `src/config.py` | ✅ Done — `seed`, `num_seeds` fields |
| 7 | `src/utils.py` | ✅ Done — `log_system_info()` |
| 7 | `run_phase1_profiling.py` | ✅ Done — `--num-seeds`, `--seed` CLI args, system logging |
| 7 | `environment.yml` | ✅ Done — conda environment specification |

---

## 2. Statistical conventions (non-negotiable)

### 2.1 Population std throughout (ddof=0)

We are profiling a fully-observed finite set of activation values, not
estimating an unobserved population. Use `ddof=0` / `correction=0` everywhere.
`merge_batch_stats` uses `b_var = b_std ** 2` where `b_std` is the population
std already stored in `batch_stats.std`.

### 2.2 Exact kurtosis via Pébay (2008)

Use the exact parallel formula from Pébay (2008), *Formulas for Robust,
One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical
Moments*, Sandia SAND2008-6212.

Track M3 and M4 as running sums across batches. Merge formulas (groups A and B,
`n = n_A + n_B`, `δ = μ_B - μ_A`):

```
M2 = M2_A + M2_B + δ² · n_A·n_B / n

M3 = M3_A + M3_B
   + δ³ · n_A·n_B·(n_A - n_B) / n²
   + 3δ · (n_A·M2_B - n_B·M2_A) / n

M4 = M4_A + M4_B
   + δ⁴ · n_A·n_B·(n_A² - n_A·n_B + n_B²) / n³
   + 6δ² · (n_A²·M2_B + n_B²·M2_A) / n²
   + 4δ  · (n_A·M3_B - n_B·M3_A) / n
```

M2, M3, M4 are **sums** (not means): `M_k = Σ(x_i - μ)^k`.

Excess kurtosis at finalisation: `κ = M4 / (n · (M2/n)²) - 3`.

Recover batch moment sums from `LayerStats`:

```
M3_batch = batch_stats.m3
M4_batch = (batch_stats.kurtosis + 3) * (batch_stats.std ** 4) * batch_n
```

### 2.3 Outlier fraction convention

`outlier_fractions` in `LayerStats` are fractions relative to **per-batch σ**,
accumulated across batches. The final reported value is a weighted average of
per-batch outlier rates — not the fraction of elements exceeding k·σ_global.
This is documented in the `WelfordAccumulator.outlier_counts` and
`finalize_accumulator` docstrings in `src/profiler.py`.

---

## 3. `src/profiler.py` — Welford multi-batch API (Step 4b-ii) ✅ Done

The following are already implemented. Shown here as the authoritative reference.

### 3.1 `_site_n`

```python
def _site_n(
    site_id: SiteId,
    B: int,
    N: int,
    D: int,
    D_mlp: int,
    num_heads: int,
) -> int:
    """Return the number of scalar elements for one batch at a given site.

    Args:
        site_id: Site identifier string (e.g. ``"blocks.3/pre_gelu"``).
        B: Batch size (number of images).
        N: Token sequence length including CLS token (e.g. 197 for ViT-B/16).
            Must be derived as ``patch_embed.num_patches + 1``, not from
            ``input_batch.shape[2]`` (which is the image height, not token count).
        D: Model embedding dimension (e.g. 768).
        D_mlp: MLP hidden dimension (e.g. 3072 for ViT-B/16).
        num_heads: Number of attention heads (e.g. 12).

    Returns:
        Total number of scalar float elements in the activation tensor.
    """
    if SITE_PRE_SOFTMAX in site_id or SITE_POST_SOFTMAX in site_id:
        return B * num_heads * N * N
    if SITE_PRE_GELU in site_id:
        return B * N * D_mlp
    # residual_stream, post_layernorm_1, post_layernorm_2
    return B * N * D
```

### 3.2 `WelfordAccumulator`

```python
@dataclass
class WelfordAccumulator:
    """Online running state for one measurement site across all batches.

    Implements exact global statistics via the Pébay (2008) parallel
    higher-moments formula for M2, M3, and M4, enabling exact kurtosis
    without any per-batch centring approximation.

    All Mk values are **sums** (not means): Mk = Σ(x_i − μ)^k.

    Attributes:
        site_identifier: Site key, e.g. ``"blocks.3/pre_softmax"``.
        n: Total scalar elements accumulated across all batches.
        mean: Running global mean (exact, Welford parallel merge).
        M2: Running Σ(x − μ)².
        M3: Running Σ(x − μ)³.
        M4: Running Σ(x − μ)⁴.
        outlier_counts: Raw element counts where |x| > k·σ per key
            ``"{k}_sigma"``.  σ is the per-batch population std; see §2.3.
        per_channel_sum: Per-channel running sum; None if not tracked.
        per_channel_sum_sq: Per-channel running sum of squares; None if not tracked.
        per_channel_n: Total per-channel sample count (B·N accumulated).
    """

    site_identifier: SiteId
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    M3: float = 0.0
    M4: float = 0.0
    outlier_counts: dict[str, int] = field(
        default_factory=lambda: {f"{k}_sigma": 0 for k in OUTLIER_SIGMAS}
    )
    per_channel_sum: list[float] | None = None
    per_channel_sum_sq: list[float] | None = None
    per_channel_n: int = 0
```

### 3.3 `merge_batch_stats`

```python
def merge_batch_stats(
    acc: WelfordAccumulator,
    batch_stats: LayerStats,
    batch_n: int,
) -> None:
    """Update a WelfordAccumulator with statistics from one batch.

    Implements the Pébay (2008) parallel higher-moments merge for exact
    global M2, M3, M4 — and therefore exact std and kurtosis.

    Args:
        acc: Accumulator to update in-place.
        batch_stats: Finalized LayerStats from one call to profile_vit.
            std is population std (ddof=0), m3 is Σ(x−μ)³.
        batch_n: Number of scalar elements in this batch for this site.
            Use _site_n() to compute this correctly.

    Raises:
        ValueError: If batch_n <= 0.
    """
    if batch_n <= 0:
        raise ValueError(f"batch_n must be positive, got {batch_n}")

    b_mean = batch_stats.mean
    b_std  = batch_stats.std   # population std (ddof=0)
    b_var  = b_std ** 2

    M2_b: float = b_var * batch_n
    M3_b: float = batch_stats.m3
    M4_b: float = (batch_stats.kurtosis + 3.0) * (b_var ** 2) * batch_n

    n_a, n_b = acc.n, batch_n
    n_ab = n_a + n_b

    if n_a == 0:
        acc.n = n_b; acc.mean = b_mean
        acc.M2 = M2_b; acc.M3 = M3_b; acc.M4 = M4_b
    else:
        delta = b_mean - acc.mean
        new_M2 = acc.M2 + M2_b + delta**2 * n_a * n_b / n_ab
        new_M3 = (
            acc.M3 + M3_b
            + delta**3 * n_a * n_b * (n_a - n_b) / n_ab**2
            + 3.0 * delta * (n_a * M2_b - n_b * acc.M2) / n_ab
        )
        new_M4 = (
            acc.M4 + M4_b
            + delta**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / n_ab**3
            + 6.0 * delta**2 * (n_a**2 * M2_b + n_b**2 * acc.M2) / n_ab**2
            + 4.0 * delta * (n_a * M3_b - n_b * acc.M3) / n_ab
        )
        acc.mean = acc.mean + delta * n_b / n_ab
        acc.M2 = new_M2; acc.M3 = new_M3; acc.M4 = new_M4; acc.n = n_ab

    for key in acc.outlier_counts:
        frac = batch_stats.outlier_fractions.get(key, 0.0)
        acc.outlier_counts[key] += round(frac * batch_n)

    # Per-channel accumulation (additive sums, no merge formula needed).
    if batch_stats.per_channel_sum is not None and batch_stats.per_channel_sum_sq is not None:
        D_ch = len(batch_stats.per_channel_sum)
        b_per_ch_n = batch_n // D_ch if D_ch > 0 else 0
        if b_per_ch_n > 0:
            if acc.per_channel_sum is None:
                acc.per_channel_sum = list(batch_stats.per_channel_sum)
                acc.per_channel_sum_sq = list(batch_stats.per_channel_sum_sq)
                acc.per_channel_n = b_per_ch_n
            else:
                for c in range(D_ch):
                    acc.per_channel_sum[c] += batch_stats.per_channel_sum[c]
                    acc.per_channel_sum_sq[c] += batch_stats.per_channel_sum_sq[c]
                acc.per_channel_n += b_per_ch_n
```

### 3.4 `finalize_accumulator`

```python
def finalize_accumulator(acc: WelfordAccumulator) -> LayerStats:
    """Convert a WelfordAccumulator to a final LayerStats.

    Args:
        acc: Fully-populated accumulator (acc.n > 0).

    Returns:
        LayerStats with exact global mean, std, kurtosis, and outlier
        fractions. outlier_fractions values are weighted averages of
        per-batch outlier rates (threshold = k·σ_batch), not fractions
        relative to global σ.

    Raises:
        ValueError: If acc.n == 0.
    """
    if acc.n == 0:
        raise ValueError(f"Accumulator '{acc.site_identifier}' has zero elements.")

    global_var = acc.M2 / acc.n
    global_std = math.sqrt(global_var) if global_var > 0.0 else 0.0
    global_var_sq = global_var ** 2
    kurtosis = acc.M4 / (acc.n * global_var_sq) - 3.0 if global_var_sq > 0.0 else 0.0
    outlier_fractions = {key: count / acc.n for key, count in acc.outlier_counts.items()}

    per_channel_std = None
    if acc.per_channel_sum is not None and acc.per_channel_sum_sq is not None and acc.per_channel_n > 0:
        per_channel_std = [
            math.sqrt(max(0.0, sq / acc.per_channel_n - (s / acc.per_channel_n) ** 2))
            for s, sq in zip(acc.per_channel_sum, acc.per_channel_sum_sq)
        ]

    return LayerStats(
        site_identifier=acc.site_identifier,
        mean=acc.mean,
        std=global_std,
        kurtosis=kurtosis,
        m3=acc.M3,
        outlier_fractions=outlier_fractions,
        n_samples=acc.n,
        per_channel_std=per_channel_std,
    )
```

### 3.5 `run_profiling_dataset_pass`

```python
def run_profiling_dataset_pass(
    wrapped_model: NNsight,
    loader: DataLoader,
    device: torch.device,
) -> dict[SiteId, LayerStats]:
    """Collect dataset-wide activation statistics at all 6 sites via exact merge.

    Iterates over all batches in loader, calls profile_vit for each, and
    merges per-batch LayerStats into WelfordAccumulators using the exact
    Pébay (2008) parallel higher-moments formula.

    All six measurement sites (residual_stream, post_layernorm_1, post_layernorm_2,
    pre_gelu, pre_softmax, post_softmax) are covered for every encoder block.

    Must be called inside torch.no_grad() — the caller is responsible.

    Args:
        wrapped_model: NNsight-wrapped VisionTransformer with fused_attn=False.
        loader: DataLoader yielding (images, labels) batches.
        device: Compute device; images are moved here per batch.

    Returns:
        Mapping from site_identifier to finalized global LayerStats.

    Raises:
        ProfilingError: Propagated from profile_vit.
        RuntimeError: If loader yields zero batches.
    """
    inner_model = wrapped_model._model
    # N = num_patches + 1 (CLS token). For ViT-B/16 on 224×224: N = 197.
    N         = inner_model.patch_embed.num_patches + 1
    D         = inner_model.embed_dim
    num_heads = inner_model.blocks[0].attn.num_heads
    D_mlp     = inner_model.blocks[0].mlp.fc1.out_features

    accumulators: dict[SiteId, WelfordAccumulator] = {}
    num_batches = 0

    for _, (images, _) in enumerate(loader):
        images = images.to(device)
        B = images.shape[0]
        batch_result = profile_vit(wrapped_model, images)
        for site_id, layer_stats in batch_result.stats.items():
            batch_n = _site_n(site_id, B, N, D, D_mlp, num_heads)
            if site_id not in accumulators:
                accumulators[site_id] = WelfordAccumulator(site_identifier=site_id)
            merge_batch_stats(accumulators[site_id], layer_stats, batch_n)
        num_batches += 1
        if num_batches % 10 == 0:
            logger.info("Profiled %d batches...", num_batches)

    if num_batches == 0:
        raise RuntimeError("DataLoader yielded zero batches; cannot produce stats.")

    return {sid: finalize_accumulator(acc) for sid, acc in accumulators.items()}
```

---

## 4. `tests/test_profiler.py` — Welford tests (Step 4b-ii) ✅ Done

### 4.1 Fast tests

```python
def test_welford_accumulator_construction() -> None:
    from src.profiler import WelfordAccumulator, OUTLIER_SIGMAS
    acc = WelfordAccumulator(site_identifier="blocks.0/pre_gelu")
    assert acc.n == 0
    assert acc.mean == 0.0
    assert acc.M2 == acc.M3 == acc.M4 == 0.0
    assert set(acc.outlier_counts.keys()) == {f"{k}_sigma" for k in OUTLIER_SIGMAS}
    assert all(v == 0 for v in acc.outlier_counts.values())


def test_merge_batch_stats_single_batch() -> None:
    from src.profiler import WelfordAccumulator, LayerStats, OUTLIER_SIGMAS, merge_batch_stats
    acc = WelfordAccumulator(site_identifier="test/site")
    batch_stats = LayerStats(
        site_identifier="test/site",
        mean=2.0, std=3.0, kurtosis=0.0, m3=0.0,
        outlier_fractions={f"{k}_sigma": 0.01 for k in OUTLIER_SIGMAS},
        n_samples=1000,
    )
    merge_batch_stats(acc, batch_stats, 1000)
    assert acc.n == 1000
    assert math.isclose(acc.mean, 2.0)
    assert math.isclose(acc.M2, 9000.0, rel_tol=1e-6)  # 3²×1000


def test_finalize_accumulator_two_equal_batches() -> None:
    from src.profiler import WelfordAccumulator, LayerStats, OUTLIER_SIGMAS
    from src.profiler import merge_batch_stats, finalize_accumulator
    acc = WelfordAccumulator(site_identifier="test/site")
    for _ in range(2):
        bs = LayerStats(
            site_identifier="test/site",
            mean=4.0, std=2.0, kurtosis=0.0, m3=0.0,
            outlier_fractions={f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS},
            n_samples=100,
        )
        merge_batch_stats(acc, bs, 100)
    result = finalize_accumulator(acc)
    assert math.isclose(result.mean, 4.0, rel_tol=1e-6)
    assert math.isclose(result.std,  2.0, rel_tol=1e-6)
    assert result.n_samples == 200
```

### 4.2 Slow tests

```python
@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_site_coverage(_vit_wrapped) -> None:
    """run_profiling_dataset_pass must return all 6 sites for every block."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.profiler import run_profiling_dataset_pass, SITE_PRE_SOFTMAX, SITE_POST_SOFTMAX

    images  = torch.randn(4, 3, 224, 224)
    labels  = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=2)

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, torch.device("cpu"))

    keys = set(stats.keys())
    assert "patch_embed/residual_stream" in keys
    for i in range(12):
        assert f"blocks.{i}/{SITE_PRE_SOFTMAX}" in keys
        assert f"blocks.{i}/{SITE_POST_SOFTMAX}" in keys


@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_exact_n_samples(_vit_wrapped) -> None:
    """n_samples must equal total elements processed."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.profiler import run_profiling_dataset_pass, SITE_PRE_GELU

    images  = torch.randn(4, 3, 224, 224)
    labels  = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=2)

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, torch.device("cpu"))

    expected_n = 4 * 197 * 3072  # total images × N × D_mlp
    assert stats["blocks.0/pre_gelu"].n_samples == expected_n
```

---

## 5. `src/plotting.py` — Phase 1 functions (Step 5) ✅ Done

### 5.1 `plot_activation_histogram`

1. `fig, ax = plt.subplots(figsize=(7, 4))`.
2. `ax.hist(activations.ravel(), bins=200, color="steelblue", alpha=0.8)`.
3. If `log_scale`: `ax.set_yscale("log")`.
4. Vertical lines at `mean ± 3σ` (dashed red) and `mean ± 6σ` (dotted red),
   computing mean and std from the input array via `np.mean` / `np.std`.
5. Title, axis labels, legend, save to `output_path`, `plt.close(fig)`.

### 5.2 `plot_per_channel_std_heatmap`

Stack per-channel std values into `(num_layers, D)` array, render with
`imshow(..., cmap="viridis")`, colorbar, y-tick layer names, save.

Smoke test:

```python
def test_plot_per_channel_std_heatmap_creates_file(tmp_path: Path) -> None:
    from src.plotting import plot_per_channel_std_heatmap
    rng = np.random.default_rng(seed=2)
    stds = {f"blocks.{i}/pre_gelu": rng.random(16).tolist() for i in range(3)}
    plot_per_channel_std_heatmap(stds, tmp_path / "heatmap.png")
    assert (tmp_path / "heatmap.png").exists()
```

---

## 6. `src/profiler.py` — `per_channel_std` (Step 4b-iii) ✅ Done

`LayerStats` has `per_channel_std: list[float] | None`, `per_channel_sum`,
and `per_channel_sum_sq` fields. `_register_stat_saves` accepts
`track_per_channel: bool = False`. When True, saves per-channel sum and
sum-of-squares proxies (shape `[D]`) for cross-batch merging.

Sites with `track_per_channel=True`: `pre_gelu`, `post_layernorm_1`, `post_layernorm_2`.

`WelfordAccumulator` carries `per_channel_sum` and `per_channel_sum_sq` (additive
across batches — no merge formula needed, sums are additive). `finalize_accumulator`
computes `per_channel_std` from accumulated sums.

---

## 7. `src/exp1_profiling.py` — `run(config)` (Step 5) ✅ Done

### 7.1 Imports

```python
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
from nnsight import NNsight

from src.config import ProfilingConfig
from src.data_loader import build_val_loader
from src.model import load_vit
from src.plotting import plot_activation_histogram, plot_per_channel_std_heatmap
from src.profiler import (
    LayerStats,
    ProfilingResult,
    SiteId,
    histogram_profile_vit,
    run_profiling_dataset_pass,
    save_profiling_result,
)
from src.utils import ensure_dir

logger = logging.getLogger(__name__)
```

### 7.2 `run(config)`

```python
def run(config: ProfilingConfig) -> None:
    # 1. Load model and wrap with NNsight.
    model, transform = load_vit(config.device)
    wrapped = NNsight(model)

    # 2. Build the validation DataLoader (shuffle=False for reproducible order).
    loader = build_val_loader(
        config.data_dir, transform, config.batch_size,
        config.num_images, config.device,
    )

    # 3. Dataset-wide profiling pass (all 6 sites, exact statistics).
    logger.info("Starting profiling pass over %d images...", config.num_images)
    with torch.no_grad():
        stats: dict[SiteId, LayerStats] = run_profiling_dataset_pass(
            wrapped, loader, config.device,
        )

    # 4. Save profiling result.
    ensure_dir(config.output_dir)
    first_images, _ = next(iter(loader))
    inner = wrapped._model
    result = ProfilingResult(
        stats=stats,
        num_blocks=len(inner.blocks),
        batch_shape=tuple(first_images.shape),
    )
    json_path = config.output_dir / "profiling_result.json"
    save_profiling_result(result, json_path)
    logger.info("Stats for %d sites saved to %s", len(stats), json_path)

    # 5. Generate plots.
    _plot_histograms(wrapped, transform, config)
    _plot_per_channel_heatmap(stats, config)
    logger.info("Phase 1 complete. Outputs in %s", config.output_dir)
```

### 7.3 `_plot_histograms` (Step 6b — implement this)

```python
def _plot_histograms(
    wrapped: NNsight,
    transform: Callable,
    config: ProfilingConfig,
    block_indices: tuple[int, ...] = (0, 5, 11),
) -> None:
    """Generate real-activation histograms for selected blocks.

    Runs one additional forward pass using ``histogram_profile_vit`` to
    collect full activation tensors at all six sites for ``block_indices``.
    Histograms show the true distribution including heavy tails.

    Args:
        wrapped: NNsight-wrapped model (already profiled by the Welford pass).
        transform: The preprocessing transform returned by ``load_vit``.
            Passed explicitly so a shuffled loader can be constructed here.
        config: Profiling config (output_dir, device, data_dir, batch_size,
            num_images).
        block_indices: Encoder blocks to generate histograms for.
    """
    # Shuffled loader ensures the histogram batch spans many classes rather
    # than the first alphabetical class(es) in the unshuffled val set.
    # seed_everything(42) is called in main() before run(), so this is
    # deterministic across runs.
    histogram_loader = build_val_loader(
        config.data_dir, transform, config.batch_size,
        config.num_images, config.device, shuffle=True,
    )
    images, _ = next(iter(histogram_loader))
    with torch.no_grad():
        raw_tensors = histogram_profile_vit(
            wrapped, images.to(config.device), block_indices,
        )
    hist_dir = config.output_dir / "histograms"
    ensure_dir(hist_dir)
    for key, tensor in raw_tensors.items():
        activations = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        safe_key = key.replace("/", "_").replace(".", "_")
        plot_activation_histogram(
            activations=activations,
            layer_name=key,
            output_path=hist_dir / f"{safe_key}.png",
            log_scale=True,
        )
    logger.info("Wrote %d real-activation histogram PNGs to %s", len(raw_tensors), hist_dir)
```

### 7.4 `_plot_per_channel_heatmap` ✅ Done

```python
def _plot_per_channel_heatmap(
    stats: dict[SiteId, LayerStats], config: ProfilingConfig,
) -> None:
    per_channel: dict[str, list[float]] = {
        key: s.per_channel_std
        for key, s in stats.items()
        if s.per_channel_std is not None
    }
    if not per_channel:
        logger.warning(
            "No per_channel_std data found. Ensure _register_stat_saves is "
            "called with track_per_channel=True for pre_gelu and post_layernorm sites."
        )
        return
    out_path = config.output_dir / "per_channel_std_heatmap.png"
    plot_per_channel_std_heatmap(per_channel, out_path)
    logger.info("Per-channel σ heatmap written to %s", out_path)
```

---

## 8. Step 6b — three things to implement

### 8.1 `build_val_loader` — add `shuffle` parameter

In `src/data_loader.py`, add `shuffle: bool = False` as the final parameter.
Pass it through to `DataLoader(dataset, ..., shuffle=shuffle, ...)`.
Default `False` preserves all existing Welford-pass behaviour.

### 8.2 `histogram_profile_vit` — add to `src/profiler.py`

```python
def histogram_profile_vit(
    wrapped_model: NNsight,
    input_batch: torch.Tensor,
    block_indices: tuple[int, ...] = (0, 5, 11),
) -> dict[SiteId, torch.Tensor]:
    """Run one forward pass and save full activation tensors for selected blocks.

    Collects real activation tensors at all six measurement sites for the
    specified encoder blocks.  Used to generate histograms showing the true
    heavy-tailed distribution.

    Intentionally separate from ``profile_vit`` so the Welford pipeline
    never retains raw tensors.

    Args:
        wrapped_model: NNsight-wrapped VisionTransformer with fused_attn=False.
        input_batch: Float tensor of shape ``(B, C, H, W)`` on the model device.
        block_indices: Encoder blocks to collect. Default (0, 5, 11) covers
            entry, midpoint, and exit of ViT-B/16.

    Returns:
        Mapping from site_identifier to a CPU float32 tensor of full activations.
        Shapes: ``(B, N, D)`` for residual/layernorm sites, ``(B, N, D_mlp)``
        for pre_gelu, ``(B, H, N, N)`` for pre/post_softmax.

    Raises:
        ProfilingError: If the nnsight trace fails.
        ValueError: If ``input_batch`` is not 4-D.
    """
```

**Implementation notes:**

- Validate `input_batch.ndim == 4`; raise `ValueError` if not.
- Check `wrapped_model._model` has `blocks`; raise `ProfilingError` if not.
- Extract `N`, `D`, `num_heads`, `head_dim`, `D_mlp` from model constants
  (same pattern as `profile_vit`).
- Inside `wrapped_model.trace(input_batch)`, iterate over `block_indices` only.
- For each selected block `i`, save:
  - `residual_stream`: `block.norm1.input[0][0].save()` — key is
    `"patch_embed/residual_stream"` if `i == 0`, else `f"blocks.{i-1}/residual_stream"`.
  - `post_layernorm_1`: `block.norm1.output.save()` — key `f"blocks.{i}/post_layernorm_1"`.
  - `post_layernorm_2`: `block.norm2.output.save()` — key `f"blocks.{i}/post_layernorm_2"`.
  - `pre_gelu`: `block.mlp.act.input[0][0].save()` — key `f"blocks.{i}/pre_gelu"`.
  - `pre_softmax`: reconstruct QKᵀ/√d from `attn.qkv.output` using the same
    split logic as `_register_pre_softmax_saves` (reshape to `(B, N, 3, H, head_dim)`,
    permute to `(3, B, H, N, head_dim)`, compute `q * scale @ k.transpose(-2,-1)`),
    then `.save()` the full `(B, H, N, N)` result — key `f"blocks.{i}/pre_softmax"`.
  - `post_softmax`: `attn.attn_drop.input[0][0].save()` — key `f"blocks.{i}/post_softmax"`.
- After the trace exits, return `{k: v.cpu() for k, v in raw.items()}`.

**Memory budget:**

| Site | Shape (B=64) | Memory |
|------|-------------|--------|
| `residual_stream` | (64, 197, 768) | 38.7 MB |
| `post_layernorm_1` | (64, 197, 768) | 38.7 MB |
| `post_layernorm_2` | (64, 197, 768) | 38.7 MB |
| `pre_gelu` | (64, 197, 3072) | 155 MB |
| `pre_softmax` | (64, 12, 197, 197) | 119 MB |
| `post_softmax` | (64, 12, 197, 197) | 119 MB |
| **Per block** | | **~510 MB** |
| **3 blocks total** | | **~1.5 GB** |

### 8.3 `test_slow_histogram_profile_vit_shapes` — add to `tests/test_profiler.py`

```python
@pytest.mark.slow
def test_slow_histogram_profile_vit_shapes() -> None:
    """histogram_profile_vit returns correct keys and tensor shapes."""
    import timm
    from nnsight import NNsight
    from src.model import disable_fused_attn
    from src.profiler import histogram_profile_vit

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    disable_fused_attn(model)
    wrapped = NNsight(model)

    B = 2
    batch = torch.zeros(B, 3, 224, 224)
    result = histogram_profile_vit(wrapped, batch, block_indices=(0,))

    N    = model.patch_embed.num_patches + 1        # 197
    D    = model.embed_dim                           # 768
    D_mlp = model.blocks[0].mlp.fc1.out_features   # 3072
    H    = model.blocks[0].attn.num_heads           # 12

    expected_shapes = {
        "patch_embed/residual_stream": (B, N, D),
        "blocks.0/post_layernorm_1":   (B, N, D),
        "blocks.0/post_layernorm_2":   (B, N, D),
        "blocks.0/pre_gelu":           (B, N, D_mlp),
        "blocks.0/pre_softmax":        (B, H, N, N),
        "blocks.0/post_softmax":       (B, H, N, N),
    }
    assert set(result.keys()) == set(expected_shapes.keys()), (
        f"Missing keys: {set(expected_shapes) - set(result.keys())}"
    )
    for key, shape in expected_shapes.items():
        assert result[key].shape == torch.Size(shape), (
            f"{key}: expected {shape}, got {tuple(result[key].shape)}"
        )
        assert result[key].device == torch.device("cpu")
        assert result[key].dtype == torch.float32
```

---

## 9. Critical constraints

| Constraint | Enforced by |
|---|---|
| Population std (ddof=0) everywhere | `_register_stat_saves(correction=0)`; `merge_batch_stats` uses `b_var = b_std**2` |
| Exact kurtosis (Pébay 2008) | `WelfordAccumulator` M3/M4; `merge_batch_stats` full formula |
| N from `patch_embed.num_patches + 1` | `_site_n` docstring; both dataset-pass functions |
| Architecture constants extracted before loop | `N, D, num_heads, D_mlp` outside `for` in both functions |
| `torch.no_grad()` wraps dataset loop | `exp1_profiling.run()` |
| `histogram_profile_vit` imported at module level | `exp1_profiling.py` top-of-file imports |
| Histogram tensors converted with `.detach().cpu().numpy()` | `_plot_histograms` |
| Output file is `profiling_result.json` | `save_profiling_result` in `run()` |
| No bare `print()` | All logging via `logger` |
| System info logged at start | `log_system_info()` in `main()` |
| nnsight ≥0.7.0 API compatibility | `.input` returns tensor directly; forward-pass dependency order; `_val()` helper |

---

## 10. Auto-shuffle behaviour

`build_val_loader` accepts `shuffle: bool | None = None`.  When `None` (default):

| `num_images` | `shuffle` | Rationale |
|---|---|---|
| Subset (< full dataset) | `True` | Randomly samples `num_images` indices via `torch.randperm`. Ensures class diversity and enables cross-seed variance. |
| Full dataset (`None` or ≥ dataset size) | `True` | Class-diverse batches produce representative per-batch σ, reducing the outlier-fraction overestimate documented in §2.3. |

Pass an explicit `bool` to override.  The histogram pass always uses `shuffle=True`.

---

## 11. Multi-seed support

`ProfilingConfig` accepts `seed: int = 42` and `num_seeds: int = 1`.

When `num_seeds > 1`, the pipeline runs `num_seeds` independent passes with
seeds `seed`, `seed+1`, ..., `seed+num_seeds-1`.  Each seed:

1. Calls `seed_everything(seed)` to reset all RNGs.
2. Builds a fresh DataLoader (different random subset if shuffling).
3. Runs the full Welford pass + histogram generation.
4. Saves results to `output_dir/seed_{s}/`.

The model is loaded once and reused across all seeds.

Cross-seed variance of key statistics (mean, std, kurtosis) can be computed
from the per-seed `profiling_result.json` files.  With 256 images, statistics
are stable to ~7 decimal places across seeds.

---

## 12. System information logging

`log_system_info()` in `src/utils.py` records at INFO level:
- Python version
- PyTorch version
- CUDA device name and memory (if available)
- nnsight version

Called once at the start of `run_phase1_profiling.main()`.

---

## 13. Environment

Reproducible environment specified in `environment.yml`:

```yaml
name: vit-quant
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - numpy>=1.24
  - matplotlib>=3.7
  - pytest>=7.4
  - pip:
      - torch>=2.5.0
      - torchvision>=0.20.0
      - timm>=1.0.0
      - nnsight>=0.7.0
      - transformers>=4.40.0
      - accelerate>=0.30.0
      - diffusers>=0.28.0
      - einops>=0.8.0
      - sentencepiece>=0.2.0
      - tokenizers>=0.19.0
      - safetensors>=0.4.0
```

Create with: `conda env create -f environment.yml`

Tested on: Python 3.13.13, PyTorch 2.12.1+cu130, nnsight 0.7.0, NVIDIA RTX 3070 (8 GB).

---

## 14. Test checklist

| Test | File | Fast/Slow | Status |
|---|---|---|---|
| `test_welford_accumulator_construction` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_single_batch` | `test_profiler.py` | Fast | ✅ Pass |
| `test_finalize_accumulator_two_equal_batches` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_exact_kurtosis_known_data` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_raises_on_zero_batch_n` | `test_profiler.py` | Fast | ✅ Pass |
| `test_finalize_accumulator_raises_on_zero_n` | `test_profiler.py` | Fast | ✅ Pass |
| `test_site_n_returns_correct_counts` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_outlier_accumulation` | `test_profiler.py` | Fast | ✅ Pass |
| `test_per_channel_merge_two_batches` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_unequal_batch_sizes` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_large_mean_delta` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_zero_variance_batch` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_idempotent` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_kurtosis_laplace` | `test_profiler.py` | Fast | ✅ Pass |
| `test_merge_batch_stats_per_channel_first_batch_none` | `test_profiler.py` | Fast | ✅ Pass |
| `test_site_n_unknown_site_type` | `test_profiler.py` | Fast | ✅ Pass |
| `test_site_n_substring_matching` | `test_profiler.py` | Fast | ✅ Pass |
| `test_load_profiling_result_raises_on_malformed_json` | `test_profiler.py` | Fast | ✅ Pass |
| `test_load_profiling_result_raises_on_missing_keys` | `test_profiler.py` | Fast | ✅ Pass |
| `test_save_profiling_result_overwrites_existing` | `test_profiler.py` | Fast | ✅ Pass |
| `test_profiling_result_batch_shape_preserves_order` | `test_profiler.py` | Fast | ✅ Pass |
| `test_layer_stats_per_channel_fields_default_none` | `test_profiler.py` | Fast | ✅ Pass |
| `test_layer_stats_m3_default_zero` | `test_profiler.py` | Fast | ✅ Pass |
| `test_layer_stats_n_samples_default_zero` | `test_profiler.py` | Fast | ✅ Pass |
| `test_welford_accumulator_outlier_keys_match_outlier_sigmas` | `test_profiler.py` | Fast | ✅ Pass |
| `test_histogram_profile_vit_raises_on_non_4d_input` | `test_profiler.py` | Fast | ✅ Pass |
| `test_histogram_profile_vit_raises_on_model_without_blocks` | `test_profiler.py` | Fast | ✅ Pass |
| `test_build_val_loader_shuffle_default_none` | `test_profiler.py` | Fast | ✅ Pass |
| `test_finalize_accumulator_single_element` | `test_profiler.py` | Fast | ✅ Pass |
| `test_finalize_accumulator_all_constant` | `test_profiler.py` | Fast | ✅ Pass |
| `test_plot_activation_histogram_creates_file` | `test_plotting.py` | Fast | ✅ Pass |
| `test_plot_per_channel_std_heatmap_creates_file` | `test_plotting.py` | Fast | ✅ Pass |
| `test_slow_run_profiling_dataset_pass_site_coverage` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_run_profiling_dataset_pass_exact_n_samples` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_run_profiling_dataset_pass_per_channel_std_present` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_run_profiling_dataset_pass_per_channel_std_shape` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_register_saves_finalize_layernorm` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_kurtosis_gaussian` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_histogram_profile_vit_shapes` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_pre_softmax_reconstruction_matches_manual` | `test_profiler.py` | Slow | ✅ Pass |
| `test_slow_per_channel_std_matches_numpy` | `test_profiler.py` | Slow | ✅ Pass |

---

## Appendix A — nnsight 0.7.0 migration notes

This codebase was originally designed for nnsight 0.2.21 (PyTorch 2.2.x).
It has been migrated to nnsight 0.7.0 (PyTorch ≥2.5).  Three API changes
required code adaptations:

### A.1 `.input` returns the tensor directly

**nnsight <0.3:** `module.input` returned `((tensor,), {})` — a tuple of
`(args, kwargs)`.  Accessing the first positional argument required
`module.input[0][0]`.

**nnsight ≥0.3:** `module.input` is a property with a `@input.preprocess`
decorator that extracts `[*value[0], *value[1].values()][0]` — the first
input tensor directly.  `module.input[0][0]` would index into the tensor's
first two dimensions, producing a slice instead of the full tensor.

**Fix:** Replace `module.input[0][0]` with `module.input` everywhere
(`block.norm1.input`, `block.mlp.act.input`, `attn.attn_drop.input`).

### A.2 Forward-pass dependency ordering

**nnsight ≥0.7:** The interleaver execution model registers "requesters"
for each module access.  When a requester is registered for a module that
has already been called during the forward pass (its output consumed by
downstream operations), the interleaver raises `MissedProviderError`:
"Did you call an Envoy out of order?"

**The correct access order** follows the model's forward pass:

```
norm1.input → norm1.output → qkv.output → attn_drop.input
→ norm2.output → mlp.act.input
```

This differs from the previous order which accessed `norm2.output` and
`mlp.act.input` before `qkv.output`.  The new order is enforced in both
`profile_vit` and `histogram_profile_vit`.

**Reference:** nnsight source — `intervention/interleaver.py` lines 710–730
(``MissedProviderError``), `intervention/envoy.py` lines 194–212
(``.input`` property with preprocess/postprocess decorators).

### A.3 `.save()` returns a concrete tensor

**nnsight <0.3:** `.save()` returned a proxy object.  After the trace
exited, the proxy's `.value` attribute held the concrete tensor.

**nnsight ≥0.3:** `.save()` returns a concrete `torch.Tensor` directly.

**Fix:** `_finalize_stats` uses a `_val()` helper that checks
`isinstance(proxy, torch.Tensor)` first, then falls back to
`proxy.value` for older nnsight versions.  This is robust to both APIs.
