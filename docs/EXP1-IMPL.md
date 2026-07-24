# EXP1-IMPL: Experiment 1 — Baseline Activation Profiling

> **Status:** ✅ Implemented. `profiler.py` (Welford extension + per-channel std),
> `exp1_profiling.py`, and Phase 1 plotting functions are complete and tested.
> 59/68 fast tests pass (9 failures are pre-existing stubs in Phase 2/3).
> Slow tests require PyTorch 2.2.x + nnsight 0.2.21.
> **Keep this file** as the authoritative implementation reference.

---

## 0. What "done" looks like

```bash
python run_phase1_profiling.py --num-images 1024
```

must produce:

```
outputs/phase1-profiling/
├── profiling_result.json      # all 5 sites × 12 blocks (+ patch_embed)
├── histograms/
│   ├── blocks.0_pre_gelu.png          # reconstructed N(μ,σ²) — labelled as such
│   ├── blocks.0_pre_softmax.png
│   └── ...                            # one PNG per (block, site)
└── per_channel_std_heatmap.png        # requires per_channel_std in LayerStats

# With --spot-batch (optional):
outputs/phase1-profiling/
└── spot_batch_histograms/
    └── blocks.0_pre_gelu.png          # real activation values, shows true tails
```

All fast tests (`pytest -m "not slow"`) must continue to pass. All new Welford
fast tests (Section 3.1) must pass. The two updated slow tests must still pass
on Linux.

---

## 1. Files to implement (in order)

| Step | File | Change |
|------|------|--------|
| **4b-i** | `src/profiler.py` | ✅ ALREADY DONE: `_register_stat_saves` updated (ddof=0, M3, n_samples) |
| **4b-ii** | `src/profiler.py` | Add `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`, `_site_n`, `run_profiling_dataset_pass` |
| **4b-ii** | `tests/test_profiler.py` | Add 3 fast + 2 slow Welford tests |
| **5** | `src/plotting.py` | Implement `plot_activation_histogram`; add `plot_per_channel_std_heatmap` |
| **5** | `src/exp1_profiling.py` | Implement `run(config)` |
| **5** | `tests/test_plotting.py` | Add `test_plot_per_channel_std_heatmap_creates_file` |
| **5** | `run_phase1_profiling.py` | Add `--spot-batch` flag |

---

## 2. Statistical conventions (resolved, non-negotiable)

### 2.1 Population variance throughout (ddof=0)

We are profiling a **fully-observed finite set** of activation values — we are
not estimating an unobserved population parameter from a sample. The correct
statistic is the **population** mean and variance. Bessel's correction (ddof=1)
would introduce a systematic negative bias of `(n-1)/n` with no statistical
justification.

**Rule:** every std and variance computation uses `ddof=0` / `correction=0`.
This is already applied in the updated `_register_stat_saves` in `profiler.py`.
The `merge_batch_stats` function must also use `b_var = b_std ** 2` where
`b_std` is the population std from `batch_stats.std` (already ddof=0).

### 2.2 Exact kurtosis via Pébay (2008) parallel formula

Approximate kurtosis (accumulating per-batch fourth central moments centred at
*batch* means) has no bounded error when batch means vary, and would need a
published error bound before appearing in any table. We use the **exact**
parallel formula from Pébay (2008), *Formulas for Robust, One-Pass Parallel
Computation of Covariances and Arbitrary-Order Statistical Moments*, Sandia
Technical Report SAND2008-6212.

This requires tracking M3 (third central moment sum) and M4 (fourth central
moment sum) across batches. `_register_stat_saves` now saves both M3 and
kurtosis per batch. `merge_batch_stats` uses exact Pébay merges for M3 and M4.

**The Pébay parallel merge formulas** (for combining group A and group B):

Let `n = n_A + n_B`, `δ = μ_B - μ_A`.

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

Where M2, M3, M4 are **sums** (not means): `M_k = Σ(x_i - μ)^k`.

Excess kurtosis at finalisation: `κ = M4 / (n · (M2/n)²) - 3`.

The `merge_batch_stats` function must recover `M3_batch` and `M4_batch` from
the per-batch `LayerStats`:

```
M3_batch = batch_stats.m3                 # stored directly as Σ(x-μ)³
M4_batch = (batch_stats.kurtosis + 3) * (batch_stats.std ** 4) * batch_n
```

---

## 3. Step 4b-ii — add to `src/profiler.py`

The following additions go **after** the existing `ProfilingResult` dataclass.
Do not modify any existing function signatures (they are already updated in
`profiler.py`).

### 3.1 `_site_n` — top-level helper (not a closure)

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
        D: Model embedding dimension (e.g. 768).
        D_mlp: MLP hidden dimension (e.g. 3072 for ViT-B/16).
        num_heads: Number of attention heads (e.g. 12).

    Returns:
        Total number of scalar float elements in the activation tensor
        for this site and batch.

    Note:
        N must be derived as ``patch_embed.num_patches + 1``, not from
        ``input_batch.shape[2]`` (which is the image height, not token count).
    """
    if SITE_PRE_SOFTMAX in site_id or SITE_POST_SOFTMAX in site_id:
        return B * num_heads * N * N
    if SITE_PRE_GELU in site_id:
        return B * N * D_mlp
    # residual_stream, post_layernorm_1, post_layernorm_2
    return B * N * D
```

### 3.2 `WelfordAccumulator` dataclass

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
            ``"{k}_sigma"``.  σ used is the per-batch population std.
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

    All batch statistics must use population conventions (ddof=0), as
    produced by the updated ``_register_stat_saves`` in ``profiler.py``.

    Args:
        acc: Accumulator to update in-place.
        batch_stats: Finalized LayerStats from one call to profile_vit.
            Must have been produced by the updated _register_stat_saves
            (i.e. LayerStats.std is population std, LayerStats.m3 is
            Σ(x−μ)³, LayerStats.kurtosis is exact population excess kurtosis).
        batch_n: Number of scalar elements in this batch for this site.
            Use _site_n() to compute this correctly.

    Raises:
        ValueError: If batch_n <= 0.
    """
    if batch_n <= 0:
        raise ValueError(f"batch_n must be positive, got {batch_n}")

    b_mean = batch_stats.mean
    b_std  = batch_stats.std   # population std (ddof=0), guaranteed by profiler
    b_var  = b_std ** 2        # population variance

    # Recover batch central moment sums from batch_stats.
    # M2_b = population variance * n  = b_var * batch_n
    # M3_b = stored directly as Σ(x−μ)³
    # M4_b = (kurtosis + 3) * σ⁴ * n  (from definition of excess kurtosis)
    M2_b: float = b_var * batch_n
    M3_b: float = batch_stats.m3               # already a sum (not mean)
    M4_b: float = (batch_stats.kurtosis + 3.0) * (b_var ** 2) * batch_n

    n_a: int   = acc.n
    n_b: int   = batch_n
    n_ab: int  = n_a + n_b

    if n_a == 0:
        # First batch: no merge needed, just copy.
        acc.n    = n_b
        acc.mean = b_mean
        acc.M2   = M2_b
        acc.M3   = M3_b
        acc.M4   = M4_b
    else:
        delta: float = b_mean - acc.mean

        # --- Pébay (2008) parallel merge, Eq. (3.1)-(3.4) ---
        # M2
        new_M2 = acc.M2 + M2_b + delta**2 * n_a * n_b / n_ab
        # M3
        new_M3 = (
            acc.M3 + M3_b
            + delta**3 * n_a * n_b * (n_a - n_b) / n_ab**2
            + 3.0 * delta * (n_a * M2_b - n_b * acc.M2) / n_ab
        )
        # M4
        new_M4 = (
            acc.M4 + M4_b
            + delta**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / n_ab**3
            + 6.0 * delta**2 * (n_a**2 * M2_b + n_b**2 * acc.M2) / n_ab**2
            + 4.0 * delta * (n_a * M3_b - n_b * acc.M3) / n_ab
        )
        acc.mean = acc.mean + delta * n_b / n_ab
        acc.M2   = new_M2
        acc.M3   = new_M3
        acc.M4   = new_M4
        acc.n    = n_ab

    # --- Outlier counts: fractions → raw counts, accumulate ---
    for key in acc.outlier_counts:
        frac = batch_stats.outlier_fractions.get(key, 0.0)
        acc.outlier_counts[key] += round(frac * batch_n)
```

### 3.4 `finalize_accumulator`

```python
def finalize_accumulator(acc: WelfordAccumulator) -> LayerStats:
    """Convert a WelfordAccumulator to a final LayerStats.

    All statistics are exact (population conventions, Pébay parallel merge).
    Kurtosis is exact, not approximate.

    Args:
        acc: Fully-populated accumulator (acc.n > 0).

    Returns:
        LayerStats with exact global mean, std, kurtosis, and outlier
        fractions.

    Raises:
        ValueError: If acc.n == 0 (no data was accumulated).
    """
    if acc.n == 0:
        raise ValueError(f"Accumulator '{acc.site_identifier}' has zero elements.")

    global_var: float = acc.M2 / acc.n            # population variance
    global_std: float = math.sqrt(global_var)

    # Exact excess kurtosis: M4/(n·σ⁴) - 3
    global_var_sq = global_var ** 2
    kurtosis: float = (
        acc.M4 / (acc.n * global_var_sq) - 3.0
        if global_var_sq > 0.0 else 0.0
    )

    outlier_fractions: dict[str, float] = {
        key: count / acc.n
        for key, count in acc.outlier_counts.items()
    }

    return LayerStats(
        site_identifier=acc.site_identifier,
        mean=acc.mean,
        std=global_std,
        kurtosis=kurtosis,
        m3=acc.M3,
        outlier_fractions=outlier_fractions,
        n_samples=acc.n,
    )
```

### 3.5 `run_profiling_dataset_pass`

```python
def run_profiling_dataset_pass(
    wrapped_model: NNsight,
    loader: DataLoader,
    device: torch.device,
) -> dict[SiteId, LayerStats]:
    """Collect dataset-wide activation statistics at all 5 sites via exact merge.

    Iterates over all batches in loader, calls profile_vit for each, and
    merges per-batch LayerStats into WelfordAccumulators using the exact
    Pébay (2008) parallel higher-moments formula.

    All five measurement sites (residual_stream, post_layernorm_1/2,
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

    # Extract model architecture constants once before the loop.
    # N = num_patches + 1 (CLS token). For ViT-B/16 on 224×224: N = 197.
    # Do NOT derive N from input_batch.shape[2] (that is image height = 224).
    N: int          = inner_model.patch_embed.num_patches + 1
    D: int          = inner_model.embed_dim
    num_heads: int  = inner_model.blocks[0].attn.num_heads
    D_mlp: int      = inner_model.blocks[0].mlp.fc1.out_features

    accumulators: dict[SiteId, WelfordAccumulator] = {}
    num_batches: int = 0

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        B: int = images.shape[0]  # actual batch size (last batch may be smaller)
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

    logger.info("Finalizing accumulators for %d sites.", len(accumulators))
    return {sid: finalize_accumulator(acc) for sid, acc in accumulators.items()}
```

---

## 4. New tests for `tests/test_profiler.py`

### 4.1 Fast tests (no trace)

```python
def test_welford_accumulator_construction() -> None:
    """WelfordAccumulator initialises with zero-state defaults."""
    from src.profiler import WelfordAccumulator, OUTLIER_SIGMAS
    acc = WelfordAccumulator(site_identifier="blocks.0/pre_gelu")
    assert acc.n == 0
    assert acc.mean == 0.0
    assert acc.M2 == 0.0
    assert acc.M3 == 0.0
    assert acc.M4 == 0.0
    assert set(acc.outlier_counts.keys()) == {f"{k}_sigma" for k in OUTLIER_SIGMAS}
    assert all(v == 0 for v in acc.outlier_counts.values())


def test_merge_batch_stats_single_batch() -> None:
    """After one merge, accumulator mean and population std must match batch."""
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
    # Population variance = std² = 9.0;  M2 = 9.0 * 1000 = 9000.
    assert math.isclose(acc.M2, 9000.0, rel_tol=1e-6)


def test_finalize_accumulator_two_equal_batches() -> None:
    """Two identical batches: global mean and std must match the batch values exactly."""
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
    """run_profiling_dataset_pass must return all 5 sites for every block."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.profiler import run_profiling_dataset_pass, SITE_PRE_SOFTMAX, SITE_POST_SOFTMAX

    # Provide labels so (images, labels) unpacking works correctly.
    images  = torch.randn(4, 3, 224, 224)
    labels  = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=2)
    device  = torch.device("cpu")

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    keys = set(stats.keys())
    assert "patch_embed/residual_stream" in keys
    for i in range(12):
        assert f"blocks.{i}/{SITE_PRE_SOFTMAX}" in keys
        assert f"blocks.{i}/{SITE_POST_SOFTMAX}" in keys


@pytest.mark.slow
def test_slow_run_profiling_dataset_pass_exact_n_samples(_vit_wrapped) -> None:
    """n_samples in finalized LayerStats must equal total elements processed."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.profiler import run_profiling_dataset_pass, SITE_PRE_GELU

    # 4 images, batch_size=2 → 2 batches of B=2.
    # For pre_gelu at ViT-B/16: N=197, D_mlp=3072, so n per batch = 2*197*3072.
    images  = torch.randn(4, 3, 224, 224)
    labels  = torch.zeros(4, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=2)
    device  = torch.device("cpu")

    with torch.no_grad():
        stats = run_profiling_dataset_pass(_vit_wrapped, loader, device)

    key = "blocks.0/pre_gelu"
    expected_n = 4 * 197 * 3072  # total images × N × D_mlp
    assert stats[key].n_samples == expected_n, (
        f"n_samples={stats[key].n_samples}, expected {expected_n}"
    )
```

---

## 5. `src/profiler.py` — `LayerStats` note on `m3` for single-pass use

The existing `profile_vit` single-pass API now populates `m3` and `n_samples`
in `LayerStats`. This is **backward-incompatible** for callers that construct
`LayerStats` by keyword argument and do not pass `m3` or `n_samples`. Both
fields have default values (`m3=0.0`, `n_samples=0`) so existing code that
constructs `LayerStats` without them will still work — but the values will be
wrong. The `_canned_result()` helper in `test_profiler.py` must be updated to
pass `m3=0.0` and `n_samples=0` explicitly so tests remain readable.

Update `_canned_result()`:
```python
stats[key] = LayerStats(
    site_identifier=key,
    mean=float(i) * 0.01,
    std=1.0 + float(i) * 0.05,
    kurtosis=0.5,
    m3=0.0,
    outlier_fractions={f"{s}_sigma": 0.001 for s in OUTLIER_SIGMAS},
    n_samples=0,
)
```

---

## 6. Step 5 — `src/plotting.py`

### 6.1 `plot_activation_histogram`

Signature unchanged. Implementation:
1. `fig, ax = plt.subplots(figsize=(7, 4))`.
2. `ax.hist(activations.ravel(), bins=200, color="steelblue", alpha=0.8)`.
3. If `log_scale`: `ax.set_yscale("log")`.
4. Vertical lines at `mean ± 3σ` (dashed, red) and `mean ± 6σ` (dotted, red),
   computing `mean` and `std` from the input array via `np.mean` / `np.std`.
5. `ax.set_title(layer_name)`, labels, legend, save, `plt.close(fig)`.

### 6.2 `plot_per_channel_std_heatmap`

**Now unblocked** because `profiler.LayerStats` gains `per_channel_std` in
Step 4b-iii (Section 7 below). Stack values into `(num_layers, D)` array,
render with `imshow(..., cmap="viridis")`, colorbar, y-tick layer names, save.

Add smoke test in `tests/test_plotting.py`:
```python
def test_plot_per_channel_std_heatmap_creates_file(tmp_path: Path) -> None:
    from src.plotting import plot_per_channel_std_heatmap
    rng = np.random.default_rng(seed=2)
    stds = {f"blocks.{i}/pre_gelu": rng.random(16).tolist() for i in range(3)}
    plot_per_channel_std_heatmap(stds, tmp_path / "heatmap.png")
    assert (tmp_path / "heatmap.png").exists()
```

---

## 7. Step 4b-iii — `per_channel_std` in `profiler.LayerStats`

### The gap explained

The spec deliverable "per-channel σ heatmaps (layers × channels)" requires
knowing, for every encoder block, the standard deviation of each individual
output channel of the pre-GELU linear layer — not just the scalar std over all
elements. The current `profiler.LayerStats` has no such field.

`hooks.LayerStats` has `per_channel_std: list[float] | None` because
`_update_per_channel` in `hooks.py` maintains running per-channel sums and
sums-of-squares. We need the same for `profiler.LayerStats`.

### What to add

**To `_StatsSavers`:** add `per_channel_std: Any = None` (a nnsight proxy or
`None` if the site doesn't track channels).

**To `LayerStats`:** add `per_channel_std: list[float] | None = None`.

**To `_register_stat_saves`:** add a `track_per_channel: bool = False` arg.
When True, compute per-channel population std:
```python
# tensor_proxy shape: (B, N, D) after reshape from the raw activation tensor.
# We want σ across B and N dims, leaving D.
# Flatten batch and token dims: shape (B*N, D).
t_bn_d = tensor_proxy.reshape(-1, tensor_proxy.shape[-1])
per_channel_std_proxy = t_bn_d.std(dim=0, correction=0).save()  # shape (D,)
```
This produces a proxy of shape `[D]` whose `.value` after the trace is a
1-D tensor of per-channel population stds.

**Sites that need it:** `pre_gelu` (shape `[B, N, D_mlp]`) and
`post_layernorm_1/2` (shape `[B, N, D]`). Pass `track_per_channel=True` for
these sites in `profile_vit`.

**In `_finalize_stats`:** if `savers.per_channel_std is not None`, extract
`per_channel_std=savers.per_channel_std.value.tolist()`.

**In `WelfordAccumulator`:** for sites with `track_per_channel=True`, maintain
`per_channel_M2: list[float]` and `per_channel_n: int`. The per-channel
merge uses the scalar Welford formula applied independently per channel. Add
helper `merge_per_channel_stats(acc, batch_per_channel_std, batch_n)`.

**In `finalize_accumulator`:** compute `per_channel_std = [sqrt(m2/n) for m2
in acc.per_channel_M2]`.

> **Scope note:** this is the most complex addition in Step 4b. It requires
> touching `_StatsSavers`, `LayerStats`, `_register_stat_saves`, `profile_vit`
> (call sites for `pre_gelu` and `post_layernorm_*`), `_finalize_stats`,
> `WelfordAccumulator`, `merge_batch_stats`, and `finalize_accumulator`.
> Implement and test it before `exp1_profiling.py` calls `_plot_per_channel_heatmap`.

---

## 8. Step 5 — `src/exp1_profiling.py`

### 8.1 Imports

```python
import torch
from nnsight import NNsight

from src.config import ProfilingConfig
from src.data_loader import build_val_loader
from src.model import load_vit
from src.profiler import (
    ProfilingResult,
    SiteId,
    LayerStats,
    run_profiling_dataset_pass,
    save_profiling_result,
)
from src.plotting import plot_activation_histogram, plot_per_channel_std_heatmap
from src.utils import ensure_dir
import numpy as np
```

### 8.2 `run(config)` body

```python
def run(config: ProfilingConfig) -> None:
    # 1. Load model and wrap
    model, transform = load_vit(config.device)
    wrapped = NNsight(model)

    # 2. Build loader
    loader = build_val_loader(
        config.data_dir, transform, config.batch_size,
        config.num_images, config.device,
    )

    # 3. Dataset-wide profiling pass (all 5 sites, exact statistics)
    logger.info("Starting profiling pass over %d images...", config.num_images)
    with torch.no_grad():
        stats: dict[SiteId, LayerStats] = run_profiling_dataset_pass(
            wrapped, loader, config.device,
        )

    # 4. Save
    ensure_dir(config.output_dir)
    inner = wrapped._model
    result = ProfilingResult(
        stats=stats,
        num_blocks=len(inner.blocks),
        batch_shape=(config.batch_size, 3, 224, 224),
    )
    json_path = config.output_dir / "profiling_result.json"
    save_profiling_result(result, json_path)
    logger.info("Stats for %d sites saved to %s", len(stats), json_path)

    # 5. Plots
    _plot_histograms(stats, config)
    _plot_per_channel_heatmap(stats, config)
    logger.info("Phase 1 complete. Outputs in %s", config.output_dir)
```

### 8.3 `_plot_histograms`

Histograms are drawn from `N(mean, std²)` synthetic samples. Every title
contains `[reconstructed N(μ,σ²)]`. This is a known limitation: the Gaussian
reconstruction cannot show the heavy tails that kurtosis captures. The
`--spot-batch` path (Section 9) produces real-data histograms.

```python
def _plot_histograms(stats: dict[SiteId, LayerStats], config: ProfilingConfig) -> None:
    hist_dir = config.output_dir / "histograms"
    ensure_dir(hist_dir)
    rng = np.random.default_rng(seed=0)
    for key, s in stats.items():
        synthetic = rng.normal(
            loc=s.mean, scale=max(s.std, 1e-8), size=50_000
        ).astype(np.float32)
        safe_key = key.replace("/", "_").replace(".", "_")
        plot_activation_histogram(
            activations=synthetic,
            layer_name=f"{key}  [reconstructed N(μ,σ²)]",
            output_path=hist_dir / f"{safe_key}.png",
            log_scale=True,
        )
    logger.info("Wrote %d histogram PNGs to %s", len(stats), hist_dir)
```

### 8.4 `_plot_per_channel_heatmap`

```python
def _plot_per_channel_heatmap(stats: dict[SiteId, LayerStats], config: ProfilingConfig) -> None:
    per_channel: dict[str, list[float]] = {
        key: s.per_channel_std
        for key, s in stats.items()
        if s.per_channel_std is not None
    }
    if not per_channel:
        logger.warning(
            "No per_channel_std data found in stats. "
            "Ensure _register_stat_saves is called with track_per_channel=True "
            "for pre_gelu and post_layernorm sites (Step 4b-iii)."
        )
        return
    out_path = config.output_dir / "per_channel_std_heatmap.png"
    plot_per_channel_std_heatmap(per_channel, out_path)
    logger.info("Per-channel σ heatmap written to %s", out_path)
```

---

## 9. `--spot-batch`: real activation histograms (Step 4b)

Add to `run_phase1_profiling.py`:
```python
parser.add_argument(
    "--spot-batch",
    action="store_true",
    default=False,
    help=(
        "After the Welford pass, run one additional forward pass on a single "
        "fixed batch and save real activation histograms under "
        "spot_batch_histograms/. These show the true heavy-tailed distribution, "
        "unlike the reconstructed N(μ,σ²) histograms from the main pass."
    ),
)
```

Pass `spot_batch=args.spot_batch` via `ProfilingConfig` (add the field) or
handle it directly in `main()`. In `exp1_profiling.run()`:

```python
if config.spot_batch:
    _run_spot_batch(wrapped, loader, config)
```

```python
def _run_spot_batch(
    wrapped: NNsight,
    loader: DataLoader,
    config: ProfilingConfig,
) -> None:
    """Run one fixed batch through profile_vit and save real activation histograms.

    This is the only path that produces histograms from real activation values.
    The batch is always the first batch from the loader (deterministic given a
    fixed seed in the DataLoader and seed_everything call in main()).
    """
    from src.profiler import profile_vit
    images, _ = next(iter(loader))
    with torch.no_grad():
        result = profile_vit(wrapped, images.to(config.device))

    spot_dir = config.output_dir / "spot_batch_histograms"
    ensure_dir(spot_dir)
    # profile_vit returns scalar stats only — no raw tensors.
    # To get raw values we need a separate hook or nnsight .save() of the full
    # tensor. This requires a dedicated spot_batch profiler that retains
    # a capped raw sample (e.g. first 50_000 elements) per site.
    # *** See NOTE below — this function body is a placeholder. ***
    logger.warning(
        "--spot-batch is not yet implemented: profile_vit discards raw tensors. "
        "Implement a spot_profile_vit() that saves raw activation samples."
    )
```

> **NOTE on `--spot-batch` implementation:** `profile_vit` does not retain raw
> activation tensors — only scalars. To get real histogram data, a separate
> `spot_profile_vit(wrapped, batch, max_samples_per_site)` function is needed
> that saves a fixed cap of raw values (e.g. `t[:50_000].save()` after
> `reshape(-1)`) for each site. Define this function in `profiler.py` alongside
> `profile_vit`. It is intentionally **separate** from `profile_vit` to ensure
> the main profiling path stays memory-efficient.

---

## 10. Per `run_phase1_profiling.py` — `batch_shape` accuracy

The `ProfilingResult.batch_shape` is currently hardcoded as
`(config.batch_size, 3, 224, 224)` in `run()`. The last batch may be smaller
than `config.batch_size` if `num_images % batch_size != 0`. Since
`batch_shape` is metadata (used for documentation, not computation), set it
from the first actual batch:

```python
# In run(), before the profiling loop:
# Peek at first batch shape for accurate metadata.
first_images, _ = next(iter(loader))
actual_batch_shape = tuple(first_images.shape)
# ... after run_profiling_dataset_pass ...
result = ProfilingResult(
    stats=stats,
    num_blocks=len(inner.blocks),
    batch_shape=actual_batch_shape,
)
```

Note: `build_val_loader` uses `shuffle=False`, so the first batch is
deterministic given the dataset order.

---

## 11. Critical constraints

| Constraint | Enforced by |
|---|---|
| Population std (ddof=0) everywhere | `_register_stat_saves(correction=0)` + `merge_batch_stats` uses `b_var = b_std**2` |
| Exact kurtosis (Pébay 2008) | `WelfordAccumulator` has M3/M4; `merge_batch_stats` uses full formula |
| N derived from `patch_embed.num_patches + 1` | `_site_n` docstring; `run_profiling_dataset_pass` |
| `_site_n` is top-level, not a closure | Defined as module-level function |
| Architecture constants extracted before loop | `N, D, num_heads, D_mlp` outside `for batch_idx...` |
| `torch.no_grad()` wraps dataset loop | `exp1_profiling.run()` |
| `profile_vit` signature unchanged | Tests still pass; only internal callers updated |
| Output is `profiling_result.json` | `save_profiling_result` called in `run()` |
| All logging via `logger` | No bare `print()` |

---

## 12. Test checklist

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
| `test_slow_run_profiling_dataset_pass_site_coverage` | `test_profiler.py` | Slow | ✅ Spec'd (needs PT 2.2) |
| `test_slow_run_profiling_dataset_pass_exact_n_samples` | `test_profiler.py` | Slow | ✅ Spec'd (needs PT 2.2) |
| `test_slow_run_profiling_dataset_pass_per_channel_std_present` | `test_profiler.py` | Slow | ✅ Spec'd (needs PT 2.2) |
| `test_slow_run_profiling_dataset_pass_per_channel_std_shape` | `test_profiler.py` | Slow | ✅ Spec'd (needs PT 2.2) |
| `test_slow_register_saves_finalize_layernorm` | `test_profiler.py` | Slow | Updated (n_samples arg) |
| `test_slow_kurtosis_gaussian` | `test_profiler.py` | Slow | Updated (n_samples arg) |
| `test_plot_activation_histogram_creates_file` | `test_plotting.py` | Fast | ✅ Pass |
| `test_plot_per_channel_std_heatmap_creates_file` | `test_plotting.py` | Fast | ✅ Pass |
| All existing fast tests | full suite | Fast | 59/68 pass (9 stub failures) |
