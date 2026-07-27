# Phase 1 Fixes — Implementation Spec

> **Citations:** All literature references in this document are catalogued with
> full bibliographic details in [`docs/CITATIONS.md`](CITATIONS.md).

> **Purpose:** Temporary implementation guide for a coding agent.  
> **Scope:** Four targeted fixes to Phase 1 profiling (F1–F4). Phases 2/3 are untouched.
> **Delete when:** All changes are implemented, tested, and merged.

---

## Context & Findings

The skeptical review of `profiling_result.json` identified three issues:

| # | Severity | Description |
|---|----------|-------------|
| F1 | 🔴 Bug | `--all` runs with `shuffle=False`, producing class-sorted batches. Per-batch σ is underestimated (within-class variance < population variance), making outlier fractions worse than the documented ~5-10% overestimate. |
| F2 | 🟡 Known limitation | Outlier fractions are per-batch-σ, not global-σ fractions. `open-issues.md §10.1` already specifies the fix: a two-pass `run_outlier_counting_pass`. Add it as default, with a `--skip-outlier-recount` flag for fast iteration. |
| F3 | 🟡 Missing | Attention entropy (`H = -Σ p_j log p_j` per head per token at post-softmax) was specified in `vit_profiling_framework.md` and `NEXT-STEPS.md §Step 6` (`plot_attention_entropy_heatmap` planned), but never implemented. |
| F4 | 🟡 Missing | Summary table of kurtosis and outlier-fraction values across all sites and layers is a specified deliverable (`vit_profiling_framework.md §Deliverables`) that has no implementation. Also: `OUTLIER_SIGMAS` in `profiler.py` is `(3.0, 5.0, 8.0)` but the framework spec says `{3, 4, 6}`. This discrepancy must be resolved before the table is generated. |

---

## Pre-existing Specifications to Reuse

Before implementing, verify you have read these sections:

- **F2:** `docs/open-issues.md §10.1` — defines the two-pass approach and names the function `run_outlier_counting_pass`.
- **F3:** `docs/vit_profiling_framework.md §"Per-Site Metrics"` — defines entropy as `H = -Σ_j p_j log p_j`, averaged per head per token across the batch.
- **F3:** `docs/NEXT-STEPS.md §Step 6` — lists `plot_attention_entropy_heatmap(entropies, layer_names, output_path)` as a planned function already in the roadmap.
- **F3:** `docs/vit_profiling_framework.md §"Pre-Softmax vs. Post-Softmax: Why Both Matter"` — documents the role of entropy in detecting attention sinks.
- **F4:** `docs/vit_profiling_framework.md §Deliverables` line 85 — specifies "A single summary table of kurtosis and outlier-fraction values across all sites and layers."
- **F4:** `docs/vit_profiling_framework.md §"Per-Site Metrics"` line 47 — specifies outlier fractions for k ∈ {3, 4, 6} (the original spec). `profiler.py` line 66 uses `(3.0, 5.0, 8.0)`. The agent must use whatever is in `OUTLIER_SIGMAS` at implementation time; do **not** change `OUTLIER_SIGMAS`.

---

## Fix F1 — Always shuffle during the profiling dataset pass

### Problem (traced through code)

`src/data_loader.py`, lines 92-93:
```python
if shuffle is None:
    shuffle = is_subset   # True for subsets, False for full dataset
```

When `num_images=None` (the `--all` case), `is_subset=False`, so `shuffle=False`.
`ImageFolder` then processes images in alphabetical class order: 50 images of
`n00001740`, then 50 of `n00004475`, etc. Each batch of 64 contains at most 2
classes. Per-batch σ is underestimated because within-class variance < population
variance. This compounds the outlier-fraction overestimate in F2.

### Also: remove the stale comment in `EXP1-IMPL.md §0`

`docs/EXP1-IMPL.md` line 19 reads:
```
# Full dataset (50k images, no shuffle needed)
python run_phase1_profiling.py --all
```
This comment asserts "no shuffle needed" — that is wrong. Remove it.

### Simplification

The original rationale for `shuffle=False` on full runs was: "all classes already
covered, shuffle adds no benefit." This is wrong for the profiling use case —
class-homogeneous batches produce biased per-batch statistics. The simplification:
**always shuffle** in `build_val_loader` when called from the profiling pipeline.

Two options; **choose Option A**:

**Option A (recommended): Change the auto-select rule.** Change the `shuffle=None`
default logic so that shuffle **always defaults to True**, regardless of whether
`num_images` is a subset or the full dataset. The full-shuffle case uses
`torch.randperm` seeded by the current RNG state (which is set by
`seed_everything(run_seed)` in `exp1_profiling.run()`).

**Option B (not chosen):** Pass `shuffle=True` explicitly from `_run_single`. This
works but leaves the misleading auto-select logic in place for other callers.

### Changes required

**`src/data_loader.py`**

Change the auto-select logic at lines 92-93:
```python
# OLD — wrong for the profiling use case
if shuffle is None:
    shuffle = is_subset

# NEW — always shuffle by default
if shuffle is None:
    shuffle = True
```

The subsetting logic below (lines 95-111) already handles both cases correctly:
- When `shuffle=True` and `num_images` is a subset: uses `torch.randperm` to
  randomly sample indices (existing behaviour, unchanged).
- When `shuffle=True` and `num_images=None` (full dataset): no subsetting occurs;
  `DataLoader(shuffle=True)` shuffles the full dataset each epoch. This is the new
  behaviour for `--all` runs.

**No other files need changes for F1.** The existing callers that already pass an
explicit `shuffle=` value are unaffected. The `_plot_histograms` call in
`exp1_profiling.py` already passes `shuffle=True` explicitly, so it is unaffected.

**`docs/EXP1-IMPL.md`**

Update `§0` (lines 18-20) and `§10` (the auto-shuffle behaviour table) to remove
the assertion that full-dataset runs do not need shuffling. Specifically:

- Line 19: change `# Full dataset (50k images, no shuffle needed)` to
  `# Full dataset (50k images, shuffled for class-diverse batches)`
- §10 table: update the "Full dataset" row to `shuffle=True` and update the
  rationale column to: "Class-diverse batches produce representative per-batch σ,
  reducing the outlier-fraction overestimate documented in §2.3."

**`docs/open-issues.md`**

Update §10.1 to note that F1 has been addressed: the overestimate is now closer
to the theoretical ~5-10% because per-batch σ is computed over class-diverse
batches. The remaining overestimate (2.3 convention) is tracked by F2.

**`tests/test_data_loader.py`**

The existing test `test_build_val_loader_auto_shuffle_full_dataset` currently
asserts that `is_subset=False` implies `shuffle=False`. This test is now wrong
and must be updated to assert the new behaviour: `shuffle` auto-selects to `True`
for both subsets and full-dataset runs.

**`tests/test_profiler.py`**

The existing test `test_build_val_loader_auto_shuffle_different_seeds` tests that
subset runs with different seeds produce different indices. No change needed there.

---

## Fix F2 — Global-σ outlier fractions via two-pass recount

### Problem

Documented in `docs/open-issues.md §10.1`. Outlier fractions in
`profiling_result.json` are weighted averages of per-batch rates
(threshold = k·σ_batch). The global σ is exact (Pébay merge), but the threshold
used to count outliers during Pass 1 is σ_batch, which is systematically lower.

### Design from open-issues.md

The fix is a second pass that uses exact global μ and σ from the already-completed
Welford accumulation:

```
function run_outlier_counting_pass(wrapped_model, loader, device, finalized_stats)
    for each batch:
        for each site:
            count |x - μ_global| > k·σ_global  for k in OUTLIER_SIGMAS
    → return corrected outlier_fractions per site
```

### Behaviour

- **Default:** the two-pass counting is run automatically after `run_profiling_dataset_pass`.
- **Flag:** `--skip-outlier-recount` on the CLI skips Pass 2. Use this only for
  fast development iteration or for tests that are explicitly not being published.
  Must be documented as producing approximate outlier fractions.

### Changes required

**`src/profiler.py`**

Add a new public function `run_outlier_counting_pass`:

```python
def run_outlier_counting_pass(
    wrapped_model: NNsight,
    loader: DataLoader,
    device: torch.device,
    finalized_stats: dict[SiteId, LayerStats],
) -> dict[SiteId, dict[str, float]]:
    """Second pass: count outlier fractions relative to exact global σ.

    Uses the global mean and std from finalized_stats (produced by
    run_profiling_dataset_pass) as fixed thresholds.  For each site and each
    batch, counts the fraction of elements where |x - μ_global| > k·σ_global
    for k in OUTLIER_SIGMAS.  Accumulates raw counts across all batches and
    returns the final fractions.

    This corrects the per-batch σ overestimate documented in open-issues.md §10.1.

    Args:
        wrapped_model: NNsight-wrapped VisionTransformer (same as Pass 1).
        loader: DataLoader over the same dataset used in Pass 1.
        device: Compute device.
        finalized_stats: dict[SiteId, LayerStats] from finalize_accumulator,
            providing exact global mean and std for each site.

    Returns:
        Mapping from site_identifier to corrected outlier_fractions dict
        (same key format as LayerStats.outlier_fractions: "3.0_sigma", etc.).

    Raises:
        RuntimeError: If loader yields zero batches.
    """
```

Implementation notes:
- Extract architecture constants (N, D, D_mlp, num_heads) from `wrapped_model._model`
  the same way `run_profiling_dataset_pass` does.
- For each site, extract `global_mean = finalized_stats[site_id].mean` and
  `global_std = finalized_stats[site_id].std`.
- Inside the nnsight trace, compute `(t - global_mean).abs() > k * global_std`
  for each k in OUTLIER_SIGMAS and save the fraction (`.float().mean().save()`).
  This is a scalar proxy per site per threshold. Do NOT use `t.std(correction=0)`
  here — that would reintroduce the per-batch-σ problem.
- Accumulate raw element counts across batches (same pattern as `WelfordAccumulator.outlier_counts`).
- At the end, divide each count by total n (from the already-known `finalized_stats[site_id].n_samples`).
- Return `dict[SiteId, dict[str, float]]`.

**`src/profiler.py`** — update `run_profiling_dataset_pass`

Do NOT change `run_profiling_dataset_pass`'s signature. The two-pass logic is
wired in `exp1_profiling._run_single`, not inside `run_profiling_dataset_pass`.
This keeps the Welford function single-responsibility.

**`src/exp1_profiling.py`** — `_run_single`

After `run_profiling_dataset_pass` returns `stats`, add:

```python
if not config.skip_outlier_recount:
    logger.info("Starting outlier recount pass (global-σ fractions)...")
    with torch.no_grad():
        corrected_fractions = run_outlier_counting_pass(
            wrapped, loader, config.device, stats
        )
    # Patch stats in-place: replace per-batch outlier fractions with global-σ ones.
    for site_id, fracs in corrected_fractions.items():
        stats[site_id] = dataclasses.replace(
            stats[site_id], outlier_fractions=fracs
        )
    logger.info("Outlier recount complete.")
else:
    logger.warning(
        "Skipping outlier recount pass (--skip-outlier-recount). "
        "Outlier fractions in output are per-batch-σ approximations."
    )
```

Note: use `dataclasses.replace` (not direct mutation) because `LayerStats` is a
mutable dataclass but mutating a dict value in-place is fragile. Import
`dataclasses` at the top of `exp1_profiling.py`.

**`src/config.py`** — `ProfilingConfig`

Add field:
```python
skip_outlier_recount: bool = False
"""If True, skip the second-pass global-σ outlier recount.
Use only for fast iteration; results will have approximate outlier fractions.
"""
```

**`run_phase1_profiling.py`**

Add CLI argument:
```python
parser.add_argument(
    "--skip-outlier-recount",
    dest="skip_outlier_recount",
    action="store_true",
    help=(
        "Skip the second-pass global-σ outlier recount. "
        "Outlier fractions in the output JSON will be approximate (per-batch-σ). "
        "Use only for fast iteration, not for publishable results."
    ),
)
```
Pass `skip_outlier_recount=args.skip_outlier_recount` to `ProfilingConfig`.

**`docs/open-issues.md`**

When F2 is implemented, resolve §10.1: change status from ⚠️ Known limitation to
✅ Resolved, noting the two-pass approach and `--skip-outlier-recount` flag.

---

## Fix F3 — Attention entropy computation and heatmap

### Pre-existing specification

From `docs/vit_profiling_framework.md §"Per-Site Metrics"`:
> **Attention entropy** (Post-Softmax site only): H = -∑_j p_j log p_j per head
> per token, averaged across the batch; near-zero entropy signals a sink token
> absorbing all probability mass.

From `docs/NEXT-STEPS.md §Step 6`:
> `plot_attention_entropy_heatmap(entropies, layer_names, output_path)` —
> listed as a planned Phase 1 figure function.

The spec is already written. Implement it.

### What entropy measures here

For each head h and each query token position i, entropy is:
`H(i, h) = -Σ_{j=1..N} p_{h,i,j} · log(p_{h,i,j} + ε)`

where ε = 1e-8 (proxy NaN guard — not a bias-correction; for softmax outputs
all p_j ≥ 0 and the guard is only relevant if proxy arithmetic produces a
numerically zero value), and p_{h,i,j} is the post-softmax attention weight
from query token i to key token j in head h.

The attention tensor has shape `(B, H, N, N)` where N = 197 for ViT-B/16.
Row 0 of the N (query) dimension is the **CLS token**. Rows 1..196 are
**patch token** queries.

**CLS and patch queries must be separated.** The literature (Maisonnave et al.
2025; Mali 2025; Lee & Kim 2025; Yadav & Das 2025; documented in
`docs/vit_entropy_methodology.md`) consistently treats CLS-to-patch attention
as a distinct distribution from patch-to-patch attention. Pooling them silently
dilutes sink signals: a head can have near-zero entropy in its CLS row (strong
sink from CLS perspective) while still averaging to moderate entropy if patch
rows are included. That is precisely the signal we are trying to detect.

Compute and store two entropy scalars per head per block:

- **`entropy_cls[h]`** — entropy of the CLS query's attention distribution:
  `H_cls(h) = mean over B of H(i=0, h)` — shape `(H,)` after mean over batch.
- **`entropy_patches[h]`** — mean entropy over all patch query rows:
  `H_patches(h) = mean over B × (N-1) of H(i=1..N-1, h)` — shape `(H,)`.

For ViT-B/16: 12 blocks × 12 heads = 24 stored lists (2 per block) —
`entropy_cls` and `entropy_patches` each of length 12.

### Data structure

Add two fields to `LayerStats`:

```python
attention_entropy_cls: list[float] | None = None
# Per-head Shannon entropy (nats) of the CLS query's attention distribution,
# averaged over the batch dimension only.  Shape: [num_heads].
# None for all sites except post_softmax.
# Cite: Maisonnave et al. 2025 (arXiv:2508.16311); Mali 2025 (arXiv:2511.18925).

attention_entropy_patches: list[float] | None = None
# Per-head mean Shannon entropy (nats) of patch query attention distributions,
# averaged over batch and all N-1 patch query rows (rows 1..N-1).
# Shape: [num_heads].  None for all sites except post_softmax.
# Cite: Maisonnave et al. 2025; Lee & Kim 2025 (10.1109/isocc66390.2025.11329950).
```

The heatmap function accepts either field and renders a (num_blocks × num_heads)
matrix. Two separate heatmaps are generated: one for CLS entropy, one for patch
entropy. This is what the literature requires to detect sink behaviour.

### Changes required

**`src/profiler.py`** — `LayerStats`

Add two fields (both default to `None` for backwards compatibility):
```python
attention_entropy_cls: list[float] | None = None
attention_entropy_patches: list[float] | None = None
```

**`src/profiler.py`** — `_register_stat_saves` and `_StatsSavers`

Do NOT compute entropy inside `_register_stat_saves`. That function operates on a
flattened tensor and does not know about the attention structure. Add a separate
helper `_register_entropy_saves` that returns two proxies:

```python
def _register_entropy_saves(
    attn_weight_proxy: Any,
) -> tuple[Any, Any]:
    """Compute per-head Shannon entropy for CLS and patch queries separately.

    Follows the literature convention of treating CLS-to-all attention and
    patch-to-patch attention as distinct distributions.
    Ref: Maisonnave et al. 2025 (arXiv:2508.16311);
         Mali 2025 (arXiv:2511.18925).

    Args:
        attn_weight_proxy: Proxy or tensor of shape (B, H, N, N).
            Row 0 of the query (dim 2) is the CLS token.
            Rows 1..N-1 are patch token queries.

    Returns:
        (cls_entropy_proxy, patch_entropy_proxy) each of shape (H,).
        cls_entropy_proxy:   mean over B of H(query=CLS, head=h).
        patch_entropy_proxy: mean over B × (N-1) of H(query=patch_i, head=h).
    """
    eps = 1e-8  # proxy NaN guard only; not a bias-correction

    # Per-query entropy: -(p * log(p + eps)).sum(dim=-1) → shape (B, H, N)
    per_query_entropy = -(attn_weight_proxy * (attn_weight_proxy + eps).log()).sum(dim=-1)

    # CLS row: query index 0 → shape (B, H)
    cls_entropy = per_query_entropy[:, :, 0]          # (B, H)
    # Mean over batch → (H,)
    cls_entropy_mean = cls_entropy.mean(dim=0).save()  # (H,)

    # Patch rows: query indices 1..N-1 → shape (B, H, N-1)
    patch_entropy = per_query_entropy[:, :, 1:]        # (B, H, N-1)
    # Sum across batch and patch queries, track total count separately
    # Use sum (not mean) so the accumulator can do sample-count weighting.
    patch_entropy_sum = patch_entropy.sum(dim=(0, 2)).save()  # (H,)

    return cls_entropy_mean, patch_entropy_sum
```

**Accumulation note:** `cls_entropy_mean` is already the batch mean (shape `(H,)`).
`patch_entropy_sum` is the raw sum over B×(N-1) for this batch (shape `(H,)`). The
accumulator tracks both separately and performs sample-count-weighted averaging at
finalization. This is consistent with the element-count weighting used for all
other Welford statistics, and prevents the last (smaller) batch from having 4×
per-image weight.

**`src/profiler.py`** — `_StatsSavers`

Add two fields:
```python
entropy_cls_proxy: Any = None    # shape (H,) batch-mean proxy; non-None for post_softmax
entropy_patch_sum_proxy: Any = None  # shape (H,) batch-sum proxy; non-None for post_softmax
```

**`src/profiler.py`** — `_finalize_stats`

```python
attention_entropy_cls: list[float] | None = None
attention_entropy_patches: list[float] | None = None
if savers.entropy_cls_proxy is not None:
    attention_entropy_cls = _val(savers.entropy_cls_proxy).tolist()
if savers.entropy_patch_sum_proxy is not None and savers.n_samples > 0:
    # n_samples for post_softmax = B * H * N * N; recover B*(N-1) per head.
    # But _finalize_stats does not have direct access to B and N.
    # Instead: store the patch sum as-is in LayerStats.attention_entropy_patches
    # temporarily (it is a sum, not a mean). The WelfordAccumulator will
    # accumulate sums and divide by total (B*(N-1)) at finalize_accumulator time.
    # LayerStats.attention_entropy_patches carries the raw per-batch sum when
    # produced by profile_vit; it carries the global mean when produced by
    # finalize_accumulator. The field name is reused for both stages.
    attention_entropy_patches = _val(savers.entropy_patch_sum_proxy).tolist()
```

Pass `attention_entropy_cls=attention_entropy_cls,
attention_entropy_patches=attention_entropy_patches` to `LayerStats(...)`.

**`src/profiler.py`** — `profile_vit`, inside the trace loop

Replace the `_register_stat_saves(attn.attn_drop.input, ...)` call for
`post_softmax` with a combined call that also registers entropy. To avoid the
nnsight double-access risk on `attn.attn_drop.input`, capture the proxy once and
pass it to both functions:

```python
# --- post_softmax ---
attn_input_proxy = attn.attn_drop.input   # capture once to avoid double .input access
ps_savers = _register_stat_saves(
    attn_input_proxy, f"blocks.{i}/{SITE_POST_SOFTMAX}", n_attn
)
ps_savers.entropy_cls_proxy, ps_savers.entropy_patch_sum_proxy = \
    _register_entropy_saves(attn_input_proxy)
all_savers.append(ps_savers)
```

This accesses `attn.attn_drop.input` exactly once (into `attn_input_proxy`) and
then passes that proxy variable to both functions. `_register_stat_saves` flattens
it; `_register_entropy_saves` uses it with its original shape. No double `.input`
access — the `MissedProviderError` risk is eliminated.

The nnsight dependency order remains:
`norm1.input → norm1.output → qkv.output → attn_drop.input → norm2.output → mlp.act.input`

**`src/profiler.py`** — `WelfordAccumulator`

Add four fields:
```python
entropy_cls_sum: list[float] | None = None
# Per-head CLS entropy sum across batches; shape [H].
# Each batch contributes its batch-mean (not a raw sum). See note below.
entropy_cls_count: int = 0
# Number of batches contributing to entropy_cls_sum.

entropy_patch_sum: list[float] | None = None
# Per-head patch entropy sum (raw sum over B*(N-1) tokens) across all batches.
entropy_patch_count: int = 0
# Total number of (B * (N-1)) patch-token samples accumulated.
```

**`src/profiler.py`** — `merge_batch_stats`

At the end of the function, accumulate entropy. For CLS, accumulate batch means
(B is irrelevant since entropy is already a mean over the batch dimension). For
patch, accumulate raw sums and the token count so we can weight-average correctly.

First, recover `B` and `N` from `batch_n` to compute the patch token count per batch:
```python
# Recover CLS and patch entropy from batch_stats.
if batch_stats.attention_entropy_cls is not None:
    H = len(batch_stats.attention_entropy_cls)
    if acc.entropy_cls_sum is None:
        acc.entropy_cls_sum = list(batch_stats.attention_entropy_cls)
        acc.entropy_cls_count = 1
    else:
        for h in range(H):
            acc.entropy_cls_sum[h] += batch_stats.attention_entropy_cls[h]
        acc.entropy_cls_count += 1

if batch_stats.attention_entropy_patches is not None:
    # attention_entropy_patches from profile_vit carries the raw patch-sum
    # for this batch (sum over B*(N-1) per head).
    # We need the per-batch (B*(N-1)) count to track total samples.
    # batch_n for post_softmax = B*H*N*N.  We need B*(N-1).
    # N is not stored in merge_batch_stats directly, but we can recover it:
    #   batch_n = B * num_heads * N * N  (from _site_n for post_softmax)
    # Without knowing B, H, N separately here, pass patch_token_count explicitly.
    # IMPLEMENTATION NOTE: add a `patch_token_count: int = 0` parameter to
    # merge_batch_stats OR store it alongside the batch_stats. Simplest: add it
    # as an optional keyword argument with default 0 (ignored for non-attn sites).
    H = len(batch_stats.attention_entropy_patches)
    if acc.entropy_patch_sum is None:
        acc.entropy_patch_sum = list(batch_stats.attention_entropy_patches)
        acc.entropy_patch_count = patch_token_count  # see note
    else:
        for h in range(H):
            acc.entropy_patch_sum[h] += batch_stats.attention_entropy_patches[h]
        acc.entropy_patch_count += patch_token_count
```

**`patch_token_count` threading note:** `merge_batch_stats` is called from
`run_profiling_dataset_pass` where `B` is known. The patch token count per batch
is `B * (N - 1)`. Add `patch_token_count: int = 0` as a keyword argument to
`merge_batch_stats`, defaulting to 0. Callers for non-attention sites pass nothing.
Callers for post_softmax sites pass `B * (N - 1)`. This requires a one-line change
to the call in `run_profiling_dataset_pass`:

```python
# In run_profiling_dataset_pass, for each batch:
for site_id, layer_stats in batch_result.stats.items():
    batch_n = _site_n(site_id, B, N, D, D_mlp, num_heads)
    ptc = B * (N - 1) if SITE_POST_SOFTMAX in site_id else 0
    merge_batch_stats(accumulators[site_id], layer_stats, batch_n,
                      patch_token_count=ptc)
```

**`src/profiler.py`** — `finalize_accumulator`

```python
attention_entropy_cls: list[float] | None = None
if acc.entropy_cls_sum is not None and acc.entropy_cls_count > 0:
    # CLS entropy: mean of batch means (each batch contributes equally
    # regardless of batch size, because entropy was already meaned over B
    # inside _register_entropy_saves for the CLS row).
    attention_entropy_cls = [s / acc.entropy_cls_count for s in acc.entropy_cls_sum]

attention_entropy_patches: list[float] | None = None
if acc.entropy_patch_sum is not None and acc.entropy_patch_count > 0:
    # Patch entropy: sample-count-weighted mean across all B*(N-1) tokens.
    attention_entropy_patches = [
        s / acc.entropy_patch_count for s in acc.entropy_patch_sum
    ]
```
Pass both fields to `LayerStats(...)`.

**`src/profiler.py`** — serialisation

`LayerStats` is serialised via `dataclasses.asdict`, which handles both new fields
automatically. `load_profiling_result` uses `LayerStats(**val)` — since both fields
default to `None`, existing JSON files without them will deserialise correctly.

**JSON schema change:** Two new optional fields per `post_softmax` site entry:
```json
"attention_entropy_cls": [1.2, 0.8, ...],     // list[float] length H, or null
"attention_entropy_patches": [3.1, 2.9, ...]  // list[float] length H, or null
```

**`src/plotting.py`** — `plot_attention_entropy_heatmap`

This function is already planned in NEXT-STEPS.md §Step 6. Implement it. Since we
now have two separate entropy fields, the function is called twice from the
orchestrator — once for CLS entropy, once for patch entropy. The function itself
is generic and takes any `dict[str, list[float]]`:

```python
def plot_attention_entropy_heatmap(
    entropies: dict[str, list[float]],
    output_path: Path,
    title: str = "Attention entropy per head (nats)",
) -> None:
    """Save a heatmap of per-head mean attention entropy across blocks.

    Parameters
    ----------
    entropies:
        Mapping from block identifier (e.g. ``"blocks.3/post_softmax"``) to a
        list of per-head mean Shannon entropies in nats. All lists must have
        the same length (num_heads).
    output_path:
        File path where the PNG is written.  Parent directories must exist.
    title:
        Plot title (used to distinguish CLS vs patch heatmaps).
    """
```

Implementation: sort keys, stack rows into `(num_blocks, num_heads)` array, render
with `imshow(..., cmap="viridis")`, label axes (y: block name, x: head index),
colorbar labelled "Mean entropy (nats)", title from `title` parameter,
save, `plt.close(fig)`.

**`src/exp1_profiling.py`** — `_run_single`

Add calls to generate both entropy heatmaps after `_plot_per_channel_heatmap`:

```python
_plot_attention_entropy_heatmaps(stats, output_dir)
```

Add function `_plot_attention_entropy_heatmaps(stats, output_dir)`:
```python
def _plot_attention_entropy_heatmaps(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    cls_entropies = {
        key: s.attention_entropy_cls
        for key, s in stats.items()
        if s.attention_entropy_cls is not None
    }
    patch_entropies = {
        key: s.attention_entropy_patches
        for key, s in stats.items()
        if s.attention_entropy_patches is not None
    }
    if not cls_entropies and not patch_entropies:
        logger.warning("No attention entropy data found in stats.")
        return
    if cls_entropies:
        plot_attention_entropy_heatmap(
            cls_entropies,
            output_dir / "attention_entropy_cls_heatmap.png",
            title="CLS query attention entropy per head (nats)",
        )
    if patch_entropies:
        plot_attention_entropy_heatmap(
            patch_entropies,
            output_dir / "attention_entropy_patches_heatmap.png",
            title="Patch query mean attention entropy per head (nats)",
        )
```

**`src/exp1_profiling.py`** — imports

Add `plot_attention_entropy_heatmap` to the imports from `src.plotting`.

**`docs/EXP1-IMPL.md §0`** — Expected outputs

Add to the expected output tree:
```
├── attention_entropy_cls_heatmap.png     # CLS query: 12 blocks × 12 heads
└── attention_entropy_patches_heatmap.png  # patch queries: 12 blocks × 12 heads
```

---

## Fix F4 — Summary table of kurtosis and outlier fractions

### What the spec requires

From `docs/vit_profiling_framework.md §Deliverables` (line 85):
> A single summary table of kurtosis and outlier-fraction values across all sites and layers.

This is the only Phase 1 deliverable with no implementation anywhere in the codebase.

### OUTLIER_SIGMAS discrepancy — do not change the values

`profiler.py` line 66: `OUTLIER_SIGMAS = (3.0, 5.0, 8.0)`
`vit_profiling_framework.md` line 47: specifies k ∈ {3, 4, 6}
`hooks.py` line 54: `_OUTLIER_SIGMAS = (3, 4, 6)` (legacy, not used in Phase 1)

The values in `profiler.py` have been in production since the original run that
produced the existing `profiling_result.json`. Changing them would invalidate all
previously collected data. **Do not change `OUTLIER_SIGMAS`.** The table will
report fractions at 3σ, 5σ, 8σ. The framework spec document should be updated
separately to reflect the decided values, but that is outside the scope of this
implementation task. The agent must only generate the table using whatever keys
are present in `LayerStats.outlier_fractions` — no hardcoding of sigma values.

### What the table contains

One row per (block, site) combination. Columns:

| Column | Source in `LayerStats` | Notes |
|--------|------------------------|-------|
| `block` | parsed from `site_identifier` | e.g. `0`, `1`, ..., `11`; `patch_embed` for the patch embed site |
| `site` | parsed from `site_identifier` | e.g. `pre_gelu`, `post_softmax` |
| `mean` | `LayerStats.mean` | global mean |
| `std` | `LayerStats.std` | global population std |
| `kurtosis` | `LayerStats.kurtosis` | excess kurtosis (Gaussian = 0) |
| outlier fraction columns | `LayerStats.outlier_fractions` | one column per key, e.g. `frac_3.0_sigma`, `frac_5.0_sigma`, `frac_8.0_sigma` |

Column names for outlier fractions: `f"frac_{key}"` for each key in
`outlier_fractions`, e.g. `"3.0_sigma"` → `"frac_3.0_sigma"`. Do **not**
hardcode the key names — iterate over `outlier_fractions.keys()` so the column
names auto-adapt to whatever `OUTLIER_SIGMAS` contains.

Row ordering: `patch_embed` rows first, then `blocks.0` through `blocks.11`,
with sites within each block in the canonical order:
`["residual_stream", "post_layernorm_1", "pre_softmax", "post_softmax",
  "post_layernorm_2", "pre_gelu"]`.
Use a lookup dict mapping site name to sort index; unknown site names sort
last without raising an error.

### Output format: CSV

Write to `output_dir / "summary_table.csv"`. CSV is machine-readable,
losslessly preserves float values, and can be loaded by pandas, Excel, or the
Phase 2/3 pipeline directly from `profiling_result.json`.

### Changes required

**`src/profiler.py`** — add `generate_summary_table`

```python
def generate_summary_table(
    result: ProfilingResult,
) -> list[dict[str, object]]:
    """Convert a ProfilingResult to a flat list of row dicts for CSV export.

    Each row corresponds to one (block, site) pair. Columns: block, site,
    mean, std, kurtosis, and one column per outlier-fraction key in
    LayerStats.outlier_fractions (column names prefixed with 'frac_').

    Rows are ordered by block (patch_embed first, then 0..11) then by
    canonical site order within each block.

    Column names for outlier fraction keys are derived from the keys in the
    first non-empty outlier_fractions dict encountered. Never hardcode sigma
    values; let the key names drive the column names.

    Args:
        result: Completed ProfilingResult (from run_profiling_dataset_pass
            or loaded via load_profiling_result).

    Returns:
        List of row dicts ordered as described above.
        Suitable for passing to csv.DictWriter.

    Raises:
        ValueError: If result.stats is empty.
    """
```

Implementation notes:
- Parse `site_identifier` by splitting on `"/"` (max 1 split):
  `"blocks.3/pre_gelu"` → `("blocks.3", "pre_gelu")`
  `"patch_embed/residual_stream"` → `("patch_embed", "residual_stream")`
  Do not assume anything about the prefix format beyond this split.
- Block sort key: `"patch_embed"` sorts before any `"blocks.N"`. For
  `"blocks.N"`, extract the integer N and sort numerically. For unknown
  prefixes, sort lexicographically after all known keys.
- Canonical site order lookup dict (site name → int):
  `{"residual_stream": 0, "post_layernorm_1": 1, "pre_softmax": 2,
    "post_softmax": 3, "post_layernorm_2": 4, "pre_gelu": 5}`.
  Unknown site names get sort index 99.
- `block` column value: the raw prefix string before the `/`:
  `"blocks.0"`, `"blocks.1"`, ..., `"patch_embed"`.
- Outlier fraction column names: derived once from the first `LayerStats` that
  has a non-empty `outlier_fractions` dict. All subsequent rows must have the
  same keys; if they do not, the missing key is written as empty string `""`.
- This function does **not** read or write files. It is a pure transformation.

**`src/profiler.py`** — add `save_summary_table`

```python
def save_summary_table(rows: list[dict[str, object]], path: Path) -> None:
    """Write summary table rows to a CSV file.

    Creates parent directories if they do not exist.

    Args:
        rows: Non-empty output of generate_summary_table.
        path: Destination CSV path.

    Raises:
        ValueError: If rows is empty.
    """
```

Implementation: `csv.DictWriter` with `fieldnames=list(rows[0].keys())`,
`extrasaction="raise"`. Write header row. Write all data rows. Floats written
at full Python float precision (no explicit rounding).

**`src/exp1_profiling.py`** — `_run_single`

After `save_profiling_result(result, json_path)`, add:

```python
table_rows = generate_summary_table(result)
table_path = output_dir / "summary_table.csv"
save_summary_table(table_rows, table_path)
logger.info("Summary table (%d rows) written to %s", len(table_rows), table_path)
```

Add `generate_summary_table` and `save_summary_table` to the imports from
`src.profiler`.

**`docs/EXP1-IMPL.md §0`** — Expected outputs

Add to the expected output tree:
```
└── summary_table.csv   # 73 rows × (5 + num_sigma_thresholds) columns
```

---

## Test Plan

All new tests must pass with `pytest -m "not slow"` (fast) or be marked
`@pytest.mark.slow`. The existing 82 passing fast tests must continue to pass with
no regressions.

### F1 — Tests for shuffle change

**Fast tests — `tests/test_data_loader.py`**

1. `test_build_val_loader_auto_shuffle_default_is_true`
   - Create a minimal ImageFolder-compatible directory (use `tmp_path`).
   - Call `build_val_loader(..., shuffle=None)`.
   - Verify the returned DataLoader's `dataset` was constructed with shuffle=True
     by checking `loader.sampler` is not a `SequentialSampler`.
   - **What this tests:** The auto-select logic now defaults to `True`.

2. `test_build_val_loader_auto_shuffle_full_dataset_is_shuffled` (UPDATE existing test)
   - The existing `test_build_val_loader_auto_shuffle_full_dataset` asserts
     `is_subset_none is False` and by implication `shuffle=False`. This assertion
     is now wrong. Update it to assert `shuffle=True` for `num_images=None`.

3. `test_build_val_loader_explicit_false_overrides_auto`
   - Call `build_val_loader(..., shuffle=False)` explicitly.
   - Verify the DataLoader uses sequential sampling.
   - **What this tests:** Explicit overrides still work.

### F2 — Tests for global-σ outlier recount

**Fast tests — `tests/test_profiler.py`**

4. `test_run_outlier_counting_pass_returns_correct_keys`
   - Construct fake `finalized_stats` with known mean/std for 2 fake site IDs.
   - Build a minimal `TensorDataset` and `DataLoader` (no GPU, no real model
     needed — mock `run_outlier_counting_pass` to accept a callable that produces
     fake stats, or restructure so the function accepts pre-computed stats and a
     loader-like iterator).
   - **What this tests:** Return value has the same site keys as input stats.
   - **Implementation note:** If `run_outlier_counting_pass` uses nnsight traces,
     mark as slow. If it can be tested with a mock/direct tensor computation,
     keep it fast.

5. `test_run_outlier_counting_pass_fractions_in_unit_interval`
   - Input: known distribution (e.g. standard normal samples).
   - Expected: all returned outlier fractions ∈ [0, 1].

6. `test_run_outlier_counting_pass_known_gaussian`
   - Create a dataset of samples drawn from N(0, 1) with known n.
   - Expected 3σ fraction: ~0.0027 (±0.002 for finite samples). Tolerance 0.01.
   - **What this tests:** The counting is actually using global μ/σ, not per-batch σ.

7. `test_profiling_config_skip_outlier_recount_default_false`
   - `ProfilingConfig(...).skip_outlier_recount is False`.

8. `test_profiling_config_skip_outlier_recount_can_be_set`
   - `ProfilingConfig(..., skip_outlier_recount=True).skip_outlier_recount is True`.

**Slow tests — `tests/test_profiler.py`**

9. `test_slow_run_outlier_counting_pass_site_coverage` (marked `@pytest.mark.slow`)
   - Run `run_outlier_counting_pass` on a 4-image `TensorDataset` with a real
     nnsight-wrapped ViT (uses `_vit_wrapped` fixture).
   - Verify all 73 site IDs are in the returned dict.

10. `test_slow_run_outlier_counting_pass_global_sigma_lower_than_per_batch`
    (marked `@pytest.mark.slow`)
    - Run `run_profiling_dataset_pass` on a 2-batch loader (8 images total, 2
      batches of 4, each batch from the same Gaussian but different means
      so per-batch σ < global σ).
    - Run `run_outlier_counting_pass` with the resulting finalized stats.
    - Assert that for at least one site, the corrected 3σ outlier fraction is
      strictly less than the per-batch fraction stored in `finalized_stats`.
    - **What this tests:** The two-pass approach actually reduces the overestimate
      when per-batch σ < global σ (the bug condition).

### F3 — Tests for attention entropy

**Fast tests — `tests/test_profiler.py`**

11. `test_layer_stats_attention_entropy_cls_defaults_none`
    - `LayerStats(...).attention_entropy_cls is None`.
    - `LayerStats(...).attention_entropy_patches is None`.

12. `test_welford_accumulator_entropy_fields_default_none`
    - `WelfordAccumulator(...).entropy_cls_sum is None`.
    - `WelfordAccumulator(...).entropy_cls_count == 0`.
    - `WelfordAccumulator(...).entropy_patch_sum is None`.
    - `WelfordAccumulator(...).entropy_patch_count == 0`.

13. `test_merge_batch_stats_entropy_cls_accumulation_first_batch`
    - Construct `LayerStats` with `attention_entropy_cls=[1.5, 2.0]` (2 fake heads),
      `attention_entropy_patches=[6.0, 8.0]` (raw sum for 4 patch tokens).
    - Call `merge_batch_stats(acc, batch_stats, batch_n=100, patch_token_count=4)`.
    - After merge: `acc.entropy_cls_sum == [1.5, 2.0]`, `acc.entropy_cls_count == 1`.
    - `acc.entropy_patch_sum == [6.0, 8.0]`, `acc.entropy_patch_count == 4`.

14. `test_merge_batch_stats_entropy_accumulation_two_batches`
    - Batch 1: `attention_entropy_cls=[1.0, 2.0]`, `attention_entropy_patches=[4.0, 8.0]`,
      `patch_token_count=4`.
    - Batch 2: `attention_entropy_cls=[3.0, 4.0]`, `attention_entropy_patches=[8.0, 12.0]`,
      `patch_token_count=4`.
    - After both merges:
      - `acc.entropy_cls_sum == [4.0, 6.0]`, `acc.entropy_cls_count == 2`.
      - `acc.entropy_patch_sum == [12.0, 20.0]`, `acc.entropy_patch_count == 8`.

15. `test_finalize_accumulator_entropy_cls_mean`
    - After merging as above:
      - `finalize_accumulator(acc).attention_entropy_cls == [2.0, 3.0]`
        (batch-count mean: [4.0, 6.0] / 2).
      - `finalize_accumulator(acc).attention_entropy_patches == [1.5, 2.5]`
        (sample-count mean: [12.0, 20.0] / 8).

16. `test_finalize_accumulator_entropy_none_when_no_data`
    - Accumulator with `entropy_cls_sum=None`, `entropy_patch_sum=None` →
      `finalize_accumulator().attention_entropy_cls is None`.
      `finalize_accumulator().attention_entropy_patches is None`.

17. `test_layer_stats_entropy_serialisation_round_trip`
    - Create `LayerStats` with `attention_entropy_cls=[1.1, 2.2]`,
      `attention_entropy_patches=[3.3, 4.4]`.
    - `save_profiling_result` → `load_profiling_result` round-trip.
    - Verify both fields are preserved exactly (within float repr).

18. `test_layer_stats_entropy_backwards_compat`
    - Deserialise a manually constructed JSON dict that does NOT contain
      `"attention_entropy_cls"` or `"attention_entropy_patches"` keys.
    - Verify both fields default to `None` (old JSON still deserialises).

19. `test_register_entropy_saves_output_shapes`
    - Call `_register_entropy_saves` on a **concrete** tensor of shape (2, 4, 5, 5)
      (B=2, H=4, N=5, so 1 CLS row and 4 patch rows).
    - Expected: `cls_entropy` has shape (4,), all values ≥ 0.
    - Expected: `patch_sum` has shape (4,).
    - All `cls_entropy` values ≤ log(5) ≈ 1.609 (max entropy over 5 key tokens).
    - **Note:** This test calls `_register_entropy_saves` directly on concrete
      tensors (plain PyTorch ops). If proxy-specific API makes this impossible,
      extract a standalone `compute_attention_entropy(tensor)` function and test
      that instead.

20. `test_register_entropy_saves_uniform_attention`
    - Input: `torch.ones(2, 4, 5, 5) / 5.0`.
    - CLS row is uniform over 5 tokens: expected `cls_entropy[h] == log(5)` for all h.
    - Patch rows are uniform: expected `patch_sum[h] == (B*(N-1)) * log(5) == 2*4*log(5)`.
    - Tolerance: 1e-4.

21. `test_register_entropy_saves_peaked_attention`
    - Input: one token gets prob 1.0 for every query, rest 0.0.
    - Expected `cls_entropy` all 0.0, `patch_sum` all 0.0.
    - Tolerance: 1e-6.

22. `test_register_entropy_saves_cls_sink_not_diluted_by_patches`
    - Input: CLS row is peaked (p=1.0 on one token → H_cls=0.0).
      Patch rows are uniform (H_patch = log(N)).
    - Expected: `cls_entropy` all ≈ 0.0, `patch_sum` all > 0.
    - **What this tests:** The separation keeps CLS and patch signals independent.
      If they were pooled, the sink signal in CLS would be diluted by high patch
      entropy. This catches the exact regression the literature analysis warned about.

**Fast tests — `tests/test_plotting.py`**

23. `test_plot_attention_entropy_heatmap_creates_file`
    - Input: `{"blocks.0/post_softmax": [1.0, 1.5, 2.0], "blocks.1/post_softmax": [0.5, 0.8, 1.2]}`.
    - Call `plot_attention_entropy_heatmap(data, tmp_path / "entropy.png")`.
    - Assert file exists.

24. `test_plot_attention_entropy_heatmap_title_in_file`
    - Call with `title="CLS entropy"`. Assert file is created (smoke test; title
      cannot be verified programmatically without image parsing, so this just
      ensures the `title` parameter does not break the function).

25. `test_plot_attention_entropy_heatmap_empty_input`
    - Call with empty dict `{}`.
    - Assert no exception and no file written (or function exits silently with warning).

**Slow tests — `tests/test_profiler.py`**

26. `test_slow_post_softmax_entropy_nonnegative` (marked `@pytest.mark.slow`)
    - Run `profile_vit` on a real ViT with a 2-image batch.
    - For all `post_softmax` sites: `attention_entropy_cls` is not None,
      `attention_entropy_patches` is not None.
    - All values in both lists are ≥ 0.

27. `test_slow_post_softmax_entropy_bounded` (marked `@pytest.mark.slow`)
    - Same setup. CLS values ≤ log(197) ≈ 5.283. Patch values ≤ log(197).
    - (Patch entropy is a mean per head, so it is also bounded by log(197).)

28. `test_slow_run_profiling_dataset_pass_entropy_present` (marked `@pytest.mark.slow`)
    - Run `run_profiling_dataset_pass` on a 4-image TensorDataset.
    - For all `post_softmax` site keys:
      - `stats[key].attention_entropy_cls is not None`.
      - `len(stats[key].attention_entropy_cls) == 12` (num_heads for ViT-B/16).
      - `stats[key].attention_entropy_patches is not None`.
      - `len(stats[key].attention_entropy_patches) == 12`.
    - Non-post_softmax sites: both fields are `None`.

### F4 — Tests for summary table

**Fast tests — `tests/test_profiler.py`**

29. `test_generate_summary_table_row_count`
    - Build a minimal `ProfilingResult` with 3 fake site IDs:
      `"patch_embed/residual_stream"`, `"blocks.0/pre_gelu"`, `"blocks.0/post_softmax"`.
    - Call `generate_summary_table(result)`.
    - Assert `len(rows) == 3`.

30. `test_generate_summary_table_column_names`
    - Use the same 3-site result, with `outlier_fractions = {"3.0_sigma": 0.01, "5.0_sigma": 0.001}`.
    - Assert columns include: `"block"`, `"site"`, `"mean"`, `"std"`, `"kurtosis"`,
      `"frac_3.0_sigma"`, `"frac_5.0_sigma"`.
    - Assert `"frac_8.0_sigma"` is NOT present (key not in this test's outlier_fractions).
    - **What this tests:** Column names are derived from the data, not hardcoded.

31. `test_generate_summary_table_block_site_parsing`
    - Site identifier `"blocks.3/pre_gelu"` → row has `block="blocks.3"`, `site="pre_gelu"`.
    - Site identifier `"patch_embed/residual_stream"` → row has `block="patch_embed"`,
      `site="residual_stream"`.

32. `test_generate_summary_table_ordering`
    - Build a result with sites for blocks 11, 0, 5 and `patch_embed`, each with one site.
    - Assert first row is `patch_embed`, then `blocks.0`, then `blocks.5`, then `blocks.11`.
    - **What this tests:** Numeric block sort, not lexicographic (`blocks.11` ≠ before `blocks.2`).

33. `test_generate_summary_table_canonical_site_order`
    - Build a result with `blocks.0` having sites `pre_gelu`, `residual_stream`, `pre_softmax`.
    - Within the `blocks.0` rows, assert order: `residual_stream` (index 0),
      `pre_softmax` (index 2), `pre_gelu` (index 5).

34. `test_generate_summary_table_values_correct`
    - Build `LayerStats` with known `mean=1.5`, `std=2.0`, `kurtosis=3.0`,
      `outlier_fractions={"3.0_sigma": 0.02}`.
    - Assert `row["mean"] == 1.5`, `row["std"] == 2.0`, `row["kurtosis"] == 3.0`,
      `row["frac_3.0_sigma"] == 0.02`.

35. `test_generate_summary_table_raises_on_empty_stats`
    - `result.stats = {}`.
    - Assert `ValueError` is raised.

36. `test_save_summary_table_creates_file`
    - Call `save_summary_table([{"block": "blocks.0", "site": "pre_gelu", "mean": 1.0}],
      tmp_path / "summary.csv")`.
    - Assert the file exists and is non-empty.

37. `test_save_summary_table_header_row`
    - Call `save_summary_table` with one row containing keys `["block", "site", "mean"]`.
    - Read the file back and assert the first line is `"block,site,mean"`.

38. `test_save_summary_table_round_trip`
    - Write a row with `{"mean": 1.23456789012345}` (high precision float).
    - Read back with `csv.DictReader` and parse the mean column as float.
    - Assert values match within `1e-10` (no lossy rounding).

39. `test_save_summary_table_raises_on_empty_rows`
    - Assert `ValueError` is raised when `rows=[]`.

40. `test_generate_summary_table_full_profiling_result`
    - Build a `ProfilingResult` with all 73 expected sites (12 blocks × 6 sites +
      1 patch_embed site) using minimal `LayerStats` objects.
    - Assert `len(rows) == 73`.
    - Assert the first row is `patch_embed/residual_stream`.
    - Assert the last row is `blocks.11/pre_gelu`.
    - **What this tests:** Complete coverage with a realistic site count.

---

## Constraints and Cross-checks

These must hold after all changes. The coding agent must verify each:

1. **All 82 existing fast tests still pass.** Run `pytest -m "not slow"` before
   and after changes.

2. **`outlier_fractions` key format unchanged.** Keys remain `"3.0_sigma"`,
   `"5.0_sigma"`, `"8.0_sigma"`. The corrected fractions from Pass 2 use the
   same keys.

3. **No changes to `LayerStats` serialisation format for existing fields.** The
   only schema change is adding two optional fields:
   `"attention_entropy_cls": null | list[float]` and
   `"attention_entropy_patches": null | list[float]`.
   Both must deserialise correctly from old JSON files that lack them (backwards
   compatibility via `None` defaults).

4. **`run_profiling_dataset_pass` signature unchanged.** The second pass is
   separate. Do not add arguments to the Welford function.

5. **`profile_vit` return type unchanged.** `ProfilingResult.stats` dict type
   is `dict[SiteId, LayerStats]` — the extended `LayerStats` is backwards
   compatible.

6. **The nnsight dependency order inside `profile_vit` must be preserved.** The
   order for each block is:
   `norm1.input → norm1.output → qkv.output → attn_drop.input → norm2.output → mlp.act.input`
   Adding entropy computation from `attn_drop.input` must not break this order.
   `_register_entropy_saves` on `attn_drop.input` must appear at the same position
   as the existing `_register_stat_saves` on `attn_drop.input` — not after
   `norm2.output`.

7. **`--skip-outlier-recount` is clearly documented as producing approximations.**
   The warning log message must mention this. The CLI help string must mention this.

8. **No bare `print()` statements.** Use `logger.*` throughout.

9. **Population std (ddof=0) in `run_outlier_counting_pass`.** The threshold
   itself uses `global_std` from `finalized_stats` (which is already population std).
   Do not recompute std inside this function.

10. **The `batch_shape` metadata field in `ProfilingResult` is unaffected.**

11. **CLS and patch entropy are always both present or both absent.** For every
    `post_softmax` site, both `attention_entropy_cls` and `attention_entropy_patches`
    must be non-None after `finalize_accumulator`. A state where one is None and
    the other is not-None indicates a bug in `_register_entropy_saves` or
    `merge_batch_stats`.

12. **`merge_batch_stats` signature change is additive only.** The new
    `patch_token_count: int = 0` keyword argument defaults to 0 so all existing
    callers (including all existing tests) work unchanged without modification.

13. **`generate_summary_table` is a pure function — no file I/O, no side effects.**
    `save_summary_table` is the only function that writes files. Tests for
    `generate_summary_table` never touch the filesystem.

14. **`OUTLIER_SIGMAS` must not be changed.** The existing `profiling_result.json`
    was produced with `(3.0, 5.0, 8.0)`. Changing this constant would invalidate
    that data and break all existing tests that depend on outlier fraction key names.
    The summary table column names must derive from `LayerStats.outlier_fractions.keys()`
    at runtime, not from a hardcoded list.

15. **All four fixes produce outputs under the same `output_dir`.** The expected
    output tree after all four fixes is:
    ```
    outputs/phase1-profiling/
    ├── profiling_result.json                  (F2: outlier fractions now global-σ)
    ├── summary_table.csv                      (F4: new)
    ├── histograms/
    │   └── ... (18 PNGs, unchanged)
    ├── per_channel_std_heatmap_d768.png       (unchanged)
    ├── per_channel_std_heatmap_d3072.png      (unchanged)
    ├── attention_entropy_cls_heatmap.png      (F3: new)
    └── attention_entropy_patches_heatmap.png  (F3: new)
    ```
