"""nnsight-based activation profiler for timm Vision Transformers.

This module replaces the legacy raw-hook approach in the original ``hooks.py``.
It wraps a ``timm`` ``VisionTransformer`` with ``nnsight.NNsight`` and collects
activation statistics at six sites across every encoder block in a single
forward pass, without retaining any full activation tensors in memory.

Measurement sites per block
---------------------------
residual_stream
    The accumulated residual representation entering ``blocks[i].norm1`` —
    i.e. what all previous blocks have built up so far.
post_layernorm_1 / post_layernorm_2
    Output of ``blocks[i].norm1`` (pre-attention) and ``blocks[i].norm2``
    (pre-MLP) respectively.
pre_gelu
    Input to ``blocks[i].mlp.act`` — the pre-activation hidden state inside
    the feed-forward network.
pre_softmax
    The raw scaled QKᵀ attention logit matrix, reconstructed from
    ``blocks[i].attn.qkv.output`` inside the trace context (shape B×H×N×N).
post_softmax
    The attention weight matrix after softmax, captured via
    ``blocks[i].attn.attn_drop.input`` (shape B×H×N×N).

Requirements
------------
* ``fused_attn=False`` must be set on every block's attention module before
  wrapping with ``NNsight``.  With SDPA/FlashAttention enabled (the default
  in timm), the QKᵀ logit matrix is never materialised in memory and cannot
  be captured.  Use :func:`src.model.disable_fused_attn` before wrapping.
* Statistics are computed as scalar proxy expressions inside the trace
  context so that no full activation tensor is ever retained.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from torch.utils.data import DataLoader

import torch
from nnsight import NNsight

from src.exceptions import ProfilingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site key constants
# ---------------------------------------------------------------------------

SITE_RESIDUAL_STREAM: str = "residual_stream"
SITE_POST_LAYERNORM_1: str = "post_layernorm_1"
SITE_POST_LAYERNORM_2: str = "post_layernorm_2"
SITE_PRE_GELU: str = "pre_gelu"
SITE_PRE_SOFTMAX: str = "pre_softmax"
SITE_POST_SOFTMAX: str = "post_softmax"

# Sigma thresholds for outlier fraction computation.
OUTLIER_SIGMAS: tuple[float, ...] = (3.0, 5.0, 8.0)

# Type alias for the site_identifier key format.
SiteId: TypeAlias = str


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Per-site summary statistics for one measurement point in the ViT.

    All values are scalars computed over the full tensor (across batch,
    token, and channel dimensions) for a single forward pass.

    Attributes:
        site_identifier: Unique string key; format ``"{scope}/{site}"``,
            e.g. ``"blocks.3/pre_gelu"`` or ``"blocks.5/pre_softmax"``.
        mean: Global mean over all tensor elements.
        std: Global standard deviation over all tensor elements.
        kurtosis: Excess kurtosis (E[(x−μ)⁴]/σ⁴ − 3). A Gaussian scores 0;
            heavy tails give positive values.
        outlier_fractions: Fraction of elements where |x| > k·σ, for each k
            in OUTLIER_SIGMAS.  Keys are formatted as ``"{k}_sigma"``, e.g.
            ``"3.0_sigma"``, ``"5.0_sigma"``, ``"8.0_sigma"``.
    """

    site_identifier: SiteId
    mean: float
    std: float
    kurtosis: float
    m3: float = 0.0
    outlier_fractions: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    per_channel_std: list[float] | None = None
    per_channel_sum: list[float] | None = None
    per_channel_sum_sq: list[float] | None = None


@dataclass
class ProfilingResult:
    """Complete profiling output for one forward pass of a ViT.

    Attributes:
        stats: Mapping from ``site_identifier`` to :class:`LayerStats`.
        num_blocks: Number of encoder blocks profiled.
        batch_shape: Shape of the input batch used for this trace.
    """

    stats: dict[SiteId, LayerStats]
    num_blocks: int
    batch_shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# Welford multi-batch accumulator (Pébay 2008 exact parallel merge)
# ---------------------------------------------------------------------------


@dataclass
class WelfordAccumulator:
    """Online running state for one measurement site across all batches.

    Implements exact global statistics via the Pébay (2008) parallel
    higher-moments formula for M2, M3, and M4, enabling exact kurtosis
    without any per-batch centring approximation.

    All Mk values are **sums** (not means): ``Mk = Σ(x_i − μ)^k``.

    Attributes:
        site_identifier: Site key, e.g. ``"blocks.3/pre_softmax"``.
        n: Total scalar elements accumulated across all batches.
        mean: Running global mean (exact, Welford parallel merge).
        M2: Running ``Σ(x − μ)²``.
        M3: Running ``Σ(x − μ)³``.
        M4: Running ``Σ(x − μ)⁴``.
        outlier_counts: Raw element counts where ``|x| > k·σ`` per key
            ``"{k}_sigma"``.  σ is the **per-batch** population std, not the
            global std.  Counts are accumulated across batches so the final
            outlier fraction from ``finalize_accumulator`` is a weighted
            average of per-batch outlier rates — not the fraction of elements
            exceeding k·σ_global.
        per_channel_M2: Per-channel running ``Σ(x_c − μ_c)²``; ``None`` if
            this site does not track per-channel statistics.  Shape ``[D]``.
        per_channel_n: Total samples per channel (B·N accumulated).
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
    b_std = batch_stats.std  # population std (ddof=0), guaranteed by profiler
    b_var = b_std**2  # population variance

    # Recover batch central moment sums from batch_stats.
    # M2_b = population variance * n  = b_var * batch_n
    # M3_b = stored directly as Σ(x−μ)³
    # M4_b = (kurtosis + 3) * σ⁴ * n  (from definition of excess kurtosis)
    M2_b: float = b_var * batch_n
    M3_b: float = batch_stats.m3  # already a sum (not mean)
    M4_b: float = (batch_stats.kurtosis + 3.0) * (b_var**2) * batch_n

    n_a: int = acc.n
    n_b: int = batch_n
    n_ab: int = n_a + n_b

    if n_a == 0:
        # First batch: no merge needed, just copy.
        acc.n = n_b
        acc.mean = b_mean
        acc.M2 = M2_b
        acc.M3 = M3_b
        acc.M4 = M4_b
    else:
        delta: float = b_mean - acc.mean

        # --- Pébay (2008) parallel merge, Eq. (3.1)-(3.4) ---
        # M2
        new_M2 = acc.M2 + M2_b + delta**2 * n_a * n_b / n_ab
        # M3
        new_M3 = (
            acc.M3
            + M3_b
            + delta**3 * n_a * n_b * (n_a - n_b) / n_ab**2
            + 3.0 * delta * (n_a * M2_b - n_b * acc.M2) / n_ab
        )
        # M4
        new_M4 = (
            acc.M4
            + M4_b
            + delta**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / n_ab**3
            + 6.0 * delta**2 * (n_a**2 * M2_b + n_b**2 * acc.M2) / n_ab**2
            + 4.0 * delta * (n_a * M3_b - n_b * acc.M3) / n_ab
        )
        acc.mean = acc.mean + delta * n_b / n_ab
        acc.M2 = new_M2
        acc.M3 = new_M3
        acc.M4 = new_M4
        acc.n = n_ab

    # --- Outlier counts: fractions → raw counts, accumulate ---
    for key in acc.outlier_counts:
        frac = batch_stats.outlier_fractions.get(key, 0.0)
        acc.outlier_counts[key] += round(frac * batch_n)

    # --- Per-channel sum accumulation (exact via sum/sum_sq) ---
    if batch_stats.per_channel_sum is not None and batch_stats.per_channel_sum_sq is not None:
        b_per_ch_sum = batch_stats.per_channel_sum
        b_per_ch_sum_sq = batch_stats.per_channel_sum_sq
        D_ch = len(b_per_ch_sum)
        b_per_ch_n = batch_n // D_ch if D_ch > 0 else 0

        if b_per_ch_n > 0:
            if acc.per_channel_sum is None:
                acc.per_channel_sum = list(b_per_ch_sum)
                acc.per_channel_sum_sq = list(b_per_ch_sum_sq)
                acc.per_channel_n = b_per_ch_n
            else:
                for c in range(D_ch):
                    acc.per_channel_sum[c] += b_per_ch_sum[c]
                    acc.per_channel_sum_sq[c] += b_per_ch_sum_sq[c]
                acc.per_channel_n += b_per_ch_n


def finalize_accumulator(acc: WelfordAccumulator) -> LayerStats:
    """Convert a WelfordAccumulator to a final LayerStats.

    All statistics are exact (population conventions, Pébay parallel merge).
    Kurtosis is exact, not approximate.

    Args:
        acc: Fully-populated accumulator (acc.n > 0).

    Returns:
        LayerStats with exact global mean, std, kurtosis, and outlier
        fractions.  Note: ``outlier_fractions`` values are weighted averages
        of per-batch outlier rates (threshold = k·σ_batch), **not** fractions
        relative to global σ.  This is a well-defined statistic but differs
        from the fraction of elements exceeding k·σ_global.

    Raises:
        ValueError: If acc.n == 0 (no data was accumulated).
    """
    if acc.n == 0:
        raise ValueError(f"Accumulator '{acc.site_identifier}' has zero elements.")

    global_var: float = acc.M2 / acc.n  # population variance
    global_std: float = math.sqrt(global_var) if global_var > 0.0 else 0.0

    # Exact excess kurtosis: M4/(n·σ⁴) - 3
    global_var_sq = global_var**2
    kurtosis: float = (
        acc.M4 / (acc.n * global_var_sq) - 3.0 if global_var_sq > 0.0 else 0.0
    )

    outlier_fractions: dict[str, float] = {
        key: count / acc.n for key, count in acc.outlier_counts.items()
    }

    per_channel_std: list[float] | None = None
    if acc.per_channel_sum is not None and acc.per_channel_sum_sq is not None and acc.per_channel_n > 0:
        per_channel_std = [
            math.sqrt(max(0.0, sum_sq / acc.per_channel_n - (s / acc.per_channel_n) ** 2))
            for s, sum_sq in zip(acc.per_channel_sum, acc.per_channel_sum_sq)
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

    # Extract model architecture constants once before the loop.
    # N = num_patches + 1 (CLS token). For ViT-B/16 on 224×224: N = 197.
    # Do NOT derive N from input_batch.shape[2] (that is image height = 224).
    N: int = inner_model.patch_embed.num_patches + 1
    D: int = inner_model.embed_dim
    num_heads: int = inner_model.blocks[0].attn.num_heads
    D_mlp: int = inner_model.blocks[0].mlp.fc1.out_features

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


# ---------------------------------------------------------------------------
# Internal two-phase statistics helpers
# ---------------------------------------------------------------------------

# nnsight proxy values are only available *after* the trace context exits.
# We register .save() calls inside the context and store the proxy objects
# in _StatsSavers, then convert them to floats via _finalize_stats afterward.


@dataclass
class _StatsSavers:
    """Holds nnsight .save() proxies for one measurement site.

    All attributes are nnsight proxy objects during the trace and become
    concrete Python values after the trace context exits.

    Attributes:
        site_identifier: Site key string set at registration time.
        mean: Saved mean proxy.
        std: Saved std proxy.
        kurtosis: Saved excess-kurtosis proxy.
        outlier_proxies: Ordered list of saved outlier-fraction proxies,
            one per entry in OUTLIER_SIGMAS.
    """

    site_identifier: SiteId
    mean: Any
    std: Any
    m3: Any
    kurtosis: Any
    outlier_proxies: list[Any]
    n_samples: int
    per_channel_sum: Any = None
    per_channel_sum_sq: Any = None


def _register_stat_saves(
    tensor_proxy: Any, site_id: SiteId, n_samples: int,
    track_per_channel: bool = False,
) -> _StatsSavers:
    """Register all statistics as .save() calls inside a nnsight trace context.

    Must be called from within a ``with wrapped_model.trace(...):`` block.
    All arithmetic is performed on nnsight proxy objects — no real tensors
    are materialised at this point.

    All statistics use **population** conventions (ddof=0) throughout:
    - std  = sqrt(E[(x−μ)²])
    - M3   = Σ(x−μ)³  (third central moment sum, for exact cross-batch merge)
    - kurtosis = E[(x−μ)⁴]/σ⁴ − 3  (excess, population)

    Using population statistics is correct here because we are measuring a
    fully-observed finite activation tensor, not estimating an unobserved
    population parameter.  Bessel's correction (ddof=1) would introduce a
    systematic negative bias of (n−1)/n with no statistical justification.

    Outlier fractions use the population σ as threshold scale, recomputed
    inline for each threshold to avoid reusing a stale saved proxy value.

    Args:
        tensor_proxy: An nnsight proxy pointing to an activation tensor.
            May have any shape; all stats are computed over all elements.
        site_id: Human-readable identifier stored verbatim in LayerStats.
        n_samples: Number of scalar elements in this tensor (B*N*D etc.).
            Must be passed explicitly because proxies do not expose .numel()
            reliably before the trace executes.
        track_per_channel: If True, also compute and save per-channel
            population std across batch and token dims (shape ``[D]``).
            Only valid when the tensor has a channel dimension as its last
            axis (e.g. ``(B, N, D)`` or ``(B, N, D_mlp)``).

    Returns:
        A :class:`_StatsSavers` holding the registered save proxies.
    """
    # Flatten so all reductions are unambiguously over every element.
    t = tensor_proxy.reshape(-1)

    mean_proxy = t.mean().save()
    # Population std: correction=0 → sqrt(mean((x-μ)²))
    std_proxy = t.std(correction=0).save()

    # Central moments — all recompute t.mean() inline because a .save()
    # proxy value is frozen after the trace and cannot re-enter proxy graphs.
    centred = t - t.mean()
    variance = (centred ** 2).mean()          # population variance = σ²
    m3_proxy = (centred ** 3).sum().save()    # M3 = Σ(x−μ)³  (sum, not mean)
    fourth_moment = (centred ** 4).mean()     # E[(x−μ)⁴] = μ₄
    # Excess kurtosis: μ₄/σ⁴ − 3.  Guard divide-by-zero: if σ²=0 the
    # distribution is a point mass; kurtosis is undefined, stored as 0.
    kurtosis_proxy = (fourth_moment / (variance ** 2) - 3.0).save()

    outlier_proxies: list[Any] = []
    for sigma in OUTLIER_SIGMAS:
        # Recompute population std inline for each threshold.
        frac = (t.abs() > sigma * t.std(correction=0)).float().mean().save()
        outlier_proxies.append(frac)

    # Per-channel population std: reduce over batch and token dims.
    per_channel_sum_proxy: Any = None
    per_channel_sum_sq_proxy: Any = None
    if track_per_channel:
        # tensor_proxy shape: (B, N, D) — flatten B and N, keep D.
        t_bn_d = tensor_proxy.reshape(-1, tensor_proxy.shape[-1])
        per_channel_sum_proxy = t_bn_d.sum(dim=0).save()       # shape (D,)
        per_channel_sum_sq_proxy = (t_bn_d**2).sum(dim=0).save()  # shape (D,)

    return _StatsSavers(
        site_identifier=site_id,
        mean=mean_proxy,
        std=std_proxy,
        m3=m3_proxy,
        kurtosis=kurtosis_proxy,
        outlier_proxies=outlier_proxies,
        n_samples=n_samples,
        per_channel_sum=per_channel_sum_proxy,
        per_channel_sum_sq=per_channel_sum_sq_proxy,
    )


def _finalize_stats(savers: _StatsSavers) -> LayerStats:
    """Convert saved proxy values to a concrete :class:`LayerStats`.

    Must be called *after* the nnsight trace context exits, at which point
    all ``.save()`` proxy objects hold concrete values.

    Compatible with both nnsight <0.3 (proxy objects with ``.value``
    attribute) and nnsight ≥0.3 (``.save()`` returns concrete tensors).

    Args:
        savers: Populated :class:`_StatsSavers` from a completed trace.

    Returns:
        A fully populated :class:`LayerStats` instance.
    """
    def _val(proxy: Any) -> torch.Tensor:
        """Extract concrete tensor from nnsight proxy or raw tensor.

        nnsight <0.3: .save() returns a proxy object whose .value attribute
        holds the concrete tensor after the trace exits.
        nnsight ≥0.3: .save() returns a concrete torch.Tensor directly.
        """
        if isinstance(proxy, torch.Tensor):
            return proxy
        if hasattr(proxy, "value") and isinstance(proxy.value, torch.Tensor):
            return proxy.value
        raise TypeError(
            f"Expected torch.Tensor or nnsight proxy with .value, got {type(proxy)}"
        )

    outlier_fractions: dict[str, float] = {
        f"{sigma}_sigma": float(_val(proxy).item())
        for sigma, proxy in zip(OUTLIER_SIGMAS, savers.outlier_proxies)
    }
    per_channel_std: list[float] | None = None
    per_channel_sum: list[float] | None = None
    per_channel_sum_sq: list[float] | None = None
    if savers.per_channel_sum is not None and savers.per_channel_sum_sq is not None:
        sum_ch = _val(savers.per_channel_sum)  # shape (D,)
        sum_sq_ch = _val(savers.per_channel_sum_sq)  # shape (D,)
        # Per-channel n = B * N (batch size × tokens).
        # Recover from the sum tensor: n = total_elements / D.
        D_ch = sum_ch.shape[0]
        per_ch_n = savers.n_samples // D_ch
        per_channel_sum = sum_ch.tolist()
        per_channel_sum_sq = sum_sq_ch.tolist()
        if per_ch_n > 0:
            mean_ch = sum_ch / per_ch_n
            var_ch = sum_sq_ch / per_ch_n - mean_ch**2
            std_ch = var_ch.clamp(min=0.0).sqrt()
            per_channel_std = std_ch.tolist()
    return LayerStats(
        site_identifier=savers.site_identifier,
        mean=float(_val(savers.mean).item()),
        std=float(_val(savers.std).item()),
        m3=float(_val(savers.m3).item()),
        kurtosis=float(_val(savers.kurtosis).item()),
        outlier_fractions=outlier_fractions,
        n_samples=savers.n_samples,
        per_channel_std=per_channel_std,
        per_channel_sum=per_channel_sum,
        per_channel_sum_sq=per_channel_sum_sq,
    )


# ---------------------------------------------------------------------------
# Pre-softmax logit reconstruction helper
# ---------------------------------------------------------------------------


def _register_pre_softmax_saves(
    qkv_proxy: Any,
    num_heads: int,
    head_dim: int,
    scale: float,
    site_id: SiteId,
    n_samples: int,
) -> _StatsSavers:
    """Reconstruct QKᵀ/√d inside a trace context and register stat saves.

    Since timm's Attention module does not expose the raw logit matrix as
    a module output, we recompute it from ``qkv.output`` — mirroring the
    model's own non-fused forward path exactly.

    Args:
        qkv_proxy: Proxy for ``attn.qkv.output``; shape ``(B, N, 3·H·D)``.
        num_heads: Number of attention heads (H).
        head_dim: Dimension per head (D).
        scale: Attention scale factor (1/√D, pre-computed by timm).
        site_id: Site identifier string.
        n_samples: Number of scalar elements in the logit tensor: B*H*N*N.

    Returns:
        A :class:`_StatsSavers` for the ``(B, H, N, N)`` logit tensor.
    """
    # Mirror timm Attention.forward() non-fused path exactly.
    # qkv_proxy: (B, N, 3*H*D) -> reshape -> (3, B, H, N, D)
    b_n_3hd = qkv_proxy.reshape(
        qkv_proxy.shape[0], qkv_proxy.shape[1], 3, num_heads, head_dim
    )
    # permute(2, 0, 3, 1, 4) → (3, B, H, N, D)
    b_n_3hd = b_n_3hd.permute(2, 0, 3, 1, 4)
    q = b_n_3hd[0] * scale   # (B, H, N, D) — scaled queries
    k = b_n_3hd[1]            # (B, H, N, D) — keys
    # (B, H, N, D) @ (B, H, D, N) → (B, H, N, N)
    logits = q @ k.transpose(-2, -1)
    return _register_stat_saves(logits, site_id, n_samples)


# ---------------------------------------------------------------------------
# Public profiling API
# ---------------------------------------------------------------------------


def profile_vit(
    wrapped_model: NNsight,
    input_batch: torch.Tensor,
) -> ProfilingResult:
    """Profile a timm ViT across all measurement sites in a single forward pass.

    Runs one forward pass inside a single nnsight trace context and collects
    statistics at every site for every encoder block.  Only scalar summary
    statistics are retained — no full activation tensors are saved.

    The caller is responsible for:
    * Setting ``fused_attn=False`` on every block before wrapping (see
      :func:`src.model.disable_fused_attn`).
    * Ensuring the underlying model is in ``eval()`` mode.
    * Moving the input batch to the same device as the model.

    Args:
        wrapped_model: An ``NNsight``-wrapped ``VisionTransformer``.
        input_batch: Float tensor of shape ``(B, C, H, W)`` on the model's
            device.

    Returns:
        A :class:`ProfilingResult` with stats for every block and site.

    Raises:
        ProfilingError: If the underlying model has no ``.blocks`` attribute,
            or if the nnsight trace raises an unexpected exception.
        ValueError: If ``input_batch`` is not a 4-D tensor.
    """
    if input_batch.ndim != 4:  # noqa: PLR2004
        raise ValueError(
            f"input_batch must be 4-D (B, C, H, W), got shape {tuple(input_batch.shape)}"
        )

    # Reach through the NNsight wrapper to check the underlying model.
    inner_model = wrapped_model._model  # NNsight stores wrapped module here
    if not hasattr(inner_model, "blocks"):
        raise ProfilingError(
            f"Wrapped model {type(inner_model).__name__} has no 'blocks' attribute. "
            "profile_vit expects a timm VisionTransformer."
        )

    num_blocks: int = len(inner_model.blocks)
    B: int = input_batch.shape[0]
    batch_shape: tuple[int, ...] = tuple(input_batch.shape)

    # Derive token count N and MLP hidden dim D_mlp from model architecture.
    # N = num_patches + 1 (CLS token).  For ViT-B/16 on 224×224: N = 197.
    # Do NOT read from input_batch.shape[2] which is the image height (224),
    # not the token sequence length.
    N: int = inner_model.patch_embed.num_patches + 1
    D: int = inner_model.embed_dim
    attn0 = inner_model.blocks[0].attn
    num_heads: int = attn0.num_heads
    head_dim: int = attn0.head_dim
    D_mlp: int = inner_model.blocks[0].mlp.fc1.out_features

    # Pre-compute per-site element counts (scalars, const for all blocks).
    n_residual: int = B * N * D              # residual_stream, post_layernorm_*
    n_pre_gelu: int = B * N * D_mlp          # pre_gelu
    n_attn: int = B * num_heads * N * N      # pre_softmax, post_softmax

    # Collect _StatsSavers inside the trace, finalize outside.
    all_savers: list[_StatsSavers] = []

    logger.info(
        "Starting nnsight trace: %d blocks, input shape %s, N=%d.",
        num_blocks,
        batch_shape,
        N,
    )

    try:
        with wrapped_model.trace(input_batch):
            for i in range(num_blocks):
                block = wrapped_model.blocks[i]
                attn = block.attn

                # Accesses must follow forward-pass dependency order.
                # nnsight ≥0.3 returns .input as the tensor directly
                # (not a (args, kwargs) tuple), so no [0][0] indexing.

                # --- residual_stream ---
                residual_label: SiteId = (
                    "patch_embed/residual_stream" if i == 0
                    else f"blocks.{i - 1}/residual_stream"
                )
                all_savers.append(
                    _register_stat_saves(block.norm1.input, residual_label, n_residual)
                )

                # --- post_layernorm_1 (pre-attention LN output) ---
                all_savers.append(
                    _register_stat_saves(
                        block.norm1.output, f"blocks.{i}/{SITE_POST_LAYERNORM_1}", n_residual,
                        track_per_channel=True,
                    )
                )

                # --- pre_softmax ---
                # Reconstruct QKᵀ/√d from qkv.output since timm computes
                # this inline (no module boundary to intercept directly).
                # Must access qkv.output BEFORE attn_drop.input (dependency order).
                attn_module = inner_model.blocks[i].attn
                all_savers.append(
                    _register_pre_softmax_saves(
                        qkv_proxy=attn.qkv.output,
                        num_heads=attn_module.num_heads,
                        head_dim=attn_module.head_dim,
                        scale=attn_module.scale,
                        site_id=f"blocks.{i}/{SITE_PRE_SOFTMAX}",
                        n_samples=n_attn,
                    )
                )

                # --- post_softmax ---
                # attn_drop receives the post-softmax attention weights.
                all_savers.append(
                    _register_stat_saves(
                        attn.attn_drop.input, f"blocks.{i}/{SITE_POST_SOFTMAX}", n_attn
                    )
                )

                # --- post_layernorm_2 (pre-MLP LN output) ---
                all_savers.append(
                    _register_stat_saves(
                        block.norm2.output, f"blocks.{i}/{SITE_POST_LAYERNORM_2}", n_residual,
                        track_per_channel=True,
                    )
                )

                # --- pre_gelu ---
                all_savers.append(
                    _register_stat_saves(
                        block.mlp.act.input, f"blocks.{i}/{SITE_PRE_GELU}", n_pre_gelu,
                        track_per_channel=True,
                    )
                )

    except Exception as exc:
        raise ProfilingError(
            f"nnsight trace failed for {type(inner_model).__name__}: {exc}"
        ) from exc

    # Trace context has exited — all .save() proxies now hold concrete values.
    stats: dict[SiteId, LayerStats] = {
        s.site_identifier: _finalize_stats(s) for s in all_savers
    }

    logger.info(
        "Trace complete: collected stats for %d sites.", len(stats)
    )

    return ProfilingResult(
        stats=stats,
        num_blocks=num_blocks,
        batch_shape=batch_shape,
    )


# ---------------------------------------------------------------------------
# Histogram profiling — full tensor capture for one batch
# ---------------------------------------------------------------------------


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
    if input_batch.ndim != 4:  # noqa: PLR2004
        raise ValueError(
            f"input_batch must be 4-D (B, C, H, W), got shape {tuple(input_batch.shape)}"
        )

    inner_model = wrapped_model._model
    if not hasattr(inner_model, "blocks"):
        raise ProfilingError(
            f"Wrapped model {type(inner_model).__name__} has no 'blocks' attribute. "
            "histogram_profile_vit expects a timm VisionTransformer."
        )

    N: int = inner_model.patch_embed.num_patches + 1
    D: int = inner_model.embed_dim
    D_mlp: int = inner_model.blocks[0].mlp.fc1.out_features

    raw: dict[SiteId, Any] = {}

    try:
        with wrapped_model.trace(input_batch):
            for i in block_indices:
                block = wrapped_model.blocks[i]
                attn = block.attn
                # Per-block architecture constants — not block 0's.
                # In standard ViT all blocks share the same values, but
                # variants (e.g. heterogeneous attention) may differ.
                block_attn = inner_model.blocks[i].attn
                block_num_heads: int = block_attn.num_heads
                block_head_dim: int = block_attn.head_dim
                block_scale: float = block_attn.scale

                # Accesses must follow forward-pass dependency order.
                # nnsight ≥0.3 returns .input as the tensor directly.

                # --- residual_stream ---
                residual_label: SiteId = (
                    "patch_embed/residual_stream" if i == 0
                    else f"blocks.{i - 1}/residual_stream"
                )
                raw[residual_label] = block.norm1.input.save()

                # --- post_layernorm_1 ---
                raw[f"blocks.{i}/{SITE_POST_LAYERNORM_1}"] = block.norm1.output.save()

                # --- pre_softmax: reconstruct QKᵀ/√d from qkv.output ---
                # Must access qkv.output BEFORE attn_drop.input (dependency order).
                qkv = attn.qkv.output
                b_n_3hd = qkv.reshape(
                    qkv.shape[0], qkv.shape[1], 3, block_num_heads, block_head_dim
                )
                b_n_3hd = b_n_3hd.permute(2, 0, 3, 1, 4)
                q = b_n_3hd[0] * block_scale
                k = b_n_3hd[1]
                logits = q @ k.transpose(-2, -1)
                raw[f"blocks.{i}/{SITE_PRE_SOFTMAX}"] = logits.save()

                # --- post_softmax ---
                raw[f"blocks.{i}/{SITE_POST_SOFTMAX}"] = attn.attn_drop.input.save()

                # --- post_layernorm_2 ---
                raw[f"blocks.{i}/{SITE_POST_LAYERNORM_2}"] = block.norm2.output.save()

                # --- pre_gelu ---
                raw[f"blocks.{i}/{SITE_PRE_GELU}"] = block.mlp.act.input.save()

    except Exception as exc:
        raise ProfilingError(
            f"nnsight trace failed for {type(inner_model).__name__}: {exc}"
        ) from exc

    return {k: v.cpu() for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_profiling_result(result: ProfilingResult, path: Path) -> None:
    """Serialise a :class:`ProfilingResult` to a JSON file.

    Creates parent directories if they do not exist.  The JSON mirrors the
    dataclass structure exactly via ``dataclasses.asdict``.

    Args:
        result: Completed profiling result to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Saved profiling result (%d sites) to %s.", len(result.stats), path)


def load_profiling_result(path: Path) -> ProfilingResult:
    """Deserialise a :class:`ProfilingResult` from a JSON file.

    Args:
        path: Path to a JSON file written by :func:`save_profiling_result`.

    Returns:
        The reconstructed :class:`ProfilingResult`.

    Raises:
        FileNotFoundError: If ``path`` does not exist on disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Profiling result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    stats: dict[SiteId, LayerStats] = {
        key: LayerStats(**val) for key, val in raw["stats"].items()
    }
    return ProfilingResult(
        stats=stats,
        num_blocks=raw["num_blocks"],
        batch_shape=tuple(raw["batch_shape"]),
    )
