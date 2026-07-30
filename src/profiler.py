"""nnsight-based activation profiler for timm Vision Transformers.

This module replaces the legacy raw-hook approach in the original ``hooks.py``.
It wraps a ``timm`` ``VisionTransformer`` with ``nnsight.NNsight`` and collects
activation statistics at six sites across every encoder block in a single
forward pass, without retaining any full activation tensors in memory.

Key references:
- Pébay (2008) SAND2008-6212: parallel higher-moments merge (M2, M3, M4).
- Zhai et al. (2023) ICML, arXiv:2303.06296: attention entropy collapse.
- Maisonnave et al. (2025) arXiv:2508.16311: CLS/patch entropy separation.
- Bondarenko et al. (2021) arXiv:2109.12948: transformer quantization challenges.
- Dettmers et al. (2022) NeurIPS, arXiv:2208.07339: LLM.int8() outlier handling.
- Xiao et al. (2023) ICML, arXiv:2211.10438: SmoothQuant.
- Wei et al. (2022) NeurIPS (Spotlight), arXiv:2209.13325: outlier suppression.

See ``docs/CITATIONS.md`` for full bibliographic details.

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
# 3σ: moderate outliers (Gaussian tail baseline ≈ 0.27%)
# 4σ: standard quantization-literature threshold (Bondarenko et al. 2021)
# 6σ: extreme outlier detection (Dettmers et al. 2022; Wei et al. 2022)
OUTLIER_SIGMAS: tuple[float, ...] = (3.0, 4.0, 6.0)

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
            ``"3.0_sigma"``, ``"4.0_sigma"``, ``"6.0_sigma"``.
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
    layernorm_gamma: list[float] | None = None
    # Learned scale parameters (γ) of the LayerNorm module at this site.
    # Shape [D] for post_layernorm_1 and post_layernorm_2; None for all other sites.
    # These are static model weights (not activation statistics), extracted from
    # inner_model.blocks[i].norm{1,2}.weight after the trace exits.
    # Critical for distinguishing learned-scale outliers from distribution outliers
    # in per-channel quantization decisions (SmoothQuant, Xiao et al. 2023).
    layernorm_beta: list[float] | None = None
    # Learned bias parameters (β) of the LayerNorm module at this site.
    # Shape [D] for post_layernorm_1 and post_layernorm_2; None for all other sites.
    residual_delta_ratio: float | None = None
    # Mean over batch and tokens of ‖mlp_output‖₂ / ‖x_skip‖₂.
    # Non-None only for residual_stream sites, where it represents the
    # MLP contribution in the block that produced this residual.
    # For patch_embed/residual_stream, this is always None (no preceding MLP).
    # This metric directly answers: "how aggressively does each MLP block
    # modify the residual stream?" — the primary driver of quantization
    # range expansion (Bondarenko et al. 2021, §4.2; Wei et al. 2022, §3.1).
    max: float = 0.0
    # Maximum observed value over all profiled batches.  Useful for
    # sanity-checking quantization ranges — the absolute range determines
    # whether uniform quantization is feasible at all (e.g. INT8 [-128, 127]).
    # Default 0.0 for backward compatibility with old JSON files.
    min: float = 0.0
    # Minimum observed value over all profiled batches.  Default 0.0 for
    # backward compatibility with old JSON files.


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

    Reference: P. Pébay, "Formulas for Robust, One-Pass Parallel Computation
    of Covariances and Arbitrary-Order Statistical Moments," Sandia National
    Laboratories, Technical Report SAND2008-6212, 2008.
    Eq. (3.1)-(3.4) provide the exact parallel merge for M2, M3, M4.
    The Chan et al. (1983) parallel formula for M2 is a special case.

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
    entropy_cls_sum: list[float] | None = None
    # Per-head CLS entropy sum across batches; shape [H].
    # Each batch contributes its batch-mean (not a raw sum).
    entropy_cls_count: int = 0
    # Number of batches contributing to entropy_cls_sum.
    entropy_patch_sum: list[float] | None = None
    # Per-head patch entropy sum (raw sum over B*(N-1) tokens) across all batches.
    entropy_patch_count: int = 0
    # Total number of (B * (N-1)) patch-token samples accumulated.
    layernorm_gamma: list[float] | None = None
    # LayerNorm γ weights (static model parameters, not merged).
    # Copied from the first batch's LayerStats; subsequent batches are
    # expected to carry identical values (model weights don't change).
    # Shape [D]; non-None for post_layernorm_1 and post_layernorm_2 only.
    layernorm_beta: list[float] | None = None
    # LayerNorm β bias (static model parameters, not merged).
    residual_delta_ratio_sum: float = 0.0
    # Running sum of per-batch residual delta ratios (‖mlp_output‖₂ / ‖x_skip‖₂).
    # Accumulated as a simple sum (not Pébay merge) because the ratio is already
    # a per-batch mean.  Divided by residual_delta_ratio_count at finalization.
    residual_delta_ratio_count: int = 0
    # Number of batches contributing to residual_delta_ratio_sum.
    max_val: float = float("-inf")
    # Running maximum across all batches (element-wise, not Pébay-merged).
    # Initialized to -inf so the first batch always sets it.
    min_val: float = float("inf")
    # Running minimum across all batches (element-wise, not Pébay-merged).
    # Initialized to +inf so the first batch always sets it.


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
    patch_token_count: int = 0,
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
        patch_token_count: For post_softmax sites, the number of patch
            token samples in this batch: ``B * (N - 1)``.  Default 0 for
            non-attention sites (ignored).

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

    # --- Attention entropy accumulation ---
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
        H = len(batch_stats.attention_entropy_patches)
        if acc.entropy_patch_sum is None:
            acc.entropy_patch_sum = list(batch_stats.attention_entropy_patches)
            acc.entropy_patch_count = patch_token_count
        else:
            for h in range(H):
                acc.entropy_patch_sum[h] += batch_stats.attention_entropy_patches[h]
            acc.entropy_patch_count += patch_token_count

    # --- LayerNorm γ/β carry-through (static model weights, not merged) ---
    # These are model parameters — they don't change between batches.
    # Store the first batch's values; subsequent batches should carry identical values.
    if batch_stats.layernorm_gamma is not None and acc.layernorm_gamma is None:
        acc.layernorm_gamma = list(batch_stats.layernorm_gamma)
    if batch_stats.layernorm_beta is not None and acc.layernorm_beta is None:
        acc.layernorm_beta = list(batch_stats.layernorm_beta)

    # --- Residual delta ratio accumulation (simple mean across batches) ---
    # The delta ratio is already a per-batch mean (averaged over B×N tokens),
    # so we accumulate a simple sum and divide by count at finalization.
    if batch_stats.residual_delta_ratio is not None:
        acc.residual_delta_ratio_sum += batch_stats.residual_delta_ratio
        acc.residual_delta_ratio_count += 1

    # --- Running max/min (element-wise, not Pébay-merged) ---
    # Max/min don't have a parallel merge formula — we track the
    # element-wise extremum across all batches seen so far.
    if batch_stats.max > acc.max_val:
        acc.max_val = batch_stats.max
    if batch_stats.min < acc.min_val:
        acc.min_val = batch_stats.min



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

    # --- Attention entropy finalization ---
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

    return LayerStats(
        site_identifier=acc.site_identifier,
        mean=acc.mean,
        std=global_std,
        kurtosis=kurtosis,
        m3=acc.M3,
        outlier_fractions=outlier_fractions,
        n_samples=acc.n,
        per_channel_std=per_channel_std,
        attention_entropy_cls=attention_entropy_cls,
        attention_entropy_patches=attention_entropy_patches,
        layernorm_gamma=acc.layernorm_gamma,
        layernorm_beta=acc.layernorm_beta,
        residual_delta_ratio=(
            acc.residual_delta_ratio_sum / acc.residual_delta_ratio_count
            if acc.residual_delta_ratio_count > 0
            else None
        ),
        max=acc.max_val if math.isfinite(acc.max_val) else 0.0,
        min=acc.min_val if math.isfinite(acc.min_val) else 0.0,
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
            ptc = B * (N - 1) if SITE_POST_SOFTMAX in site_id else 0
            if site_id not in accumulators:
                accumulators[site_id] = WelfordAccumulator(site_identifier=site_id)
            merge_batch_stats(accumulators[site_id], layer_stats, batch_n,
                              patch_token_count=ptc)

        num_batches += 1
        if num_batches % 10 == 0:
            logger.info("Profiled %d batches...", num_batches)

    if num_batches == 0:
        raise RuntimeError("DataLoader yielded zero batches; cannot produce stats.")

    logger.info("Finalizing accumulators for %d sites.", len(accumulators))
    return {sid: finalize_accumulator(acc) for sid, acc in accumulators.items()}


# ---------------------------------------------------------------------------
# Global-σ outlier recount — second pass (F2)
# ---------------------------------------------------------------------------


def _count_outliers_in_trace(
    tensor_proxy: Any,
    site_id: SiteId,
    site_params: dict[SiteId, tuple[float, float]],
    batch_counts: dict[SiteId, dict[str, Any]],
) -> None:
    """Register outlier count proxies for one site inside an nnsight trace.

    Computes the count of elements where |x - μ_global| > k·σ_global for
    each k in OUTLIER_SIGMAS, using global μ and σ as Python float
    constants (not proxies).  Only the scalar counts are saved — no raw
    tensors are materialized in host memory.

    Must be called from within a ``with wrapped_model.trace(...):`` block.

    Args:
        tensor_proxy: nnsight proxy for the activation tensor at this site.
        site_id: Site identifier string.
        site_params: Mapping from site_id to (global_mean, global_std).
        batch_counts: Dict to populate with .save() proxies for each sigma
            key.  Modified in-place.
    """
    if site_id not in site_params:
        return
    global_mean, global_std = site_params[site_id]
    if global_std <= 0.0:
        return

    count_proxies: dict[str, Any] = {}
    deviation = (tensor_proxy - global_mean).abs()
    for k in OUTLIER_SIGMAS:
        key = f"{k}_sigma"
        # Count elements exceeding the threshold; save only the scalar sum.
        count_proxies[key] = (deviation > k * global_std).sum().save()
    batch_counts[site_id] = count_proxies


def _extract_scalar(proxy: Any) -> int:
    """Extract a scalar integer from an nnsight .save() proxy or tensor.

    Handles both nnsight <0.3 proxy objects and nnsight ≥0.3 concrete tensors.

    Args:
        proxy: nnsight proxy or torch.Tensor containing a scalar count.

    Returns:
        Integer value of the scalar.
    """
    if isinstance(proxy, torch.Tensor):
        return int(proxy.item())
    if hasattr(proxy, "value") and isinstance(proxy.value, torch.Tensor):
        return int(proxy.value.item())
    raise TypeError(
        f"Expected torch.Tensor or nnsight proxy with .value, got {type(proxy)}"
    )


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

    The standard practice in the quantization literature (Bondarenko et al. 2023;
    Dettmers et al. 2022; Xiao et al. 2023; Wei et al. 2022) is to report
    outlier fractions relative to global σ, computed in a two-pass manner.

    References:
    - Bondarenko et al. (2023), "Understanding and Overcoming the Challenges
      of Efficient Transformer Quantization," arXiv:2109.12948.
    - Dettmers et al. (2022), "LLM.int8(): 8-bit Matrix Multiplication for
      Transformers at Scale," NeurIPS 2022, arXiv:2208.07339.
    - Xiao et al. (2023), "SmoothQuant: Accurate and Efficient Post-Training
      Quantization for Large Language Models," ICML 2023, arXiv:2211.10438.
    - Wei et al. (2022), "Outlier Suppression: Pushing the Limit of Low-bit
      Transformer Language Models," NeurIPS 2022 (Spotlight), arXiv:2209.13325.

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
    inner_model = wrapped_model._model

    # Extract architecture constants.
    N: int = inner_model.patch_embed.num_patches + 1
    D: int = inner_model.embed_dim
    num_heads: int = inner_model.blocks[0].attn.num_heads
    D_mlp: int = inner_model.blocks[0].mlp.fc1.out_features

    # Build a mapping from site_id to (global_mean, global_std).
    site_params: dict[SiteId, tuple[float, float]] = {}
    for site_id, stats in finalized_stats.items():
        site_params[site_id] = (stats.mean, stats.std)

    # Accumulate raw outlier counts per site per sigma threshold.
    outlier_counts: dict[SiteId, dict[str, int]] = {
        sid: {f"{k}_sigma": 0 for k in OUTLIER_SIGMAS}
        for sid in finalized_stats
    }
    total_elements: dict[SiteId, int] = {sid: 0 for sid in finalized_stats}

    num_batches: int = 0

    for images, _ in loader:
        images = images.to(device)
        B: int = images.shape[0]

        # Compute outlier counts inside the nnsight trace to avoid
        # materializing full activation tensors in host memory.
        # For each site and each sigma threshold, we save only the
        # scalar count of elements exceeding k·σ_global.
        batch_counts: dict[SiteId, dict[str, Any]] = {}

        try:
            with wrapped_model.trace(images):
                for i in range(len(inner_model.blocks)):
                    block = wrapped_model.blocks[i]
                    attn = block.attn
                    block_attn = inner_model.blocks[i].attn

                    # --- residual_stream ---
                    # Site labeling convention: blocks.{k}/residual_stream = output
                    # of block k (input to block k+1).  See docs/EXP1-IMPL.md §0.1.
                    residual_label: SiteId = (
                        "patch_embed/residual_stream" if i == 0
                        else f"blocks.{i - 1}/residual_stream"
                    )
                    _count_outliers_in_trace(
                        block.norm1.input, residual_label,
                        site_params, batch_counts,
                    )

                    # --- post_layernorm_1 ---
                    _count_outliers_in_trace(
                        block.norm1.output,
                        f"blocks.{i}/{SITE_POST_LAYERNORM_1}",
                        site_params, batch_counts,
                    )

                    # --- pre_softmax (reconstruct from qkv) ---
                    qkv = attn.qkv.output
                    b_n_3hd = qkv.reshape(
                        qkv.shape[0], qkv.shape[1], 3,
                        block_attn.num_heads, block_attn.head_dim,
                    )
                    b_n_3hd = b_n_3hd.permute(2, 0, 3, 1, 4)
                    q = b_n_3hd[0] * block_attn.scale
                    k = b_n_3hd[1]
                    logits = q @ k.transpose(-2, -1)
                    _count_outliers_in_trace(
                        logits, f"blocks.{i}/{SITE_PRE_SOFTMAX}",
                        site_params, batch_counts,
                    )

                    # --- post_softmax ---
                    _count_outliers_in_trace(
                        attn.attn_drop.input,
                        f"blocks.{i}/{SITE_POST_SOFTMAX}",
                        site_params, batch_counts,
                    )

                    # --- post_layernorm_2 ---
                    _count_outliers_in_trace(
                        block.norm2.output,
                        f"blocks.{i}/{SITE_POST_LAYERNORM_2}",
                        site_params, batch_counts,
                    )

                    # --- pre_gelu ---
                    _count_outliers_in_trace(
                        block.mlp.act.input,
                        f"blocks.{i}/{SITE_PRE_GELU}",
                        site_params, batch_counts,
                    )

                # --- Final residual stream (output of last encoder block, before head LN) ---
                # Same gap as profile_vit — the block loop labels block[i].norm1.input
                # as blocks.{i-1}/residual_stream, so the output of the final block is
                # never counted.  Capture it from the final LayerNorm's input.
                _count_outliers_in_trace(
                    wrapped_model.norm.input,
                    f"blocks.{len(inner_model.blocks) - 1}/residual_stream",
                    site_params, batch_counts,
                )

        except Exception as exc:
            raise ProfilingError(
                f"Outlier recount trace failed: {exc}"
            ) from exc

        # Trace exited — accumulate counts.
        for site_id, count_proxies in batch_counts.items():
            if site_id not in site_params:
                continue
            _, global_std = site_params[site_id]
            if global_std <= 0.0:
                continue
            for key, proxy in count_proxies.items():
                val = _extract_scalar(proxy)
                outlier_counts[site_id][key] += int(val)
            # Track total elements from the first sigma key's count proxy
            # (all proxies for a site have the same tensor shape).
            # We estimate n from the first batch's known architecture.
            # Actually, we track n via _site_n below.

        # Track total elements per site using architecture constants.
        for site_id in site_params:
            if site_id in batch_counts:
                batch_n = _site_n(site_id, B, N, D, D_mlp, num_heads)
                total_elements[site_id] += batch_n

        num_batches += 1
        if num_batches % 10 == 0:
            logger.info("Outlier recount: %d batches...", num_batches)

    if num_batches == 0:
        raise RuntimeError("DataLoader yielded zero batches; cannot count outliers.")

    # Compute final fractions.
    result: dict[SiteId, dict[str, float]] = {}
    for site_id in finalized_stats:
        n = total_elements[site_id]
        if n == 0:
            result[site_id] = {f"{k}_sigma": 0.0 for k in OUTLIER_SIGMAS}
        else:
            result[site_id] = {
                key: count / n for key, count in outlier_counts[site_id].items()
            }

    return result


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
        entropy_cls_proxy: Saved CLS entropy proxy (shape (H,)); non-None
            for post_softmax sites only.
        entropy_patch_sum_proxy: Saved patch entropy sum proxy (shape (H,));
            non-None for post_softmax sites only.
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
    entropy_cls_proxy: Any = None
    entropy_patch_sum_proxy: Any = None
    residual_delta_ratio: Any = None
    # Saved proxy for the residual delta ratio (‖mlp_output‖₂ / ‖x_skip‖₂).
    # Non-None only for residual_stream sites (except patch_embed).
    # Set after the post_layernorm_2 registration when both norm1.input
    # and norm2.output proxies are available.
    max_proxy: Any = None
    # Saved proxy for the element-wise maximum of the activation tensor.
    min_proxy: Any = None
    # Saved proxy for the element-wise minimum of the activation tensor.


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

    # Element-wise max/min for quantization range analysis.
    max_proxy = t.max().save()
    min_proxy = t.min().save()

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
        max_proxy=max_proxy,
        min_proxy=min_proxy,
    )


def _register_entropy_saves(
    attn_weight_proxy: Any,
) -> tuple[Any, Any]:
    """Compute per-head Shannon entropy for CLS and patch queries separately.

    Follows the literature convention of treating CLS-to-all attention and
    patch-to-patch attention as distinct distributions.
    Ref: Maisonnave et al. 2025 (arXiv:2508.16311);
         Mali 2025 (arXiv:2511.18925).

    The Shannon entropy formula H = -Σ p_j log(p_j) follows Zhai et al. (2023,
    ICML, arXiv:2303.06296), who define attention entropy collapse as a
    diagnostic for transformer training stability.

    For each head h and query position i, entropy is:
        H(i, h) = -Σ_{j=1..N} p_{h,i,j} · log(p_{h,i,j} + ε)
    where ε = 1e-8 is a proxy NaN guard (not a bias-correction).

    When called inside an nnsight trace context, returns .save() proxies.
    When called with a concrete torch.Tensor, returns concrete tensors directly
    (for standalone testing without nnsight).

    Args:
        attn_weight_proxy: Proxy or tensor of shape (B, H, N, N).
            Row 0 of the query (dim 2) is the CLS token.
            Rows 1..N-1 are patch token queries.

    Returns:
        (cls_entropy_proxy, patch_entropy_sum_proxy) each of shape (H,).
        cls_entropy_proxy:   mean over B of H(query=CLS, head=h).
        patch_entropy_sum_proxy: sum over B×(N-1) of H(query=patch_i, head=h).
    """
    eps = 1e-8  # proxy NaN guard only; not a bias-correction

    # Per-query entropy: -(p * log(p + eps)).sum(dim=-1) → shape (B, H, N)
    per_query_entropy = -(attn_weight_proxy * (attn_weight_proxy + eps).log()).sum(dim=-1)

    # CLS row: query index 0 → shape (B, H)
    cls_entropy = per_query_entropy[:, :, 0]          # (B, H)
    # Mean over batch → (H,)
    cls_entropy_mean = cls_entropy.mean(dim=0)         # (H,)

    # Patch rows: query indices 1..N-1 → shape (B, H, N-1)
    patch_entropy = per_query_entropy[:, :, 1:]        # (B, H, N-1)
    # Sum across batch and patch queries, track total count separately.
    # Use sum (not mean) so the accumulator can do sample-count weighting.
    patch_entropy_sum = patch_entropy.sum(dim=(0, 2))  # (H,)

    # If we're inside an nnsight trace, .save() the results.
    # If called with concrete tensors (standalone testing), return as-is.
    if hasattr(cls_entropy_mean, "save"):
        return cls_entropy_mean.save(), patch_entropy_sum.save()
    return cls_entropy_mean, patch_entropy_sum


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

    # --- Attention entropy finalization ---
    attention_entropy_cls: list[float] | None = None
    attention_entropy_patches: list[float] | None = None
    if savers.entropy_cls_proxy is not None:
        attention_entropy_cls = _val(savers.entropy_cls_proxy).tolist()
    if savers.entropy_patch_sum_proxy is not None:
        # attention_entropy_patches carries the raw per-batch sum when
        # produced by profile_vit; it carries the global mean when produced
        # by finalize_accumulator.  The field name is reused for both stages.
        attention_entropy_patches = _val(savers.entropy_patch_sum_proxy).tolist()

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
        attention_entropy_cls=attention_entropy_cls,
        attention_entropy_patches=attention_entropy_patches,
        residual_delta_ratio=(
            float(_val(savers.residual_delta_ratio).item())
            if savers.residual_delta_ratio is not None
            else None
        ),
        max=float(_val(savers.max_proxy).item()) if savers.max_proxy is not None else 0.0,
        min=float(_val(savers.min_proxy).item()) if savers.min_proxy is not None else 0.0,
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
            # Delta ratio for blocks.{i}/residual_stream is computed using
            # block i's norm1.input (skip) and norm2.output (MLP).
            # Since norm1.input is consumed by _register_stat_saves and
            # cannot be re-accessed (nnsight 0.7.0 OutOfOrderError), we
            # capture the per-token skip norm as a separate proxy at the
            # point of first access and store it for later use.
            pending_skip_norm: Any = None  # skip norm proxy from previous iteration

            for i in range(num_blocks):
                block = wrapped_model.blocks[i]
                attn = block.attn

                # Accesses must follow forward-pass dependency order.
                # nnsight ≥0.3 returns .input as the tensor directly
                # (not a (args, kwargs) tuple), so no [0][0] indexing.

                # --- residual_stream ---
                # Site labeling convention (see docs/EXP1-IMPL.md §0.1):
                #   blocks.{k}/residual_stream = output of block k (input to block k+1)
                #   patch_embed/residual_stream = patch embed + pos encoding + CLS (input to block 0)
                #   blocks.11/residual_stream = final encoder output (before head LN)
                # All other sites (post_layernorm_1, pre_softmax, etc.) are measured
                # INSIDE the block whose index appears in the label.
                residual_label: SiteId = (
                    "patch_embed/residual_stream" if i == 0
                    else f"blocks.{i - 1}/residual_stream"
                )
                # Capture skip norm BEFORE _register_stat_saves consumes the proxy.
                # This is the per-token L2 norm of the residual entering block i,
                # averaged over batch and tokens: mean_{b,t} ‖x_skip[b,t,:]‖₂.
                skip_norm_proxy = block.norm1.input.norm(dim=-1).mean().save()

                residual_savers = _register_stat_saves(
                    block.norm1.input, residual_label, n_residual
                )
                # Attach delta ratio computed from the previous iteration's
                # skip_norm and mlp_output.  For i=0 (patch_embed/residual_stream),
                # pending_skip_norm is None — correct, no preceding MLP block.
                if pending_skip_norm is not None:
                    residual_savers.residual_delta_ratio = pending_skip_norm
                    pending_skip_norm = None
                all_savers.append(residual_savers)

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
                # Capture the proxy once to avoid double .input access.
                attn_input_proxy = attn.attn_drop.input
                ps_savers = _register_stat_saves(
                    attn_input_proxy, f"blocks.{i}/{SITE_POST_SOFTMAX}", n_attn
                )
                # Register entropy saves on the same proxy.
                ps_savers.entropy_cls_proxy, ps_savers.entropy_patch_sum_proxy = \
                    _register_entropy_saves(attn_input_proxy)
                all_savers.append(ps_savers)

                # --- post_layernorm_2 (pre-MLP LN output) ---
                all_savers.append(
                    _register_stat_saves(
                        block.norm2.output, f"blocks.{i}/{SITE_POST_LAYERNORM_2}", n_residual,
                        track_per_channel=True,
                    )
                )

                # --- Residual delta ratio: ‖mlp_output‖₂ / ‖x_skip‖₂ ---
                # Computed per-token and averaged over batch and tokens.
                # skip_norm_proxy was captured above (before norm1.input was consumed).
                # mlp_norm uses norm2.output which is available now.
                # The ratio will be attached to blocks.{i}/residual_stream in the
                # next loop iteration (or after the loop for the final block).
                # Ref: Bondarenko et al. (2021), arXiv:2109.12948, §4.2;
                #      Wei et al. (2022), NeurIPS, arXiv:2209.13325, §3.1.
                mlp_norm = block.norm2.output.norm(dim=-1).mean()  # scalar proxy
                pending_skip_norm = (mlp_norm / (skip_norm_proxy + 1e-8)).save()

                # --- pre_gelu ---
                all_savers.append(
                    _register_stat_saves(
                        block.mlp.act.input, f"blocks.{i}/{SITE_PRE_GELU}", n_pre_gelu,
                        track_per_channel=True,
                    )
                )

            # --- Final residual stream (output of last encoder block, before head LN) ---
            # The block loop labels block[i].norm1.input as blocks.{i-1}/residual_stream,
            # so the output of the final block (block 11) is never captured inside the
            # loop.  We capture it here from the final LayerNorm's input, which is the
            # raw residual stream exiting block num_blocks-1.
            #
            # This is the single most important activation tensor for quantization
            # range calibration in Phase 2/3 — it represents the cumulative effect of
            # all encoder blocks before the classification head.
            # Ref: Bondarenko et al. (2021), arXiv:2109.12948, §4.2.
            final_residual_savers = _register_stat_saves(
                wrapped_model.norm.input,
                f"blocks.{num_blocks - 1}/residual_stream",
                n_residual,
            )
            # Attach the delta ratio from the final block's MLP.
            if pending_skip_norm is not None:
                final_residual_savers.residual_delta_ratio = pending_skip_norm
            all_savers.append(final_residual_savers)

    except Exception as exc:
        raise ProfilingError(
            f"nnsight trace failed for {type(inner_model).__name__}: {exc}"
        ) from exc

    # Trace context has exited — all .save() proxies now hold concrete values.
    stats: dict[SiteId, LayerStats] = {
        s.site_identifier: _finalize_stats(s) for s in all_savers
    }

    # --- Attach LayerNorm γ/β weights to post_layernorm sites ---
    # These are static model parameters, extracted outside the trace from the
    # underlying PyTorch model.  They enable distinguishing learned-scale
    # outliers (large γ) from distribution outliers (large activation variance)
    # during per-channel quantization calibration (SmoothQuant, Xiao et al. 2023).
    for i in range(num_blocks):
        block = inner_model.blocks[i]
        # norm1 → post_layernorm_1
        ln1_site = f"blocks.{i}/{SITE_POST_LAYERNORM_1}"
        if ln1_site in stats:
            ln1 = block.norm1
            stats[ln1_site].layernorm_gamma = ln1.weight.detach().cpu().tolist()
            stats[ln1_site].layernorm_beta = (
                ln1.bias.detach().cpu().tolist() if ln1.bias is not None else None
            )
        # norm2 → post_layernorm_2
        ln2_site = f"blocks.{i}/{SITE_POST_LAYERNORM_2}"
        if ln2_site in stats:
            ln2 = block.norm2
            stats[ln2_site].layernorm_gamma = ln2.weight.detach().cpu().tolist()
            stats[ln2_site].layernorm_beta = (
                ln2.bias.detach().cpu().tolist() if ln2.bias is not None else None
            )

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
# Summary table generation (F4)
# ---------------------------------------------------------------------------

# Canonical site order for row sorting within each block.
_CANONICAL_SITE_ORDER: dict[str, int] = {
    SITE_RESIDUAL_STREAM: 0,
    SITE_POST_LAYERNORM_1: 1,
    SITE_PRE_SOFTMAX: 2,
    SITE_POST_SOFTMAX: 3,
    SITE_POST_LAYERNORM_2: 4,
    SITE_PRE_GELU: 5,
}


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
    if not result.stats:
        raise ValueError("ProfilingResult.stats is empty; cannot generate summary table.")

    # Discover outlier fraction column names from the first LayerStats that
    # has a non-empty outlier_fractions dict.
    frac_keys: list[str] = []
    for stats in result.stats.values():
        if stats.outlier_fractions:
            frac_keys = sorted(stats.outlier_fractions.keys())
            break

    # Parse site identifiers into (block_key, site_name) pairs.
    parsed: list[tuple[str, str, str, LayerStats]] = []
    for site_id, stats in result.stats.items():
        # Split on "/" — max 1 split.
        parts = site_id.split("/", 1)
        if len(parts) == 2:
            block_key, site_name = parts[0], parts[1]
        else:
            block_key, site_name = site_id, ""
        parsed.append((block_key, site_name, site_id, stats))

    # Sort key: (block_sort_key, site_sort_key).
    def _block_sort_key(block_key: str) -> tuple[int, int | str]:
        """Return sort key for a block prefix.

        patch_embed sorts first (0, 0).
        blocks.N sorts by numeric N (1, N).
        Unknown prefixes sort last (2, block_key).
        """
        if block_key == "patch_embed":
            return (0, 0)
        if block_key.startswith("blocks."):
            try:
                n = int(block_key.split(".", 1)[1])
                return (1, n)
            except (ValueError, IndexError):
                pass
        return (2, block_key)

    def _site_sort_key(site_name: str) -> int:
        """Return sort index for a site name; unknown sites sort last."""
        return _CANONICAL_SITE_ORDER.get(site_name, 99)

    parsed.sort(key=lambda x: (_block_sort_key(x[0]), _site_sort_key(x[1])))

    # Build rows.
    rows: list[dict[str, object]] = []
    for block_key, site_name, _site_id, stats in parsed:
        row: dict[str, object] = {
            "block": block_key,
            "site": site_name,
            "mean": stats.mean,
            "std": stats.std,
            "kurtosis": stats.kurtosis,
        }
        for fk in frac_keys:
            col_name = f"frac_{fk}"
            row[col_name] = stats.outlier_fractions.get(fk, "")
        rows.append(row)

    return rows


def save_summary_table(rows: list[dict[str, object]], path: Path) -> None:
    """Write summary table rows to a CSV file.

    Creates parent directories if they do not exist.

    Args:
        rows: Non-empty output of generate_summary_table.
        path: Destination CSV path.

    Raises:
        ValueError: If rows is empty.
    """
    import csv

    if not rows:
        raise ValueError("rows is empty; cannot write summary table.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary table (%d rows) written to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_profiling_result(result: ProfilingResult, path: Path) -> None:
    """Serialise a :class:`ProfilingResult` to a JSON file.

    Creates parent directories if they do not exist.  The JSON mirrors the
    dataclass structure exactly via ``dataclasses.asdict``.

    All floating-point values are serialised with full float64 precision.
    Python's ``json.dump`` uses ``repr()`` internally, which guarantees
    round-trip fidelity (the shortest decimal representation that reproduces
    the exact IEEE 754 binary value).  This means a value like ``0.00261``
    in the JSON is the exact float64 that was computed — not a truncated
    approximation.  See ``test_float_precision_round_trip`` for verification.

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
