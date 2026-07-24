"""nnsight-based activation profiler for timm Vision Transformers.

This module replaces the legacy raw-hook approach in the original ``hooks.py``.
It wraps a ``timm`` ``VisionTransformer`` with ``nnsight.NNsight`` and collects
activation statistics at five sites across every encoder block in a single
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


def _register_stat_saves(tensor_proxy: Any, site_id: SiteId, n_samples: int) -> _StatsSavers:
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

    return _StatsSavers(
        site_identifier=site_id,
        mean=mean_proxy,
        std=std_proxy,
        m3=m3_proxy,
        kurtosis=kurtosis_proxy,
        outlier_proxies=outlier_proxies,
        n_samples=n_samples,
    )


def _finalize_stats(savers: _StatsSavers) -> LayerStats:
    """Convert saved proxy values to a concrete :class:`LayerStats`.

    Must be called *after* the nnsight trace context exits, at which point
    all ``.save()`` proxy objects have a populated ``.value`` attribute
    holding the concrete ``torch.Tensor`` or scalar.

    Args:
        savers: Populated :class:`_StatsSavers` from a completed trace.

    Returns:
        A fully populated :class:`LayerStats` instance.
    """
    outlier_fractions: dict[str, float] = {
        f"{sigma}_sigma": float(proxy.value.item())
        for sigma, proxy in zip(OUTLIER_SIGMAS, savers.outlier_proxies)
    }
    return LayerStats(
        site_identifier=savers.site_identifier,
        mean=float(savers.mean.value.item()),
        std=float(savers.std.value.item()),
        m3=float(savers.m3.value.item()),
        kurtosis=float(savers.kurtosis.value.item()),
        outlier_fractions=outlier_fractions,
        n_samples=savers.n_samples,
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

                # --- residual_stream ---
                # norm1.input is ((tensor,), {}) — index [0][0] for the tensor.
                residual_label: SiteId = (
                    "patch_embed/residual_stream" if i == 0
                    else f"blocks.{i - 1}/residual_stream"
                )
                all_savers.append(
                    _register_stat_saves(block.norm1.input[0][0], residual_label, n_residual)
                )

                # --- post_layernorm_1 (pre-attention LN output) ---
                all_savers.append(
                    _register_stat_saves(
                        block.norm1.output, f"blocks.{i}/{SITE_POST_LAYERNORM_1}", n_residual
                    )
                )

                # --- post_layernorm_2 (pre-MLP LN output) ---
                all_savers.append(
                    _register_stat_saves(
                        block.norm2.output, f"blocks.{i}/{SITE_POST_LAYERNORM_2}", n_residual
                    )
                )

                # --- pre_gelu ---
                # mlp.act.input is ((tensor,), {}) — index [0][0].
                all_savers.append(
                    _register_stat_saves(
                        block.mlp.act.input[0][0], f"blocks.{i}/{SITE_PRE_GELU}", n_pre_gelu
                    )
                )

                # --- pre_softmax ---
                # Reconstruct QKᵀ/√d from qkv.output since timm computes
                # this inline (no module boundary to intercept directly).
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
                # attn_drop.input is ((tensor,), {}) with shape (B, H, N, N).
                all_savers.append(
                    _register_stat_saves(
                        attn.attn_drop.input[0][0], f"blocks.{i}/{SITE_POST_SOFTMAX}", n_attn
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
