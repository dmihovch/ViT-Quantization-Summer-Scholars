"""Outlier-zeroing ablation for Phase 2 (nnsight-based intervention).

For each sigma threshold ``k``, activation elements whose deviation from the
mean exceeds ``k * σ`` (i.e. ``|x − μ| > k·σ``) are hard-zeroed at a specified
measurement site.  This mean-centered definition is consistent with Phase 1's
outlier definition and the standard statistical convention in the quantization
literature (Wei et al. 2022, §3.1; Bondarenko et al. 2021, §4.1).

The experiment sweeps multiple ``k`` values across three sites and records the
resulting top-1/top-5 accuracy, percentage of zeroed activations per layer,
and (for pre_softmax) attention entropy deltas relative to Phase 1 baselines.

A random-zeroing control condition is also supported: instead of zeroing
outliers, a random subset of elements matching the same fraction is zeroed.
This distinguishes the effect of zeroing *outliers specifically* from the
effect of zeroing *any* elements.

All intervention is performed inside nnsight trace contexts — no raw PyTorch
hooks are used.  This gives us the same granular access to intermediate
activations that Phase 1 profiling uses, but with tensor replacement instead
of observation.

.. note::

    Functions suffixed with ``_in_trace`` or prefixed with ``_intervene_``
    are called **inside** an nnsight trace context (``with wrapped.trace(...)``).
    Their ``block`` parameters are nnsight proxy objects, not concrete
    ``nn.Module`` instances.  Tensor operations on proxies are deferred until
    the trace executes.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from nnsight import NNsight

from src.profiler import LayerStats

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result record for a single (site, sigma threshold, layer) combination.

    Attributes
    ----------
    site:
        Measurement site being ablated: ``"pre_gelu"``, ``"pre_softmax"``,
        or ``"residual_stream"``.
    sigma_threshold:
        The multiplier ``k`` used to compute the absolute threshold
        ``k * stats.std``.
    site_identifier:
        Fully-qualified site identifier matching :attr:`LayerStats.site_identifier`,
        e.g. ``"blocks.3/pre_gelu"``.
    pct_zeroed:
        Fraction of activation elements zeroed by the mask, expressed as a
        percentage in the range ``[0, 100]``.
    top1_accuracy:
        Top-1 classification accuracy (%) measured after applying zeroing
        at this site and threshold.
    top5_accuracy:
        Top-5 classification accuracy (%) under the same conditions.
    baseline_top1:
        Unablated top-1 accuracy (%) for degradation computation.
    baseline_top5:
        Unablated top-5 accuracy (%) for degradation computation.
    cls_entropy:
        Per-head CLS query entropy (nats) after ablation, shape ``[H]``.
        Populated only for ``pre_softmax`` site; empty list for other sites.
    patch_entropy:
        Per-head patch query mean entropy (nats) after ablation, shape ``[H]``.
        Populated only for ``pre_softmax`` site; empty list for other sites.
    baseline_cls_entropy:
        Per-head CLS query entropy (nats) from Phase 1 baseline.
        Populated only for ``pre_softmax`` site; empty list for other sites.
    baseline_patch_entropy:
        Per-head patch query mean entropy (nats) from Phase 1 baseline.
        Populated only for ``pre_softmax`` site; empty list for other sites.
    """

    site: str
    sigma_threshold: float
    site_identifier: str
    pct_zeroed: float
    top1_accuracy: float
    top5_accuracy: float
    baseline_top1: float
    baseline_top5: float
    seed: int = 0
    """Random seed used for this run (for multi-seed aggregation)."""
    is_random: bool = False
    """Whether this result used random (not outlier-threshold) zeroing."""
    granularity: str = "global"
    """Zeroing granularity: ``"global"`` or ``"per_channel"``."""
    ablation_mode: str = "outlier"
    """Ablation variant for per-channel mode: ``"outlier"``, ``"mean_only"``,
    or ``"var_only"``.  Ignored in global granularity mode."""
    cls_entropy: list[float] = field(default_factory=list)
    patch_entropy: list[float] = field(default_factory=list)
    baseline_cls_entropy: list[float] = field(default_factory=list)
    baseline_patch_entropy: list[float] = field(default_factory=list)




def compute_entropy_delta(
    ablated_cls: list[float],
    ablated_patch: list[float],
    baseline_cls: list[float],
    baseline_patch: list[float],
) -> dict[str, float]:
    """Compute per-head mean entropy change relative to Phase 1 baseline.

    Parameters
    ----------
    ablated_cls:
        Per-head CLS entropy after ablation, shape ``[H]``.
    ablated_patch:
        Per-head patch entropy after ablation, shape ``[H]``.
    baseline_cls:
        Per-head CLS entropy from Phase 1, shape ``[H]``.
    baseline_patch:
        Per-head patch entropy from Phase 1, shape ``[H]``.

    Returns
    -------
    dict[str, float]
        Keys: ``"mean_cls_delta"``, ``"mean_patch_delta"`` — mean per-head
        entropy change in nats.  Positive means entropy *increased* (more
        uniform attention) after zeroing.
    """
    if not ablated_cls or not baseline_cls:
        return {"mean_cls_delta": 0.0, "mean_patch_delta": 0.0}

    assert len(ablated_cls) == len(baseline_cls), (
        f"CLS entropy length mismatch: ablated={len(ablated_cls)}, baseline={len(baseline_cls)}"
    )
    assert len(ablated_patch) == len(baseline_patch), (
        f"Patch entropy length mismatch: ablated={len(ablated_patch)}, baseline={len(baseline_patch)}"
    )

    cls_deltas = [a - b for a, b in zip(ablated_cls, baseline_cls)]
    patch_deltas = [a - b for a, b in zip(ablated_patch, baseline_patch)]

    return {
        "mean_cls_delta": sum(cls_deltas) / len(cls_deltas),
        "mean_patch_delta": sum(patch_deltas) / len(patch_deltas),
    }


def _build_random_mask(
    tensor: torch.Tensor,
    fraction: float,
    seed: int | None = None,
    salt: int = 0,
) -> torch.Tensor:
    """Build a boolean mask where ``True`` means *keep* (not zeroed).

    Zeros exactly ``fraction`` of elements at uniformly random positions.
    Used as a control condition to distinguish the effect of zeroing
    *outliers specifically* from the effect of zeroing *any* elements.

    When called inside an nnsight trace context, ``tensor`` is an nnsight
    proxy and the returned mask is also a proxy.

    Parameters
    ----------
    tensor:
        Activation tensor or nnsight proxy (any shape).
    fraction:
        Fraction of elements to zero, in ``[0, 1]``.
    seed:
        Seed for reproducibility.  Combined with ``salt`` to produce
        per-layer distinct masks from a single base seed.
    salt:
        Per-layer offset added to the seed (e.g., block index).

    Returns
    -------
    torch.Tensor
        Boolean mask of same shape as ``tensor``, ``True`` where the element
        should be preserved.
    """
    if fraction <= 0.0:
        return torch.ones_like(tensor, dtype=torch.bool)
    if fraction >= 1.0:
        return torch.zeros_like(tensor, dtype=torch.bool)

    effective_seed = (seed if seed is not None else 42) + salt
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(effective_seed)

    numel = tensor.numel()
    k = int(fraction * numel)
    if k == 0:
        return torch.ones_like(tensor, dtype=torch.bool)

    # Generate random permutation indices for the elements to zero.
    # We select k positions to zero; the rest are kept.
    perm = torch.randperm(numel, generator=generator, device=tensor.device)
    zero_indices = perm[:k]

    mask = torch.ones(numel, dtype=torch.bool, device=tensor.device)
    mask[zero_indices] = False
    return mask.reshape(tensor.shape)


def _build_zeroing_mask(
    tensor: torch.Tensor,
    sigma_k: float,
    sigma: float,
    mean: float,
) -> torch.Tensor:
    """Build a boolean mask where ``True`` means *keep* (not an outlier).

    An element is kept if ``|x − μ| ≤ sigma_k * sigma``, consistent with
    Phase 1's mean-centered outlier definition (``|x − μ| > k·σ``).
    This is the standard statistical definition used in the quantization
    literature (Wei et al. 2022, §3.1; Bondarenko et al. 2021, §4.1).

    When called inside an nnsight trace context, ``tensor`` is an nnsight
    proxy and the returned mask is also a proxy — no concrete tensors are
    materialised until the trace exits.

    Parameters
    ----------
    tensor:
        Activation tensor or nnsight proxy (any shape).
    sigma_k:
        Threshold multiplier.
    sigma:
        Per-layer population standard deviation from Phase 1.
    mean:
        Per-layer population mean from Phase 1.

    Returns
    -------
    torch.Tensor
        Boolean mask of same shape as ``tensor``, ``True`` where the element
        should be preserved.
    """
    threshold = sigma_k * sigma
    return (tensor - mean).abs() <= threshold


def _build_per_channel_zeroing_mask(
    tensor: torch.Tensor,
    sigma_k: float,
    per_channel_sigma: list[float],
    per_channel_mean: list[float],
    device: torch.device,
) -> torch.Tensor:
    """Build a per-channel boolean mask where ``True`` means *keep*.

    For each channel ``c``, an element ``x_c`` is kept if
    ``|x_c − μ_c| ≤ sigma_k * σ_c``, where ``μ_c`` and ``σ_c`` are the
    per-channel mean and standard deviation from Phase 1 profiling.

    The mask broadcasts over the batch and token dimensions: ``tensor`` has
    shape ``(B, N, D)`` and the per-channel statistics have shape ``(D,)``.

    When called inside an nnsight trace context, ``tensor`` is an nnsight
    proxy and the returned mask is also a proxy.

    Parameters
    ----------
    tensor:
        Activation tensor or nnsight proxy of shape ``(B, N, D)``.
    sigma_k:
        Threshold multiplier.
    per_channel_sigma:
        Per-channel population standard deviation, length ``D``.
    per_channel_mean:
        Per-channel population mean, length ``D``.
    device:
        Device on which to create the statistics tensors.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(B, N, D)``, ``True`` where the element
        should be preserved.
    """
    pc_sigma = torch.tensor(per_channel_sigma, device=device, dtype=torch.float32)
    pc_mean = torch.tensor(per_channel_mean, device=device, dtype=torch.float32)
    threshold = sigma_k * pc_sigma  # shape (D,)
    # Broadcast: (B, N, D) - (D,) → (B, N, D)
    return (tensor - pc_mean).abs() <= threshold


def zero_outliers_in_trace(
    wrapped_model: NNsight,
    input_batch: torch.Tensor,
    site: str,
    sigma_k: float,
    layer_stats: dict[str, LayerStats],
    random_fractions: dict[str, float] | None = None,
    random_seed: int | None = None,
    per_channel: bool = False,
    ablation_mode: str = "outlier",
    layer_range: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, dict[str, list[float]]]]:
    """Run one forward pass with outlier zeroing at the specified site.

    Inside the nnsight trace, for every encoder block, the activation tensor
    at ``site`` is replaced with a zeroed version where ``|x − μ| > k·σ``
    (mean-centered, consistent with Phase 1's outlier definition).
    The model then continues the forward pass with the modified activations.

    If ``random_fractions`` is provided, zeros a random subset of elements
    matching the given fraction per layer instead of using the outlier
    threshold — this is the random-zeroing control condition.

    If ``per_channel`` is True and ``site`` is ``"pre_gelu"``, uses
    per-channel μ_c and σ_c for thresholding (see
    :func:`_intervene_pre_gelu`).  Ignored for other sites.

    For ``pre_softmax``, also captures per-head CLS and patch attention
    entropy on the zeroed attention weights for entropy delta computation.

    Parameters
    ----------
    wrapped_model:
        NNsight-wrapped VisionTransformer with ``fused_attn=False``.
    input_batch:
        Float tensor of shape ``(B, C, H, W)`` on the model's device.
    site:
        Which measurement site to zero: ``"pre_gelu"``, ``"pre_softmax"``,
        or ``"residual_stream"``.
    sigma_k:
        Threshold multiplier (e.g. ``3.0`` means zero elements > 3σ).
        Ignored when ``random_fractions`` is provided.
    layer_stats:
        Per-site statistics from Phase 1, keyed by ``site_identifier``.
        Must contain ``std`` and ``mean`` for every site being zeroed.
    random_fractions:
        If provided, maps ``site_identifier`` → fraction (0–1) of elements
        to zero randomly.  Overrides the outlier threshold logic.
    random_seed:
        Seed for the random mask generator.  Only used when
        ``random_fractions`` is provided.
    per_channel:
        If True, use per-channel thresholds for pre_gelu ablation.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float], dict[str, dict[str, list[float]]]]
        - Logits tensor of shape ``(B, num_classes)``.
        - Dict mapping ``site_identifier`` → percentage zeroed for this batch.
        - Dict mapping ``site_identifier`` → ``{"cls": [H], "patch": [H]}``
          of per-head entropy values.  Empty for non-pre_softmax sites.

    Raises
    ------
    ValueError:
        If ``site`` is not one of the three supported values.
    """
    if site not in ("pre_gelu", "pre_softmax", "residual_stream"):
        raise ValueError(
            f"Unknown site '{site}'.  Must be 'pre_gelu', 'pre_softmax', or 'residual_stream'."
        )

    inner_model = wrapped_model._model
    num_blocks: int = len(inner_model.blocks)
    pct_zeroed: dict[str, float] = {}
    entropy_data: dict[str, dict[str, list[float]]] = {}

    start_blk, end_blk = layer_range if layer_range is not None else (0, num_blocks - 1)

    with torch.no_grad():
        with wrapped_model.trace(input_batch):
            for i in range(num_blocks):
                if i < start_blk or i > end_blk:
                    continue
                block = wrapped_model.blocks[i]

                if site == "pre_gelu":
                    _intervene_pre_gelu(
                        block, i, sigma_k, layer_stats, pct_zeroed,
                        random_fractions=random_fractions, random_seed=random_seed,
                        per_channel=per_channel, ablation_mode=ablation_mode,
                    )
                elif site == "residual_stream":
                    _intervene_residual_stream(
                        block, i, sigma_k, layer_stats, pct_zeroed,
                        random_fractions=random_fractions, random_seed=random_seed,
                    )
                elif site == "pre_softmax":
                    _intervene_pre_softmax(
                        block, inner_model.blocks[i].attn, i, sigma_k,
                        layer_stats, pct_zeroed, entropy_data,
                    )

            logits = wrapped_model.output.save()

    return logits, pct_zeroed, entropy_data


# ---------------------------------------------------------------------------
# Per-site intervention helpers (called inside nnsight trace context)
#
# All ``block`` parameters are nnsight proxy objects, not concrete
# ``nn.Module`` instances.  Tensor operations on proxies are deferred
# until the trace executes.  We use ``Any`` for the block type because
# nnsight does not expose a public type for its proxy objects.
# ---------------------------------------------------------------------------


def _intervene_pre_gelu(
    block: Any,
    block_idx: int,
    sigma_k: float,
    layer_stats: dict[str, LayerStats],
    pct_zeroed: dict[str, float],
    random_fractions: dict[str, float] | None = None,
    random_seed: int | None = None,
    per_channel: bool = False,
    ablation_mode: str = "outlier",
) -> None:
    """Zero pre-GELU outliers for a single encoder block.

    Replaces ``block.mlp.act.input`` with a zeroed version where
    ``|x − μ| > sigma_k * sigma`` (mean-centered, consistent with Phase 1).

    If ``per_channel`` is True, uses per-channel μ_c and σ_c from
    ``layer_stats[site_id].per_channel_mean`` and ``.per_channel_std``
    instead of the global scalar μ and σ.  This tests whether outlier
    concentration in high-variance channels drives accuracy degradation.

    If ``random_fractions`` is provided, zeros a random subset of elements
    matching the given fraction instead of using the outlier threshold.

    Called inside an nnsight trace context — ``block`` is an nnsight proxy.

    Parameters
    ----------
    block:
        Nnsight proxy for ``wrapped_model.blocks[i]``.
    block_idx:
        Index of this encoder block (0-based).
    sigma_k:
        Threshold multiplier.
    layer_stats:
        Per-site statistics from Phase 1.
    pct_zeroed:
        Dict mutated in-place with ``site_identifier`` → percentage zeroed.
    random_fractions:
        If provided, maps ``site_identifier`` → fraction (0–1) of elements
        to zero randomly.  Overrides the outlier threshold logic.
    random_seed:
        Seed for the random mask generator.  Only used when
        ``random_fractions`` is provided.
    per_channel:
        If True, use per-channel μ_c and σ_c for thresholding.
    """
    site_id = f"blocks.{block_idx}/pre_gelu"
    if site_id not in layer_stats:
        return

    tensor = block.mlp.act.input

    if random_fractions is not None and site_id in random_fractions:
        mask = _build_random_mask(tensor, random_fractions[site_id], random_seed, block_idx)
        pct_zeroed[site_id] = 100.0 * random_fractions[site_id]
    elif per_channel:
        stats = layer_stats[site_id]
        if stats.per_channel_std is None or stats.per_channel_mean is None:
            logger.warning(
                "Per-channel stats missing for %s; falling back to global.", site_id,
            )
            sigma = stats.std
            if sigma == 0.0:
                return
            mean = stats.mean
            mask = _build_zeroing_mask(tensor, sigma_k, sigma, mean)
        elif ablation_mode == "mean_only":
            # Per-channel μ_c, global σ
            d_ch = len(stats.per_channel_std)
            mask = _build_per_channel_zeroing_mask(
                tensor, sigma_k, [stats.std] * d_ch,
                stats.per_channel_mean, device=tensor.device,
            )
        elif ablation_mode == "var_only":
            # Global μ, per-channel σ_c
            d_ch = len(stats.per_channel_std)
            mask = _build_per_channel_zeroing_mask(
                tensor, sigma_k, stats.per_channel_std,
                [stats.mean] * d_ch, device=tensor.device,
            )
        else:
            # "outlier" mode: full per-channel μ_c and σ_c
            mask = _build_per_channel_zeroing_mask(
                tensor, sigma_k, stats.per_channel_std, stats.per_channel_mean,
                device=tensor.device,
            )
        pct_zeroed[site_id] = 100.0 * (~mask).float().mean().item()
    else:
        sigma = layer_stats[site_id].std
        if sigma == 0.0:
            return
        mean = layer_stats[site_id].mean
        mask = _build_zeroing_mask(tensor, sigma_k, sigma, mean)
        pct_zeroed[site_id] = 100.0 * (~mask).float().mean().item()

    block.mlp.act.input = tensor * mask


def _intervene_residual_stream(
    block: Any,
    block_idx: int,
    sigma_k: float,
    layer_stats: dict[str, LayerStats],
    pct_zeroed: dict[str, float],
    random_fractions: dict[str, float] | None = None,
    random_seed: int | None = None,
) -> None:
    """Zero residual stream outliers for a single encoder block.

    Replaces ``block.norm1.input`` (the residual stream entering this block)
    with a zeroed version.  The CLS token (position 0) is preserved to avoid
    destroying the classification signal.

    Note: for block 0, this is the patch embedding output.  For block i>0,
    this is the output of block i-1.

    If ``random_fractions`` is provided, zeros a random subset of elements
    matching the given fraction instead of using the outlier threshold.

    Called inside an nnsight trace context — ``block`` is an nnsight proxy.

    Parameters
    ----------
    block:
        Nnsight proxy for ``wrapped_model.blocks[i]``.
    block_idx:
        Index of this encoder block (0-based).
    sigma_k:
        Threshold multiplier.
    layer_stats:
        Per-site statistics from Phase 1.
    pct_zeroed:
        Dict mutated in-place with ``site_identifier`` → percentage zeroed.
    random_fractions:
        If provided, maps ``site_identifier`` → fraction (0–1) of elements
        to zero randomly.  Overrides the outlier threshold logic.
    random_seed:
        Seed for the random mask generator.
    """
    site_id = (
        "patch_embed/residual_stream" if block_idx == 0
        else f"blocks.{block_idx - 1}/residual_stream"
    )
    if site_id not in layer_stats:
        return

    tensor = block.norm1.input  # (B, N, D)

    if random_fractions is not None and site_id in random_fractions:
        mask = _build_random_mask(tensor, random_fractions[site_id], random_seed, block_idx)
        pct_zeroed[site_id] = 100.0 * random_fractions[site_id]
    else:
        sigma = layer_stats[site_id].std
        if sigma == 0.0:
            return
        mean = layer_stats[site_id].mean
        mask = _build_zeroing_mask(tensor, sigma_k, sigma, mean)
        pct_zeroed[site_id] = 100.0 * (~mask).float().mean().item()

    # Preserve the CLS token (position 0 along the token dimension).
    # Zeroing the CLS token would destroy the classification signal
    # regardless of outlier status — this is not what we're measuring.
    # Ref: Wei et al. (2022), arXiv:2209.13325, §3.1.
    mask[:, 0, :] = True

    block.norm1.input = tensor * mask


def _intervene_pre_softmax(
    block: Any,
    attn_module: Any,
    block_idx: int,
    sigma_k: float,
    layer_stats: dict[str, LayerStats],
    pct_zeroed: dict[str, float],
    entropy_data: dict[str, dict[str, list[float]]],
) -> None:
    """Zero pre-softmax attention logit outliers for a single encoder block.

    Reconstructs QKᵀ/√d from ``attn.qkv.output`` using the exact same
    computation as Phase 1's ``_register_pre_softmax_saves`` (profiler.py
    lines 1226-1237), applies the zeroing mask, recomputes attention, and
    injects the modified output via ``attn.proj.input``.

    Also captures per-head CLS and patch attention entropy on the zeroed
    attention weights for entropy delta computation against Phase 1 baselines.

    Called inside an nnsight trace context — ``block`` is an nnsight proxy.

    Parameters
    ----------
    block:
        Nnsight proxy for ``wrapped_model.blocks[i]``.
    attn_module:
        Concrete ``Attention`` module from ``inner_model.blocks[i].attn``.
        Used to read architecture constants (num_heads, head_dim, scale).
    block_idx:
        Index of this encoder block (0-based).
    sigma_k:
        Threshold multiplier.
    layer_stats:
        Per-site statistics from Phase 1.
    pct_zeroed:
        Dict mutated in-place with ``site_identifier`` → percentage zeroed.
    entropy_data:
        Dict mutated in-place with ``site_identifier`` →
        ``{"cls": [H], "patch": [H]}`` of per-head entropy values.
    """
    site_id = f"blocks.{block_idx}/pre_softmax"
    if site_id not in layer_stats:
        return

    sigma = layer_stats[site_id].std
    if sigma == 0.0:
        return

    mean = layer_stats[site_id].mean
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    scale = attn_module.scale

    # Reconstruct QKᵀ/√d from qkv.output.
    # Mirrors Phase 1's _register_pre_softmax_saves (profiler.py L1226-1237)
    # and timm's own non-fused Attention.forward() path exactly.
    qkv = block.attn.qkv.output  # (B, N, 3*H*D)
    B = qkv.shape[0]
    N = qkv.shape[1]
    b_n_3hd = qkv.reshape(B, N, 3, num_heads, head_dim)
    b_n_3hd = b_n_3hd.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
    q_scaled = b_n_3hd[0] * scale  # (B, H, N, D) — scaled queries
    k = b_n_3hd[1]                  # (B, H, N, D) — keys
    v = b_n_3hd[2]                  # (B, H, N, D) — values
    attn_logits = q_scaled @ k.transpose(-2, -1)  # (B, H, N, N)

    # Zero outliers in the logit matrix (mean-centered, consistent with Phase 1).
    mask = _build_zeroing_mask(attn_logits, sigma_k, sigma, mean)
    pct_zeroed[site_id] = 100.0 * (~mask).float().mean().item()
    zeroed_logits = attn_logits * mask

    # Recompute attention weights.
    attn_weights = torch.softmax(zeroed_logits, dim=-1)  # (B, H, N, N)

    # Capture per-head entropy on the zeroed attention weights.
    # Uses torch.special.entr for consistency with Phase 1 (T-017).
    per_query_entropy = torch.special.entr(attn_weights).sum(dim=-1)  # (B, H, N)
    cls_entropy = per_query_entropy[:, :, 0].mean(dim=0)  # (H,) — mean over batch
    patch_entropy = per_query_entropy[:, :, 1:].mean(dim=(0, 2))  # (H,) — mean over batch & patches
    entropy_data[site_id] = {
        "cls": cls_entropy.tolist(),
        "patch": patch_entropy.tolist(),
    }

    # Recompute attention output and inject.
    attn_output = attn_weights @ v  # (B, H, N, D)
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, num_heads * head_dim)

    # Replace the attention module's output by writing to proj.input.
    # The proj linear layer will process our modified attention output.
    block.attn.proj.input = attn_output


def save_ablation_results(results: list[AblationResult], path: Path) -> None:
    """Persist ablation results to a CSV file.

    The CSV includes a header row matching the fields of :class:`AblationResult`.
    Parent directories are created if they do not exist.
    Entropy fields are serialised as JSON strings within CSV cells.

    Parameters
    ----------
    results:
        List of :class:`AblationResult` instances, one per (site, threshold,
        site_identifier) combination.
    path:
        Destination file path (e.g. ``output_dir / "ablation_results.csv"``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "site", "sigma_threshold", "site_identifier",
        "pct_zeroed", "top1_accuracy", "top5_accuracy",
        "baseline_top1", "baseline_top5", "seed", "is_random", "granularity",
        "ablation_mode",
        "cls_entropy", "patch_entropy",
        "baseline_cls_entropy", "baseline_patch_entropy",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "site": r.site,
                "sigma_threshold": r.sigma_threshold,
                "site_identifier": r.site_identifier,
                "pct_zeroed": r.pct_zeroed,
                "top1_accuracy": r.top1_accuracy,
                "top5_accuracy": r.top5_accuracy,
                "baseline_top1": r.baseline_top1,
                "baseline_top5": r.baseline_top5,
                "seed": r.seed,
                "is_random": r.is_random,
                "granularity": r.granularity,
                "ablation_mode": r.ablation_mode,
                "cls_entropy": json.dumps(r.cls_entropy),
                "patch_entropy": json.dumps(r.patch_entropy),
                "baseline_cls_entropy": json.dumps(r.baseline_cls_entropy),
                "baseline_patch_entropy": json.dumps(r.baseline_patch_entropy),
            })

    logger.info("Saved %d ablation results to %s", len(results), path)


def save_entropy_deltas(
    results: list[AblationResult],
    path: Path,
) -> None:
    """Persist per-layer entropy deltas for pre_softmax ablation to a CSV file.

    Filters results to ``site == "pre_softmax"`` and writes one row per
    (sigma_threshold, site_identifier) with mean CLS and patch entropy deltas.

    Parameters
    ----------
    results:
        Full list of :class:`AblationResult` from Phase 2.
    path:
        Destination file path (e.g. ``output_dir / "entropy_deltas.csv"``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    pre_softmax_results = [r for r in results if r.site == "pre_softmax" and not r.is_random]
    if not pre_softmax_results:
        logger.info("No pre_softmax results; skipping entropy delta CSV.")
        return

    fieldnames = [
        "sigma_threshold", "site_identifier",
        "mean_cls_delta", "mean_patch_delta",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pre_softmax_results:
            delta = compute_entropy_delta(
                r.cls_entropy, r.patch_entropy,
                r.baseline_cls_entropy, r.baseline_patch_entropy,
            )
            writer.writerow({
                "sigma_threshold": r.sigma_threshold,
                "site_identifier": r.site_identifier,
                "mean_cls_delta": delta["mean_cls_delta"],
                "mean_patch_delta": delta["mean_patch_delta"],
            })

    logger.info("Saved %d entropy delta rows to %s", len(pre_softmax_results), path)