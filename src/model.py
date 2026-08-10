"""Model loading and accuracy evaluation utilities.

Centralises all interactions with ``timm`` so the experiment scripts never
import ``timm`` directly.  The transform is always derived from the model's
own pretrained config to avoid accidental mean/std mismatches.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import timm
import timm.data
import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def disable_fused_attn(model: VisionTransformer) -> None:
    """Set ``fused_attn=False`` on every attention block.

    Must be called before wrapping with NNsight.  With ``fused_attn=True``,
    PyTorch dispatches to SDPA / FlashAttention, which never materialises the
    QKᵀ attention-logit matrix as a Python tensor — it therefore cannot be
    captured by any hook or trace.

    Args:
        model: A timm VisionTransformer instance whose ``blocks`` attribute
            contains standard :class:`timm.models.vision_transformer.Block`
            objects.
    """
    for i, block in enumerate(model.blocks):
        block.attn.fused_attn = False
        logger.debug("Disabled fused_attn on block %d", i)
    logger.info("fused_attn disabled on all %d blocks", len(model.blocks))


def load_vit(device: torch.device) -> tuple[VisionTransformer, Callable[[Image.Image], torch.Tensor]]:
    """Load the pretrained ``vit_base_patch16_224`` model and its transform.

    The preprocessing transform is derived exclusively from the model's own
    pretrained data config (via ``timm.data.resolve_data_config`` and
    ``timm.data.create_transform``) so that mean, std, and resize parameters
    are always consistent with the checkpoint.

    Parameters
    ----------
    device:
        Target compute device.  The model is moved to this device before
        being returned.

    Returns
    -------
    tuple[VisionTransformer, Callable]
        A 2-tuple of ``(model, transform)`` where ``model`` is in eval mode
        and ``transform`` is a callable that converts a PIL image to a
        correctly normalised ``torch.Tensor``.

    Raises
    ------
    RuntimeError
        If ``timm`` cannot locate the pretrained weights.
    """
    logger.info("Loading vit_base_patch16_224.augreg2_in21k_ft_in1k pretrained weights...")
    model: VisionTransformer = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    )
    model.eval()
    disable_fused_attn(model)
    model.to(device)

    data_config = timm.data.resolve_data_config({}, model=model)
    transform: Callable = timm.data.create_transform(**data_config)

    logger.info("Model loaded on %s", device)
    return model, transform


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute top-1 and top-5 accuracy over the provided DataLoader.

    All forward passes are executed inside ``torch.no_grad()`` to avoid
    unnecessary gradient computation.

    Parameters
    ----------
    model:
        An ``nn.Module`` already in eval mode and on ``device``.
    loader:
        DataLoader yielding ``(images, labels)`` batches.
    device:
        Compute device that both ``model`` and input tensors reside on.

    Returns
    -------
    tuple[float, float]
        ``(top1_pct, top5_pct)`` as percentages in the range ``[0, 100]``.
    """
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)  # (B, num_classes)

            # top-5 indices for each sample
            top5_preds = outputs.topk(5, dim=1).indices  # (B, 5)
            top1_preds = top5_preds[:, 0]  # (B,)

            correct_top1 += (top1_preds == labels).sum().item()
            correct_top5 += (
                top5_preds == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("DataLoader yielded zero samples; cannot compute accuracy.")

    top1_pct = 100.0 * correct_top1 / total
    top5_pct = 100.0 * correct_top5 / total
    logger.info("Top-1: %.2f%%  Top-5: %.2f%%  (n=%d)", top1_pct, top5_pct, total)
    return top1_pct, top5_pct
