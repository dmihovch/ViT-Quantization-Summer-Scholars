"""
Model loading and evaluation utilities for ViT-B/16.
"""

from __future__ import annotations

from typing import Tuple

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn

# Type alias for the specific transform required by the ViT model.
ImageTransform = nn.Module


def load_vit_b_16(device: torch.device) -> Tuple[VisionTransformer, ImageTransform]:
    """Load a pre-trained ViT-B/16 model, place it on ``device``, and return
    ``(model, transform)``.

    The model is returned in eval mode with dropout disabled. The transform
    is the exact preprocessing pipeline the model was trained with, fetched via
    timm's ``resolve_data_config`` / ``create_transform`` helpers.

    Args:
        device: The device to move the model to before returning.

    Returns:
        A (model, transform) tuple ready for inference.
    """
    model_name = "vit_base_patch16_224.orig_in21k_ft_in1k"
    model = timm.create_model(model_name, pretrained=True)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    transform = timm.data.create_transform(**data_config)
    model.eval()
    model.to(device)
    return model, transform


def evaluate_top1_top5_accuracy(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate Top-1 and Top-5 accuracy of ``model`` on ``data_loader``.

    The loader must yield ``(images, targets)`` tuples. The model is kept in
    eval mode and run under ``torch.no_grad()``.

    Args:
        model: The model to evaluate. Must be already on ``device``.
        data_loader: DataLoader yielding (images, targets) batches.
        device: Device that both model and data will live on.

    Returns:
        A (top1_accuracy, top5_accuracy) tuple, both as percentages (0–100).
    """
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            batch_size = targets.size(0)
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += (
                top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds))
                .any(dim=1)
                .sum()
                .item()
            )
            top1_correct += top5_preds[:, 0].eq(targets).sum().item()
            total += batch_size

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5
