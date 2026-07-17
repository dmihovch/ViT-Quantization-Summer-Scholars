"""Model loading and accuracy evaluation utilities.

Centralises all interactions with ``timm`` so the experiment scripts never
import ``timm`` directly.  The transform is always derived from the model's
own pretrained config to avoid accidental mean/std mismatches.
"""

from __future__ import annotations

import logging
from typing import Callable

import timm
import timm.data
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_vit(device: torch.device) -> tuple[VisionTransformer, Callable]:
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
    raise NotImplementedError


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
    raise NotImplementedError
