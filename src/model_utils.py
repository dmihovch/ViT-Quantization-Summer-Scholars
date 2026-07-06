"""
Model loading and manipulation utilities for ViT-B/16.
"""

from collections.abc import Iterator
from enum import Enum
from typing import Dict, NamedTuple, Tuple

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn


class LayerType(Enum):
    """Categorizes a layer as either part of an Attention block or an MLP block."""

    ATTENTION_QKV = "Attention_QKV"
    ATTENTION_PROJ = "Attention_proj"
    FEEDFORWARD_FC1 = "FeedForward_fc1"
    FEEDFORWARD_FC2 = "FeedForward_fc2"
    OTHER = "Other"


# Type alias for the specific transform required by the ViT model.
ImageTransform = nn.Module


class ModelAndTransform(NamedTuple):
    model: VisionTransformer
    transform: ImageTransform


def load_vit_model() -> ModelAndTransform:
    """
    Loads the pre-trained ViT-B/16 model and its corresponding image transform.
    """
    model_name = "vit_base_patch16_224.orig_in21k_ft_in1k"
    # resolve_data_config is a timm helper that fetches the correct transform for a given model
    # create_transform is a timm helper that constructs the transform
    # See: https://huggingface.co/docs/timm/main/en/feature_extraction#getting-image-preprocessing-for-a-model
    model = timm.create_model(model_name, pretrained=True)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    transform = timm.data.create_transform(**data_config)
    return ModelAndTransform(model, transform)


def load_vit_b_16(device: torch.device) -> Tuple[VisionTransformer, ImageTransform]:
    """
    Load ViT-B/16, move it to `device`, and return (model, transform).

    This is the canonical entry point for Experiment 1. It wraps
    :func:`load_vit_model` and handles the device placement so callers
    get a ready-to-use model.
    """
    model, transform = load_vit_model()
    model.to(device)
    return model, transform


def classify_linear_layer(layer_name: str) -> LayerType:
    """
    Infer the :class:`LayerType` from a linear layer's fully-qualified name.

    This is the single canonical classification function; the private copies
    in the test files should be replaced with calls to this.
    """
    if "attn" in layer_name:
        if "qkv" in layer_name:
            return LayerType.ATTENTION_QKV
        return LayerType.ATTENTION_PROJ
    if "mlp.fc1" in layer_name:
        return LayerType.FEEDFORWARD_FC1
    if "mlp.fc2" in layer_name:
        return LayerType.FEEDFORWARD_FC2
    return LayerType.OTHER


def iter_measured_modules(
    model: VisionTransformer,
) -> Iterator[Tuple[str, nn.Linear, LayerType]]:
    """
    Yield every ``nn.Linear`` in the model as ``(name, module, layer_type)``.

    Layers are yielded in model traversal order (early → late), which is the
    natural network order for per-layer charts and summaries.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module, classify_linear_layer(name)


def get_linear_layers(model: VisionTransformer) -> Dict[str, nn.Linear]:
    """
    Finds all nn.Linear layers in the Vision Transformer and returns them
    in a dictionary with their fully-qualified names.
    """
    linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module
    return linear_layers


def evaluate_top1_accuracy(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate Top-1 accuracy of ``model`` on ``data_loader``.

    The loader must yield ``(images, targets)`` tuples.  The model is put in
    eval mode and run under ``torch.no_grad()``.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device)
            target = target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total
