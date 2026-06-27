"""
model_utils.py
==============

Helpers for loading the Vision Transformer we study (ViT-B/16) and for
classifying its internal linear layers.

Mental model of the network
----------------------------
A ViT-B/16 is a stack of 12 identical "transformer blocks". Inside each block
there are two places where large matrix multiplications (the things INT8
quantization targets) happen:

  1. Self-attention   - projects tokens into Query/Key/Value spaces, then
                        projects the attention result back out (`out_proj`).
  2. Feed-forward MLP  - two `nn.Linear` layers with a GELU non-linearity in
                        between. This is where our hypothesis expects DENSE
                        activation outliers.

Because INT8 quantization happens at these linear layers, they are exactly the
layers we attach measurement hooks to in Experiment 1.
"""

from collections.abc import Callable, Iterator
from enum import Enum
from typing import TypeAlias

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision.models import VisionTransformer, ViT_B_16_Weights, vit_b_16

# A preprocessing transform turns one PIL image into a normalized float tensor
# of shape [3, 224, 224]. We give it a name so every module agrees on the type.
ImageTransform: TypeAlias = Callable[[Image.Image], Tensor]


class LayerType(str, Enum):
    """
    A tag describing what role a linear layer plays inside the transformer.

    We inherit from `str` (not just `Enum`) for two practical reasons:
      * the member is already a string, so `json.dump` serializes it directly;
      * its `.value` is the human-readable label we want in charts and files.
    """

    ATTENTION = "Attention_QKV"
    FEEDFORWARD = "FeedForward_MLP"
    OTHER = "Other_Linear"


def classify_linear_layer(layer_name: str) -> LayerType:
    """
    Decide whether a linear layer belongs to the attention block, the
    feed-forward (MLP) block, or neither, based purely on its module path.

    Examples of the names we match against:
        "encoder.layers.encoder_layer_5.self_attention.out_proj" -> ATTENTION
        "encoder.layers.encoder_layer_5.mlp.0"                    -> FEEDFORWARD
        "heads.head"                                              -> OTHER
    """
    if "self_attention" in layer_name:
        return LayerType.ATTENTION
    if "mlp" in layer_name:
        return LayerType.FEEDFORWARD
    return LayerType.OTHER


def iter_measured_modules(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Module, LayerType]]:
    """
    Yield every module we attach a measurement hook to, as
    (name, module, layer_type) triples, by walking the whole module tree.

    We measure two kinds of module:

      * `nn.MultiheadAttention` -> tagged ATTENTION. We hook the WHOLE attention
        sub-block rather than its internal `out_proj` linear layer. This is
        deliberate: torchvision runs attention through a fused kernel that
        BYPASSES the `out_proj` submodule, so a forward hook placed on
        `out_proj` would never fire. The attention module's own output (the
        activation flowing out of the block) is exactly what the next layer -
        and any quantizer - would see.

      * `nn.Linear` that is NOT inside an attention block -> the two MLP layers
        per transformer block (tagged FEEDFORWARD) and the final classifier head
        (tagged OTHER). We skip linears whose name contains "self_attention"
        because those are the un-hookable `out_proj` layers handled above.

    CAVEAT ABOUT Q/K/V
    ------------------
    torchvision fuses the Query/Key/Value projection into a single weight tensor
    (`in_proj_weight`) inside `nn.MultiheadAttention`, so there is no separate
    Q/K/V `nn.Linear` to hook. Our ATTENTION measurement therefore characterizes
    the attention block's OUTPUT activations, not the raw Q/K/V projections.
    In practice this yields 12 attention blocks + 24 MLP linears + 1 head = 37
    measured modules, rather than the idealized 48 separate Q/K/V/O projections.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            yield name, module, LayerType.ATTENTION
        elif isinstance(module, nn.Linear) and "self_attention" not in name:
            yield name, module, classify_linear_layer(name)


def load_vit_b_16(device: torch.device) -> tuple[VisionTransformer, ImageTransform]:
    """
    Load ViT-B/16 pre-trained on ImageNet-1K, ready for inference.

    Returns two things:
      * the model, moved onto `device` and switched to eval mode, and
      * the exact preprocessing transform the weights were trained with
        (resize -> center-crop -> to-tensor -> normalize).

    Using the *matching* transform matters: feeding differently scaled inputs
    would shift the very activation statistics we are trying to measure.
    """
    weights = ViT_B_16_Weights.DEFAULT

    # `.to(device)` moves parameters onto the GPU; `.eval()` disables dropout so
    # the activations (and therefore our statistics) are deterministic.
    model: VisionTransformer = vit_b_16(weights=weights).to(device).eval()

    transform: ImageTransform = weights.transforms()
    return model, transform
