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

  1. Self-attention   - projects tokens into Query/Key/Value spaces (a single
                        fused `qkv` linear), then projects the attention result
                        back out (`proj`).
  2. Feed-forward MLP  - two `nn.Linear` layers (`fc1`, `fc2`) with a GELU
                        non-linearity in between. This is where our hypothesis
                        expects DENSE activation outliers.

Why timm (and not torchvision)
------------------------------
We deliberately use `timm`'s ViT implementation rather than torchvision's.
torchvision routes attention through `nn.MultiheadAttention`, whose fused kernel
BYPASSES the internal `out_proj` submodule and fuses Q/K/V into one opaque
weight - so a forward hook can only observe the WHOLE attention block's output,
collapsing four projections into a single measurement point (37 hookable modules
in total).

timm instead exposes every projection as a plain `nn.Linear`:

    blocks.N.attn.qkv   - fused Query/Key/Value projection (hookable)
    blocks.N.attn.proj  - attention output projection      (hookable)
    blocks.N.mlp.fc1    - MLP up-projection                (hookable)
    blocks.N.mlp.fc2    - MLP down-projection              (hookable)

That yields 12 x 4 = 48 in-block linears plus the final classifier `head` = 49
separately hookable layers, matching Experiment 1's goal of characterizing every
linear projection independently. (Q, K and V still share one fused `qkv` weight,
so they cannot be split apart at the module level; if needed, the `qkv` output
tensor can be chunked into three along its feature axis inside the hook.)

Because INT8 quantization happens at these linear layers, they are exactly the
layers we attach measurement hooks to in Experiment 1.
"""

from collections.abc import Callable, Iterator
from enum import Enum
from typing import Any, TypeAlias, cast

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor

# The timm model identifier for ImageNet-1K pretrained ViT-B/16.
VIT_B_16_MODEL_NAME: str = "vit_base_patch16_224"

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

    Examples of the names we match against (timm's ViT naming):
        "blocks.5.attn.qkv"  -> ATTENTION   (fused Q/K/V projection)
        "blocks.5.attn.proj" -> ATTENTION   (attention output projection)
        "blocks.5.mlp.fc1"   -> FEEDFORWARD
        "blocks.5.mlp.fc2"   -> FEEDFORWARD
        "head"               -> OTHER        (final classifier)
    """
    if "attn" in layer_name:
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

    With timm's ViT every projection is a plain `nn.Linear`, so the rule is
    simple: measure every `nn.Linear` in the model. For ViT-B/16 this is

      * 12 x `attn.qkv`  + 12 x `attn.proj`  -> 24 modules tagged ATTENTION
      * 12 x `mlp.fc1`   + 12 x `mlp.fc2`    -> 24 modules tagged FEEDFORWARD
      * 1  x `head`                          ->  1 module  tagged OTHER

    for a total of 49 separately hooked linear layers. Unlike torchvision's
    fused `nn.MultiheadAttention`, hooks on these linears fire reliably because
    each one is invoked as its own submodule during the forward pass.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module, classify_linear_layer(name)


def load_vit_b_16(device: torch.device) -> tuple[VisionTransformer, ImageTransform]:
    """
    Load ViT-B/16 pre-trained on ImageNet-1K (via timm), ready for inference.

    Returns two things:
      * the model, moved onto `device` and switched to eval mode, and
      * the exact preprocessing transform the weights were trained with
        (resize -> center-crop -> to-tensor -> normalize), derived from the
        model's own pretrained data config.

    Using the *matching* transform matters: feeding differently scaled inputs
    would shift the very activation statistics we are trying to measure.
    """
    # `pretrained=True` downloads the ImageNet-1K weights on first use.
    # `create_model` is declared to return a generic `nn.Module`, so we narrow it
    # to the concrete `VisionTransformer` we know we asked for.
    model: VisionTransformer = cast(
        VisionTransformer, timm.create_model(VIT_B_16_MODEL_NAME, pretrained=True)
    )

    # `.to(device)` moves parameters onto the GPU; `.eval()` disables dropout so
    # the activations (and therefore our statistics) are deterministic.
    model = model.to(device).eval()

    # Build the preprocessing transform straight from the model's pretrained
    # config so it always matches the weights we just loaded. timm's factory has
    # an untyped, kitchen-sink signature, so we type the config as `Any` and cast
    # the resulting transform to our `ImageTransform` alias.
    data_config: dict[str, Any] = resolve_data_config({}, model=model)
    transform: ImageTransform = cast(ImageTransform, create_transform(**data_config))
    return model, transform
