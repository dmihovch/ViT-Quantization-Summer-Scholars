"""
Core components for simulated Post-Training Quantization (PTQ).

This module provides functions for fake-quantizing tensors to INT8 using various
granularity strategies (per-tensor, per-channel, per-token). "Fake quantization"
means the tensor is quantized and then immediately de-quantized, so the output
tensor has the same float dtype as the input but has lost precision.

The quantization is symmetric, mapping the range `[-absmax, absmax]` to `[-127, 127]`.
An offset of 128 is used for the zero point to support unsigned hardware, but the
quantization itself is symmetric around zero.
"""

from typing import Callable, Dict, Tuple

import torch
from torch import nn

# Following PyTorch's convention for quantized tensors.
# We simulate INT8, which has a range of [-128, 127].
# We use a symmetric range [-127, 127] for our mapping.
QMIN = -127
QMAX = 127


def _calculate_scale_zeropoint(
    x: torch.Tensor,
    qmin: int,
    qmax: int,
    dim: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the scale and zero-point for symmetric quantization.

    When ``dim`` is provided, the reduction is over that dimension (keeping
    it for broadcasting), so each slice along that axis gets its own scale.
    """
    if dim is not None:
        absmax = torch.max(torch.abs(x), dim=dim, keepdim=True)[0]
    else:
        absmax = torch.max(torch.abs(x))

    scale = absmax / ((qmax - qmin) / 2)
    zeropoint = torch.zeros_like(scale)  # Symmetric quantization
    return scale, zeropoint


def quantize_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zeropoint: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Quantizes and then dequantizes a tensor."""
    # Quantize
    x_q = torch.round(x / scale) + zeropoint
    x_q = torch.clamp(x_q, qmin, qmax)

    # Dequantize
    x_dq = (x_q - zeropoint) * scale
    return x_dq


def quantize_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Applies per-tensor symmetric quantization to a tensor.

    One scale and zero-point is used for the entire tensor.

    Args:
        x: The input float tensor.

    Returns:
        The fake-quantized float tensor.
    """
    scale, zeropoint = _calculate_scale_zeropoint(x, QMIN, QMAX)
    return quantize_dequantize(x, scale, zeropoint, QMIN, QMAX)


def quantize_per_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies per-channel symmetric quantization, for weights.

    One scale and zero-point is used per output channel (dim 0).

    Args:
        x: The input weight tensor of shape (out_channels, ...).

    Returns:
        The fake-quantized float tensor.
    """
    if x.dim() < 2:
        raise ValueError("Per-channel quantization requires at least a 2D tensor.")

    scale, zeropoint = _calculate_scale_zeropoint(x, QMIN, QMAX, dim=0)
    return quantize_dequantize(x, scale, zeropoint, QMIN, QMAX)


def quantize_per_token(x: torch.Tensor) -> torch.Tensor:
    """
    Applies per-token symmetric quantization, for activations.

    One scale and zero-point is used per token in the sequence. It treats
    the activation tensor of shape (batch, tokens, features) as a 2D
    matrix of shape (batch * tokens, features) for quantization.

    Args:
        x: The input activation tensor of shape (batch, tokens, features).

    Returns:
        The fake-quantized float tensor.
    """
    if x.dim() != 3:
        raise ValueError(
            "Per-token quantization requires a 3D tensor (batch, tokens, features)."
        )

    original_shape = x.shape
    x_reshaped = x.view(-1, original_shape[-1])

    # Calculate scale and zeropoint for each token's feature vector.
    # After reshape to (batch*tokens, features), dim=1 reduces over features
    # so each token gets its own scale.
    scale, zeropoint = _calculate_scale_zeropoint(x_reshaped, QMIN, QMAX, dim=1)

    # Reshape scale and zeropoint to be broadcastable with the original tensor
    scale = scale.view(original_shape[0], original_shape[1], 1)
    zeropoint = zeropoint.view(original_shape[0], original_shape[1], 1)

    return quantize_dequantize(x, scale, zeropoint, QMIN, QMAX)


# ---------------------------------------------------------------------------
# Experiment 2: full-model quantization helpers
# ---------------------------------------------------------------------------


def quantize_all_weights(
    model: nn.Module,
    strategy: str,
    linear_layers: Dict[str, nn.Linear] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Quantize every nn.Linear weight in-place and return copies of the originals.

    Args:
        model: The model whose linear layers will be quantized.
        strategy: One of ``"per_tensor"`` or ``"per_channel"``.
        linear_layers: Optional pre-collected dict of {name: nn.Linear}.
            If None, discovered via ``model.named_modules()``.

    Returns:
        A dict mapping layer name to the original (unquantized) weight tensor,
        suitable for passing to :func:`restore_weights`.
    """
    if strategy not in ("per_tensor", "per_channel"):
        raise ValueError(
            f"Unknown weight strategy '{strategy}'. Expected 'per_tensor' or 'per_channel'."
        )

    if linear_layers is None:
        linear_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

    originals: Dict[str, torch.Tensor] = {}
    quant_fn = (
        quantize_per_channel if strategy == "per_channel" else quantize_per_tensor
    )

    for name, layer in linear_layers.items():
        originals[name] = layer.weight.data.clone()
        layer.weight.data = quant_fn(layer.weight.data)

    return originals


def restore_weights(
    model: nn.Module,
    originals: Dict[str, torch.Tensor],
    linear_layers: Dict[str, nn.Linear] | None = None,
) -> None:
    """
    Restore original (unquantized) weights from the dict returned by
    :func:`quantize_all_weights`.

    Args:
        model: The model whose linear layers will be restored.
        originals: Dict of {name: original_weight_tensor}.
        linear_layers: Optional pre-collected dict. If None, discovered
            via ``model.named_modules()``.
    """
    if linear_layers is None:
        linear_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

    for name, original_weight in originals.items():
        if name in linear_layers:
            linear_layers[name].weight.data = original_weight


class MSEAccumulator:
    """
    Tracks per-layer mean squared error between original and quantized
    activations across multiple forward passes.

    Accumulates ``(sum_sq_error, count)`` per layer so the final MSE is a
    correctly weighted mean over all tokens across all batches.
    """

    def __init__(self) -> None:
        self._sum_sq_error: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def record(
        self, name: str, original: torch.Tensor, quantized: torch.Tensor
    ) -> None:
        """
        Record the squared error for one layer's activation.

        Both tensors are expected to have the same shape. The error is
        summed over all elements and accumulated.
        """
        sq_error = ((original - quantized) ** 2).sum().item()
        num_elements = original.numel()

        if name not in self._sum_sq_error:
            self._sum_sq_error[name] = 0.0
            self._count[name] = 0

        self._sum_sq_error[name] += sq_error
        self._count[name] += num_elements

    def get_all(self) -> Dict[str, float]:
        """
        Return per-layer MSE as ``{layer_name: mean_squared_error}``.

        Layers that were never recorded return 0.0.
        """
        return {
            name: self._sum_sq_error[name] / self._count[name]
            if self._count[name] > 0
            else 0.0
            for name in self._sum_sq_error
        }


def make_activation_quant_hook(
    strategy: str,
    layer_name: str,
    mse_tracker: MSEAccumulator | None = None,
) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], None]:
    """
    Build a forward pre-hook that quantizes the input activation before the
    matmul runs.

    The hook is designed to be registered via
    ``module.register_forward_pre_hook(...)``. It modifies ``args[0]`` in-place
    (via tuple reassignment) so the quantized activation is what the layer's
    ``forward()`` sees.

    Args:
        strategy: ``"per_tensor"`` or ``"per_token"``.
        layer_name: Human-readable name for the layer (used for MSE tracking).
        mse_tracker: Optional :class:`MSEAccumulator` to record per-layer
            quantization error.

    Returns:
        A callable suitable for ``register_forward_pre_hook``.
    """
    if strategy not in ("per_tensor", "per_token"):
        raise ValueError(
            f"Unknown activation strategy '{strategy}'. Expected 'per_tensor' or 'per_token'."
        )

    def hook(
        module: nn.Module, args: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...] | None:
        x = args[0]

        if strategy == "per_token" and x.dim() == 3:
            x_q = quantize_per_token(x)
        else:
            # Per-tensor, or per-token on a 2D input (e.g., classifier head
            # after pooling) — fall back to per-tensor.
            x_q = quantize_per_tensor(x)

        if mse_tracker is not None:
            mse_tracker.record(layer_name, x.detach(), x_q.detach())

        # Return modified args tuple — the quantized activation replaces the
        # original as the first positional argument.
        return (x_q,) + args[1:]

    return hook
