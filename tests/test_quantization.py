"""
Unit tests for the quantization module.
"""

import pytest
import torch
from torch import nn

from src.quantization import (
    QMAX,
    QMIN,
    MSEAccumulator,
    make_activation_quant_hook,
    quantize_all_weights,
    quantize_per_channel,
    quantize_per_tensor,
    quantize_per_token,
    restore_weights,
)


@pytest.fixture
def example_tensor() -> torch.Tensor:
    """A simple, non-trivial tensor for testing."""
    return torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=torch.float32)


@pytest.fixture
def activation_tensor() -> torch.Tensor:
    """A 3D tensor simulating a batch of activations."""
    return torch.randn(4, 197, 768, dtype=torch.float32)  # Batch, Tokens, Features


@pytest.fixture
def weight_tensor() -> torch.Tensor:
    """A 2D tensor simulating a weight matrix."""
    return torch.randn(1024, 768, dtype=torch.float32)  # Out_features, In_features


def test_per_tensor_quantization(example_tensor: torch.Tensor):
    """Verify per-tensor quantization produces a tensor of the same shape and type."""
    quantized_tensor = quantize_per_tensor(example_tensor)

    assert quantized_tensor.shape == example_tensor.shape
    assert quantized_tensor.dtype == example_tensor.dtype

    # The maximum absolute value should be preserved (or close to it)
    absmax = torch.max(torch.abs(example_tensor))
    scale = absmax / ((QMAX - QMIN) / 2)

    # Check that values are multiples of the scale by dividing and checking for roundness
    dequantized_divided_by_scale = quantized_tensor / scale
    assert torch.allclose(
        dequantized_divided_by_scale,
        torch.round(dequantized_divided_by_scale),
        atol=1e-6,
    )

    # Check that values are within the quantized range
    assert torch.max(torch.abs(quantized_tensor)) <= absmax + 1e-6


def test_per_channel_quantization(weight_tensor: torch.Tensor):
    """Verify per-channel quantization for weights."""
    quantized_tensor = quantize_per_channel(weight_tensor)

    assert quantized_tensor.shape == weight_tensor.shape
    assert quantized_tensor.dtype == weight_tensor.dtype

    # Check that each output channel (dim 0) has its own scale.
    absmax_per_channel = torch.max(torch.abs(weight_tensor), dim=0, keepdim=True)[0]
    scales = absmax_per_channel / ((QMAX - QMIN) / 2)  # [1, in_features]

    # Check that values across each output channel are multiples of that
    # channel's scale and are within the quantized range.
    for i in range(weight_tensor.shape[1]):
        dequantized_divided_by_scale = quantized_tensor[:, i] / scales[0, i]
        assert torch.allclose(
            dequantized_divided_by_scale,
            torch.round(dequantized_divided_by_scale),
            atol=1e-6,
        )
        assert (
            torch.max(torch.abs(quantized_tensor[:, i]))
            <= absmax_per_channel[0, i] + 1e-6
        )


def test_per_channel_quantization_raises_on_1d_tensor():
    """Per-channel quantization is not defined for 1D tensors."""
    with pytest.raises(ValueError):
        quantize_per_channel(torch.randn(10))


def test_per_token_quantization(activation_tensor: torch.Tensor):
    """Verify per-token quantization for activations."""
    quantized_tensor = quantize_per_token(activation_tensor)

    assert quantized_tensor.shape == activation_tensor.shape
    assert quantized_tensor.dtype == activation_tensor.dtype

    # Reshape and check scale per token
    reshaped_activation = activation_tensor.view(-1, activation_tensor.shape[-1])
    absmax_per_token = torch.max(torch.abs(reshaped_activation), dim=1, keepdim=True)[0]
    scales = absmax_per_token / ((QMAX - QMIN) / 2)
    scales = scales.view(activation_tensor.shape[0], activation_tensor.shape[1], 1)

    # Check that values for each token are multiples of their scale
    for i in range(activation_tensor.shape[0]):
        for j in range(activation_tensor.shape[1]):
            dequantized_divided_by_scale = quantized_tensor[i, j] / scales[i, j]
            assert torch.allclose(
                dequantized_divided_by_scale,
                torch.round(dequantized_divided_by_scale),
                atol=1e-6,
            )
            assert (
                torch.max(torch.abs(quantized_tensor[i, j]))
                <= torch.max(torch.abs(activation_tensor[i, j])) + 1e-6
            )


def test_per_token_quantization_raises_on_non_3d_tensor():
    """Per-token quantization is only defined for 3D tensors."""
    with pytest.raises(ValueError):
        quantize_per_token(torch.randn(10, 10))
    with pytest.raises(ValueError):
        quantize_per_token(torch.randn(10))


# ---------------------------------------------------------------------------
# Experiment 2: quantize_all_weights / restore_weights
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    """A tiny model with two linear layers for testing weight quant/restore."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)


def test_quantize_all_weights_and_restore_roundtrip():
    """Weights after quantize → restore are identical to originals."""
    model = _DummyModel()
    original_fc1 = model.fc1.weight.data.clone()
    original_fc2 = model.fc2.weight.data.clone()

    originals = quantize_all_weights(model, "per_tensor")

    # After quantization, weights should differ from originals.
    assert not torch.equal(model.fc1.weight.data, original_fc1)
    assert not torch.equal(model.fc2.weight.data, original_fc2)

    restore_weights(model, originals)

    # After restore, weights should be identical.
    assert torch.equal(model.fc1.weight.data, original_fc1)
    assert torch.equal(model.fc2.weight.data, original_fc2)


def test_quantize_all_weights_per_channel():
    """Per-channel weight quantization runs without error."""
    model = _DummyModel()
    originals = quantize_all_weights(model, "per_channel")
    assert len(originals) == 2
    restore_weights(model, originals)


def test_quantize_all_weights_rejects_unknown_strategy():
    """Unknown weight strategy raises ValueError."""
    model = _DummyModel()
    with pytest.raises(ValueError, match="Unknown weight strategy"):
        quantize_all_weights(model, "per_token")


def test_restore_weights_ignores_missing_layers():
    """Restoring weights for layers not in the model is a no-op."""
    model = _DummyModel()
    originals = {"nonexistent_layer": torch.randn(4, 4)}
    # Should not raise.
    restore_weights(model, originals)


# ---------------------------------------------------------------------------
# Experiment 2: make_activation_quant_hook
# ---------------------------------------------------------------------------


def test_activation_quant_hook_per_tensor():
    """Hook quantizes a 3D activation with per-tensor strategy."""
    hook = make_activation_quant_hook("per_tensor", "test_layer")
    x = torch.randn(2, 10, 16)
    module = nn.Linear(16, 8)

    result = hook(module, (x,))
    assert result is not None
    x_q = result[0]

    assert x_q.shape == x.shape
    assert x_q.dtype == x.dtype
    # Quantized values should differ from original (unless all zeros).
    assert not torch.allclose(x_q, x)


def test_activation_quant_hook_per_token():
    """Hook quantizes a 3D activation with per-token strategy."""
    hook = make_activation_quant_hook("per_token", "test_layer")
    x = torch.randn(2, 10, 16)
    module = nn.Linear(16, 8)

    result = hook(module, (x,))
    assert result is not None
    x_q = result[0]

    assert x_q.shape == x.shape
    assert x_q.dtype == x.dtype
    assert not torch.allclose(x_q, x)


def test_activation_quant_hook_2d_fallback():
    """Per-token on 2D input falls back to per-tensor (no crash)."""
    hook = make_activation_quant_hook("per_token", "test_layer")
    x = torch.randn(32, 16)  # 2D, like classifier head after pooling
    module = nn.Linear(16, 8)

    result = hook(module, (x,))
    assert result is not None
    x_q = result[0]

    assert x_q.shape == x.shape
    assert x_q.dtype == x.dtype


def test_activation_quant_hook_rejects_unknown_strategy():
    """Unknown activation strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown activation strategy"):
        make_activation_quant_hook("per_channel", "test_layer")


def test_activation_quant_hook_with_mse_tracker():
    """MSEAccumulator records per-layer error correctly."""
    tracker = MSEAccumulator()
    hook = make_activation_quant_hook("per_tensor", "layer_a", tracker)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    module = nn.Linear(2, 2)

    hook(module, (x,))

    mse = tracker.get_all()
    assert "layer_a" in mse
    assert mse["layer_a"] > 0.0  # Quantization introduces error


# ---------------------------------------------------------------------------
# Experiment 2: MSEAccumulator
# ---------------------------------------------------------------------------


def test_mse_accumulator_correctness():
    """MSEAccumulator computes the correct weighted mean across batches."""
    tracker = MSEAccumulator()

    # Batch 1: 4 elements, error 1.0 each → sum_sq = 4.0
    x1 = torch.ones(2, 2)
    x1_q = torch.zeros(2, 2)
    tracker.record("layer_a", x1, x1_q)

    # Batch 2: 6 elements, error 2.0 each → sum_sq = 24.0
    x2 = torch.ones(2, 3) * 2.0
    x2_q = torch.zeros(2, 3)
    tracker.record("layer_a", x2, x2_q)

    mse = tracker.get_all()
    # Total sum_sq = 4.0 + 24.0 = 28.0, total elements = 4 + 6 = 10
    # MSE = 28.0 / 10 = 2.8
    assert abs(mse["layer_a"] - 2.8) < 1e-6


def test_mse_accumulator_multiple_layers():
    """MSEAccumulator tracks multiple layers independently."""
    tracker = MSEAccumulator()

    x = torch.ones(2, 2)
    x_q = torch.zeros(2, 2)
    tracker.record("layer_a", x, x_q)
    tracker.record("layer_b", x * 2, x_q)

    mse = tracker.get_all()
    assert "layer_a" in mse
    assert "layer_b" in mse
    # layer_a: error = 1.0 per element → MSE = 1.0
    assert abs(mse["layer_a"] - 1.0) < 1e-6
    # layer_b: error = 2.0 per element → MSE = 4.0
    assert abs(mse["layer_b"] - 4.0) < 1e-6


def test_mse_accumulator_empty_returns_zero():
    """get_all() on an unused tracker returns empty dict."""
    tracker = MSEAccumulator()
    assert tracker.get_all() == {}
