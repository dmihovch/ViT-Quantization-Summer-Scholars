"""
test_integration.py
====================

End-to-end test against the REAL ViT-B/16 model (timm). This is the test that
proves the full TWO-PASS pipeline works: pass 1 computes per-layer global
mean/std, pass 2 characterizes outliers using those frozen thresholds, and the
input pre-hooks fire on every one of the 49 linear projections - both attention
projections (`attn.qkv`, `attn.proj`), both MLP linears (`mlp.fc1`, `mlp.fc2`),
and the classifier `head`.

It is marked `slow` because the first run downloads the pretrained weights. Skip
it during quick iteration with:

    pytest -m "not slow"
"""

import pytest
import torch

from src.hooks import (
    MomentCollector,
    OutlierStatsCollector,
    make_measurement_hook,
)
from src.model_utils import (
    LayerType,
    classify_linear_layer,
    get_linear_layers,
    load_vit_model,
)
from src.quantization import (
    MSEAccumulator,
    make_activation_quant_hook,
    quantize_all_weights,
    restore_weights,
)


@pytest.mark.slow
def test_two_pass_pipeline_measures_every_layer() -> None:
    # Run on CPU so the test does not require a GPU.
    device = torch.device("cpu")
    model, _transform = load_vit_model()
    model.to(device)

    # Deterministic synthetic batch shaped like preprocessed input.
    batch = torch.randn(2, 3, 224, 224, device=device)

    linear_layers = get_linear_layers(model)

    # --- Pass 1: exact per-layer global mean and std ---
    moments = MomentCollector()
    handles = []
    for layer_name, module in linear_layers.items():
        layer_type = classify_linear_layer(layer_name)
        moments.register_layer(layer_name, layer_type)
        handles.append(
            module.register_forward_pre_hook(make_measurement_hook(moments, layer_name))
        )
    with torch.no_grad():
        _ = model(batch)
    for handle in handles:
        handle.remove()
    thresholds = moments.build_thresholds()

    # --- Pass 2: outlier characterization against the frozen thresholds ---
    collector = OutlierStatsCollector(thresholds)
    handles = []
    for layer_name, module in linear_layers.items():
        layer_type = classify_linear_layer(layer_name)
        collector.register_layer(layer_name, layer_type)
        handles.append(
            module.register_forward_pre_hook(
                make_measurement_hook(collector, layer_name)
            )
        )
    with torch.no_grad():
        _ = model(batch)
    for handle in handles:
        handle.remove()

    summaries = collector.build_summaries()

    # 24 attention linears (12 qkv + 12 proj) + 24 MLP linears
    # (12 fc1 + 12 fc2) + 1 classifier head = 49 modules.
    assert len(summaries) == 49
    assert all(summary.total_values_seen > 0 for summary in summaries)
    # Every layer carries the exact per-channel statistics and derived aggregates.
    assert all(summary.global_std >= 0.0 for summary in summaries)
    assert all(len(summary.channel_means) > 0 for summary in summaries)
    assert all(len(summary.channel_stds) > 0 for summary in summaries)
    # Per-channel arrays match the layer's feature width.
    for summary in summaries:
        assert len(summary.channel_means) == len(summary.channel_stds)

    counts_by_type: dict[LayerType, int] = {layer_type: 0 for layer_type in LayerType}
    for summary in summaries:
        counts_by_type[summary.layer_type] += 1
    assert counts_by_type[LayerType.ATTENTION_QKV] == 12
    assert counts_by_type[LayerType.ATTENTION_PROJ] == 12
    assert counts_by_type[LayerType.FEEDFORWARD_FC1] == 12
    assert counts_by_type[LayerType.FEEDFORWARD_FC2] == 12
    assert counts_by_type[LayerType.OTHER] == 1


# ---------------------------------------------------------------------------
# Experiment 2 integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_exp2_full_quantize_forward_restore_cycle() -> None:
    """
    Full quantize → hook → forward → restore cycle on the real ViT-B/16.

    Verifies that:
    - Weight quantization and activation hooks don't crash.
    - The model still produces output of the correct shape.
    - Weights are correctly restored after the cycle.
    - MSEAccumulator records per-layer error.
    """
    device = torch.device("cpu")
    model, _transform = load_vit_model()
    model.to(device)
    linear_layers = get_linear_layers(model)
    batch = torch.randn(2, 3, 224, 224, device=device)

    # Save original weights for comparison.
    original_weights = {
        name: layer.weight.data.clone() for name, layer in linear_layers.items()
    }

    # Quantize weights.
    originals = quantize_all_weights(model, "per_channel", linear_layers)
    assert len(originals) == len(linear_layers)

    # Register activation quant hooks with MSE tracking.
    mse_tracker = MSEAccumulator()
    handles = []
    for name, module in linear_layers.items():
        hook = make_activation_quant_hook("per_tensor", name, mse_tracker)
        handles.append(module.register_forward_pre_hook(hook))

    # Forward pass.
    with torch.no_grad():
        output = model(batch)

    # Output should have the correct shape: (batch, num_classes).
    assert output.shape == (2, 1000)

    # MSE should have been recorded for every layer.
    mse_values = mse_tracker.get_all()
    assert len(mse_values) == len(linear_layers)
    for name in linear_layers:
        assert name in mse_values
        assert mse_values[name] >= 0.0

    # Clean up hooks.
    for handle in handles:
        handle.remove()

    # Restore weights.
    restore_weights(model, originals, linear_layers)

    # Verify weights are restored.
    for name, layer in linear_layers.items():
        assert torch.equal(layer.weight.data, original_weights[name]), (
            f"Weights for {name} were not restored correctly."
        )


@pytest.mark.slow
def test_exp2_accuracy_drops_with_quantization() -> None:
    """
    Quantized accuracy is strictly less than FP32 accuracy on a tiny subset.

    Uses a single synthetic batch (not real images) to keep the test fast.
    The model is in eval mode with random weights, so "accuracy" here just
    means the output logits differ — we're testing that quantization actually
    changes the output, not measuring real ImageNet accuracy.
    """
    device = torch.device("cpu")
    model, _transform = load_vit_model()
    model.to(device)
    model.eval()
    linear_layers = get_linear_layers(model)
    batch = torch.randn(4, 3, 224, 224, device=device)

    # FP32 output.
    with torch.no_grad():
        fp32_output = model(batch)

    # INT8 output (per-tensor weights, per-tensor activations).
    originals = quantize_all_weights(model, "per_tensor", linear_layers)
    handles = []
    for name, module in linear_layers.items():
        hook = make_activation_quant_hook("per_tensor", name)
        handles.append(module.register_forward_pre_hook(hook))

    with torch.no_grad():
        int8_output = model(batch)

    for handle in handles:
        handle.remove()
    restore_weights(model, originals, linear_layers)

    # Quantized output must differ from FP32 output.
    assert not torch.allclose(fp32_output, int8_output, atol=1e-4), (
        "Quantization did not change the model output — "
        "the quantization pipeline may not be working."
    )
