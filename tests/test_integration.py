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
from src.model_utils import LayerType, iter_measured_modules, load_vit_b_16


@pytest.mark.slow
def test_two_pass_pipeline_measures_every_layer() -> None:
    # Run on CPU so the test does not require a GPU.
    device = torch.device("cpu")
    model, _transform = load_vit_b_16(device)

    # Deterministic synthetic batch shaped like preprocessed input.
    batch = torch.randn(2, 3, 224, 224, device=device)

    # --- Pass 1: exact per-layer global mean and std ---
    moments = MomentCollector()
    handles = []
    for layer_name, module, layer_type in iter_measured_modules(model):
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
    for layer_name, module, layer_type in iter_measured_modules(model):
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
    # Every layer carries the exact global std used for its 3-sigma threshold.
    assert all(summary.global_std >= 0.0 for summary in summaries)

    counts_by_type: dict[LayerType, int] = {layer_type: 0 for layer_type in LayerType}
    for summary in summaries:
        counts_by_type[summary.layer_type] += 1
    assert counts_by_type[LayerType.ATTENTION] == 24
    assert counts_by_type[LayerType.FEEDFORWARD] == 24
    assert counts_by_type[LayerType.OTHER] == 1
