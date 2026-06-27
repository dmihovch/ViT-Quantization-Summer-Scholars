"""
test_integration.py
====================

End-to-end test against the REAL ViT-B/16 model. This is the test that proves
hooks fire on every layer (including attention, whose fused kernel bypasses the
internal out_proj linear).

It is marked `slow` because the first run downloads ~330 MB of pretrained
weights. Skip it during quick iteration with:

    pytest -m "not slow"
"""

import pytest
import torch

from src.hooks import OutlierStatsCollector, make_outlier_hook
from src.model_utils import LayerType, iter_measured_modules, load_vit_b_16


@pytest.mark.slow
def test_every_measured_layer_receives_data() -> None:
    # Run on CPU so the test does not require a GPU.
    device = torch.device("cpu")
    model, _transform = load_vit_b_16(device)

    collector = OutlierStatsCollector()
    handles = []
    for layer_name, module, layer_type in iter_measured_modules(model):
        collector.register_layer(layer_name, layer_type)
        handles.append(
            module.register_forward_hook(make_outlier_hook(collector, layer_name))
        )

    # One small synthetic batch shaped like preprocessed input: [B, 3, 224, 224].
    with torch.no_grad():
        _ = model(torch.randn(2, 3, 224, 224, device=device))
    for handle in handles:
        handle.remove()

    summaries = collector.build_summaries()

    # 12 attention blocks + 24 MLP linears + 1 classifier head = 37 modules.
    assert len(summaries) == 37
    assert all(summary.total_values_seen > 0 for summary in summaries)

    counts_by_type: dict[LayerType, int] = {layer_type: 0 for layer_type in LayerType}
    for summary in summaries:
        counts_by_type[summary.layer_type] += 1
    assert counts_by_type[LayerType.ATTENTION] == 12
    assert counts_by_type[LayerType.FEEDFORWARD] == 24
    assert counts_by_type[LayerType.OTHER] == 1
