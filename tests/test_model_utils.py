"""
test_model_utils.py
===================

Unit tests for layer classification. These are pure functions with no model or
GPU involved, so they run in milliseconds.
"""

import pytest

from src.model_utils import LayerType, classify_linear_layer


@pytest.mark.parametrize(
    ("layer_name", "expected_type"),
    [
        # Anything under self_attention is tagged ATTENTION.
        ("encoder.layers.encoder_layer_0.self_attention.out_proj", LayerType.ATTENTION),
        ("encoder.layers.encoder_layer_11.self_attention", LayerType.ATTENTION),
        # The two MLP linears in each block are FEEDFORWARD.
        ("encoder.layers.encoder_layer_0.mlp.0", LayerType.FEEDFORWARD),
        ("encoder.layers.encoder_layer_5.mlp.3", LayerType.FEEDFORWARD),
        # The final classifier head is neither attention nor MLP.
        ("heads.head", LayerType.OTHER),
    ],
)
def test_classify_linear_layer(layer_name: str, expected_type: LayerType) -> None:
    """Each kind of layer name maps to the expected LayerType tag."""
    assert classify_linear_layer(layer_name) is expected_type


def test_layer_type_values_are_human_readable() -> None:
    """The enum values are the labels we export to JSON and show on charts."""
    assert LayerType.ATTENTION.value == "Attention_QKV"
    assert LayerType.FEEDFORWARD.value == "FeedForward_MLP"
    assert LayerType.OTHER.value == "Other_Linear"
