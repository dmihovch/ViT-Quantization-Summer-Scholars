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
        ("blocks.0.attn.qkv", LayerType.ATTENTION_QKV),
        ("blocks.11.attn.proj", LayerType.ATTENTION_PROJ),
        ("blocks.0.mlp.fc1", LayerType.FEEDFORWARD_FC1),
        ("blocks.5.mlp.fc2", LayerType.FEEDFORWARD_FC2),
        ("head", LayerType.OTHER),
    ],
)
def test_infer_layer_type(layer_name: str, expected_type: LayerType) -> None:
    """Each kind of layer name maps to the expected LayerType tag."""
    assert classify_linear_layer(layer_name) is expected_type


def test_layer_type_values_are_human_readable() -> None:
    """The enum values are the labels we export to JSON and show on charts."""
    assert LayerType.ATTENTION_QKV.value == "Attention_QKV"
    assert LayerType.ATTENTION_PROJ.value == "Attention_proj"
    assert LayerType.FEEDFORWARD_FC1.value == "FeedForward_fc1"
    assert LayerType.FEEDFORWARD_FC2.value == "FeedForward_fc2"
    assert LayerType.OTHER.value == "Other"
