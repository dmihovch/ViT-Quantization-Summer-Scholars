"""
test_model_utils.py
===================

Smoke tests for model loading utilities. These do not download the model,
so they only verify module-level imports and the shape contract for the
accuracy evaluation helper using a tiny dummy model.
"""

from __future__ import annotations

import torch
from src.model import evaluate_top1_top5_accuracy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TinyClassifier(nn.Module):
    """A trivial 3-class linear classifier for unit testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(x)


def test_evaluate_top1_top5_accuracy_perfect_model() -> None:
    """A model that always predicts class 0 gets 100% on a dataset of all 0s."""
    # Build a model that always outputs class 0 as the highest logit.
    model = _TinyClassifier()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.fill_(0.0)
        model.fc.bias[0] = 10.0  # class 0 always wins

    inputs = torch.randn(8, 4)
    labels = torch.zeros(8, dtype=torch.long)  # all class 0
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)

    device = torch.device("cpu")
    top1, top5 = evaluate_top1_top5_accuracy(model, loader, device)

    assert top1 == 100.0
    assert top5 == 100.0


def test_evaluate_top1_top5_accuracy_returns_floats() -> None:
    """Return types are both plain Python floats."""
    model = _TinyClassifier()
    inputs = torch.randn(4, 4)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)

    device = torch.device("cpu")
    top1, top5 = evaluate_top1_top5_accuracy(model, loader, device)

    assert isinstance(top1, float)
    assert isinstance(top5, float)
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top5 <= 100.0
    assert top5 >= top1  # Top-5 is always at least as good as Top-1
