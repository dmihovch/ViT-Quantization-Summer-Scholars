"""Tests for the model loading and evaluation utilities in :mod:`src.model`."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import evaluate_accuracy


class _MockViT(torch.nn.Module):
    """A mock model that returns logits based on the first pixel of the input.

    The first pixel value encodes the correct class index.  The model returns
    a one-hot logit vector at that index.  This way the mock works correctly
    regardless of DataLoader batching or shuffling.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # First pixel of first channel encodes the correct label.
        labels = x[:, 0, 0, 0].long()
        logits = torch.zeros(B, self.num_classes, device=x.device)
        logits[torch.arange(B), labels] = 1.0
        return logits


def test_evaluate_accuracy_perfect_top1() -> None:
    """evaluate_accuracy must return 100% top-1 when the model is always correct."""
    num_samples = 10
    num_classes = 10
    labels = torch.arange(num_samples) % num_classes
    # Encode the label in the first pixel.
    images = torch.zeros(num_samples, 3, 224, 224)
    images[:, 0, 0, 0] = labels.float()

    model = _MockViT(num_classes)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    device = torch.device("cpu")

    top1, top5 = evaluate_accuracy(model, loader, device)
    assert top1 == 100.0, f"Expected 100% top-1, got {top1}%"
    assert top5 == 100.0, f"Expected 100% top-5, got {top5}%"


def test_evaluate_accuracy_known_result() -> None:
    """evaluate_accuracy must compute correct top-1 and top-5 for known logits.

    Uses 10 classes so topk(5) is valid.

    Sample 0: correct class = 0, logits = [1.0, 0.9, 0.8, 0.7, 0.6, ...] → rank 1
    Sample 1: correct class = 1, logits = [0.9, 1.0, 0.8, 0.7, 0.6, ...] → rank 1
    Sample 2: correct class = 5, logits = [0.5, 0.4, 0.3, 0.2, 0.1, 1.0, ...] → rank 1

    All should be top-1 correct.
    """
    num_samples = 3
    num_classes = 10
    labels = torch.tensor([0, 1, 5])
    images = torch.zeros(num_samples, 3, 224, 224)
    images[:, 0, 0, 0] = labels.float()

    model = _MockViT(num_classes)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    device = torch.device("cpu")

    top1, top5 = evaluate_accuracy(model, loader, device)
    assert top1 == 100.0, f"Expected 100% top-1, got {top1}%"
    assert top5 == 100.0, f"Expected 100% top-5, got {top5}%"


def test_evaluate_accuracy_empty_loader_raises() -> None:
    """evaluate_accuracy must raise RuntimeError when the loader yields no samples."""
    model = _MockViT(100)
    loader = DataLoader(
        TensorDataset(torch.randn(0, 3, 224, 224), torch.zeros(0, dtype=torch.long)),
        batch_size=1,
    )
    with pytest.raises(RuntimeError, match="zero samples"):
        evaluate_accuracy(model, loader, torch.device("cpu"))