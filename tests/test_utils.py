"""Tests for general-purpose utilities in :mod:`src.utils`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.utils import ensure_dir, get_device, seed_everything


def test_seed_everything_runs_without_error() -> None:
    """seed_everything should complete without raising and return None."""
    result = seed_everything(42)
    assert result is None


def test_get_device_returns_torch_device() -> None:
    """get_device should return a torch.device instance."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    """ensure_dir should create nested directories that do not yet exist."""
    target = tmp_path / "new" / "nested"
    assert not target.exists()
    ensure_dir(target)
    assert target.exists()
    assert target.is_dir()
