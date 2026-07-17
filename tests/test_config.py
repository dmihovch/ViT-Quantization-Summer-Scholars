"""Tests verifying that all config dataclasses are immutable (frozen)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from src.config import AblationConfig, IntegerGELUConfig, ProfilingConfig


def test_profiling_config_is_frozen(tmp_path: Path) -> None:
    """Assigning to any field of ProfilingConfig must raise FrozenInstanceError."""
    config = ProfilingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        num_images=256,
        batch_size=32,
        device=torch.device("cpu"),
    )
    with pytest.raises(FrozenInstanceError):
        config.num_images = 512  # type: ignore[misc]


def test_ablation_config_is_frozen(tmp_path: Path) -> None:
    """Assigning to any field of AblationConfig must raise FrozenInstanceError."""
    config = AblationConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        num_images=256,
        batch_size=32,
        device=torch.device("cpu"),
        sigma_thresholds=(2.0, 3.0),
        layer_stats_path=tmp_path / "layer_stats.json",
    )
    with pytest.raises(FrozenInstanceError):
        config.sigma_thresholds = (1.0,)  # type: ignore[misc]


def test_integer_gelu_config_is_frozen(tmp_path: Path) -> None:
    """Assigning to any field of IntegerGELUConfig must raise FrozenInstanceError."""
    config = IntegerGELUConfig(
        output_dir=tmp_path,
        device=torch.device("cpu"),
        layer_stats_path=tmp_path / "layer_stats.json",
        ablation_stats_path=tmp_path / "ablation_results.csv",
    )
    with pytest.raises(FrozenInstanceError):
        config.output_dir = tmp_path / "other"  # type: ignore[misc]
