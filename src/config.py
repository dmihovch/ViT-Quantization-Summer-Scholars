"""Frozen dataclass configurations for all three experimental phases.

All paths are ``pathlib.Path`` objects so callers never manipulate raw
strings.  The dataclasses are frozen to prevent accidental mutation during
a run; construct a new instance if a parameter needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ProfilingConfig:
    """Configuration for Phase 1 — Baseline Pre-GELU Profiling.

    Attributes
    ----------
    data_dir:
        Root of the ImageNet validation split (ImageFolder layout).
    output_dir:
        Directory where stats JSON and histogram PNGs are written.
    num_images:
        Number of validation images to pass through the model.  Using a
        subset keeps wall-clock time manageable during development.
    batch_size:
        Mini-batch size for the DataLoader.
    device:
        Compute device (CPU or CUDA) used for the forward passes.
    """

    data_dir: Path
    output_dir: Path
    num_images: int
    batch_size: int
    device: torch.device


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for Phase 2 — Outlier Ablation (Zeroing).

    Attributes
    ----------
    data_dir:
        Root of the ImageNet validation split.
    output_dir:
        Directory where results CSV and accuracy/zeroed plots are written.
    num_images:
        Number of validation images used for each accuracy evaluation.
    batch_size:
        Mini-batch size for the DataLoader.
    device:
        Compute device used for forward passes.
    sigma_thresholds:
        Tuple of multipliers ``k`` such that the zeroing threshold is
        ``k * σ`` for each layer.  A tuple is used to guarantee immutability
        inside the frozen dataclass.
    layer_stats_path:
        Path to the JSON file produced by Phase 1 (``hooks.save_stats``).
    """

    data_dir: Path
    output_dir: Path
    num_images: int
    batch_size: int
    device: torch.device
    sigma_thresholds: tuple[float, ...]
    layer_stats_path: Path


@dataclass(frozen=True)
class IntegerGELUConfig:
    """Configuration for Phase 3 — Integer GELU Exploration.

    Attributes
    ----------
    output_dir:
        Directory where LUT comparison plots and metrics JSON are written.
    device:
        Compute device (used for any tensor operations during comparison).
    layer_stats_path:
        Path to the JSON file produced by Phase 1.  Per-layer ``std`` values
        are used to derive quantisation scales.
    ablation_stats_path:
        Path to the CSV produced by Phase 2.  Accuracy figures are included
        in the final Phase 3 report for cross-phase comparison.
    """

    output_dir: Path
    device: torch.device
    layer_stats_path: Path
    ablation_stats_path: Path
