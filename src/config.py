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
        Pass ``None`` to use the entire split.
    batch_size:
        Mini-batch size for the DataLoader.
    device:
        Compute device (CPU or CUDA) used for the forward passes.
    seed:
        Base random seed for reproducibility.  When ``num_seeds > 1``,
        seeds ``seed``, ``seed+1``, ..., ``seed+num_seeds-1`` are used.
    num_seeds:
        Number of independent runs with different seeds.  Results are
        saved to ``output_dir/seed_{s}/`` for each seed ``s``.  Default 1
        produces a single run written directly to ``output_dir/``.
    """

    data_dir: Path
    output_dir: Path
    num_images: int | None
    batch_size: int
    device: torch.device
    seed: int = 42
    num_seeds: int = 1
    skip_outlier_recount: bool = False
    """If True, skip the second-pass global-σ outlier recount.
    Use only for fast iteration; results will have approximate outlier fractions.
    """


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
        Path to the ``profiling_result.json`` file produced by Phase 1
        (``profiler.save_profiling_result``).  Provides exact global σ
        for all six measurement sites across every encoder block.
    attn_profile_num_images:
        **Deprecated.**  Phase 1 now produces dataset-wide ``pre_softmax`` σ
        via ``run_profiling_dataset_pass``.  This field is ignored.
    attn_profile_seed:
        **Deprecated.**  See ``attn_profile_num_images``.
    """

    data_dir: Path
    output_dir: Path
    num_images: int
    batch_size: int
    device: torch.device
    sigma_thresholds: tuple[float, ...]
    layer_stats_path: Path
    seed: int = 42
    """Global random seed for reproducibility.  Used for the random-zeroing
    control mask generation."""
    granularity: str = "global"
    """Zeroing granularity: ``"global"`` uses per-layer μ and σ;
    ``"per_channel"`` uses per-channel μ_c and σ_c for pre_gelu only.
    Per-channel mode skips pre_softmax and residual_stream sites."""
    attn_profile_num_images: int = 64
    attn_profile_seed: int = 42


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
