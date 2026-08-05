"""Frozen dataclass configurations for both experimental phases.

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
    approximate_outliers:
        If True, skip the second-pass global-σ outlier recount and use
        approximate per-batch-σ outlier fractions instead.  Use only for fast
        iteration; the output will have systematically over-estimated outlier
        fractions relative to the correct global-σ definition.
    """

    data_dir: Path
    output_dir: Path
    num_images: int | None
    batch_size: int
    device: torch.device
    seed: int = 42
    num_seeds: int = 1
    approximate_outliers: bool = False


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
    granularity:
        Zeroing granularity: ``"global"`` uses per-layer μ and σ;
        ``"per_channel"`` uses per-channel μ_c and σ_c for pre_gelu only.
        Per-channel mode skips pre_softmax and residual_stream sites.
    ablation_mode:
        Ablation variant for per-channel mode:

        - ``"outlier"``: per-channel μ_c and σ_c (full per-channel).
        - ``"mean_only"``: per-channel μ_c but global σ.  Isolates the
          mean-correction component of the per-channel benefit.
        - ``"var_only"``: global μ but per-channel σ_c.  Isolates the
          variance-correction component.
    layer_range:
        If not ``None``, only intervene on blocks in this inclusive range
        ``(start, end)``, 0-based.  Blocks outside the range pass through
        unchanged.  Enables per-block-group ablation.
    seed:
        Base random seed for reproducibility.  When ``num_seeds > 1``,
        seeds ``seed``, ``seed+1``, ..., ``seed+num_seeds-1`` are used.
        Used for the random-zeroing control mask generation.
    num_seeds:
        Number of independent runs with different seeds.  Results are
        saved to ``output_dir/seed_{s}/`` for each seed ``s``.  Default 1
        produces a single run written directly to ``output_dir/``.
    """

    data_dir: Path
    output_dir: Path
    num_images: int
    batch_size: int
    device: torch.device
    sigma_thresholds: tuple[float, ...]
    layer_stats_path: Path
    seed: int = 42
    """Base random seed for reproducibility.  When ``num_seeds > 1``,
    seeds ``seed``, ``seed+1``, ..., ``seed+num_seeds-1`` are used.
    Used for the random-zeroing control mask generation."""
    num_seeds: int = 1
    """Number of independent runs with different seeds.  Results are
    saved to ``output_dir/seed_{s}/`` for each seed ``s``.  Default 1
    produces a single run written directly to ``output_dir/``."""
    granularity: str = "global"
    """Zeroing granularity: ``"global"`` uses per-layer μ and σ;
    ``"per_channel"`` uses per-channel μ_c and σ_c for pre_gelu only.
    Per-channel mode skips pre_softmax and residual_stream sites."""
    ablation_mode: str = "outlier"
    """Per-channel ablation variant: ``"outlier"``, ``"mean_only"``, or
    ``"var_only"``.  Ignored in global granularity mode."""
    layer_range: tuple[int, int] | None = None
    """If not None, only ablate blocks in this inclusive range (0-based)."""
