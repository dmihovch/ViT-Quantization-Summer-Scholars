"""Outlier-zeroing ablation for Phase 2.

For each sigma threshold ``k``, every pre-GELU activation element whose
absolute value exceeds ``k * σ`` is hard-zeroed.  The experiment sweeps
multiple ``k`` values and records the resulting top-1/top-5 accuracy and
percentage of zeroed activations per layer.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from src.hooks import LayerStats

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result record for a single (sigma threshold, layer) combination.

    Attributes
    ----------
    sigma_threshold:
        The multiplier ``k`` used to compute the absolute threshold
        ``k * stats.std``.
    layer_name:
        Fully-qualified GELU module name this result refers to.
    pct_zeroed:
        Fraction of activation elements zeroed by the mask, expressed as a
        percentage in the range ``[0, 100]``.
    top1_accuracy:
        Top-1 classification accuracy (%) measured after applying zeroing
        hooks across *all* GELU layers at this threshold.  Filled in after
        full-dataset evaluation; initialised to ``0.0`` at construction.
    top5_accuracy:
        Top-5 classification accuracy (%) under the same conditions.
    """

    sigma_threshold: float
    layer_name: str
    pct_zeroed: float
    top1_accuracy: float
    top5_accuracy: float


def build_zeroing_hook(
    layer_name: str,
    threshold: float,
    stats: LayerStats,
) -> Callable:
    """Create a forward pre-hook that zeros outlier activations.

    The returned hook zeros every element of the input tensor where
    ``|x| > threshold * stats.std``.  It does **not** modify the tensor
    in-place; instead it returns a new tensor so the original gradient graph
    is unaffected.

    Parameters
    ----------
    layer_name:
        Name of the layer this hook will be attached to (used for logging).
    threshold:
        Multiplier applied to ``stats.std`` to compute the absolute cutoff.
    stats:
        Per-layer statistics from Phase 1 (must contain a valid ``std``).

    Returns
    -------
    Callable
        A forward pre-hook compatible with ``nn.Module.register_forward_pre_hook``.
        Signature: ``hook(module, args) -> tuple``.
    """
    raise NotImplementedError


def patch_model_for_ablation(
    model: nn.Module,
    sigma_k: float,
    layer_stats: dict[str, LayerStats],
) -> list[RemovableHandle]:
    """Register outlier-zeroing pre-hooks on every GELU layer in ``model``.

    For each ``nn.GELU`` submodule whose name appears in ``layer_stats``,
    a zeroing pre-hook is registered using the per-layer ``std`` from Phase 1
    and the provided threshold multiplier ``sigma_k``.

    Parameters
    ----------
    model:
        The ViT model to patch.  Must contain ``nn.GELU`` submodules whose
        names match keys in ``layer_stats``.
    sigma_k:
        Threshold multiplier (e.g. ``2.0`` means zero elements  > 2σ).
    layer_stats:
        Per-layer statistics loaded from the Phase 1 JSON.

    Returns
    -------
    list[RemovableHandle]
        Handles for every registered hook.  Call ``.remove()`` on each (or
        pass to ``hooks.remove_hooks``) when ablation is complete.
    """
    raise NotImplementedError


def compute_pct_zeroed(tensor: torch.Tensor, threshold: float) -> float:
    """Compute the percentage of tensor elements whose absolute value exceeds ``threshold``.

    This is a pure function with no side effects.

    Parameters
    ----------
    tensor:
        Input activation tensor (any shape).
    threshold:
        Absolute value cutoff.

    Returns
    -------
    float
        Percentage in the range ``[0, 100]`` of elements satisfying
        ``|x| > threshold``.
    """
    raise NotImplementedError


def save_ablation_results(results: list[AblationResult], path: Path) -> None:
    """Persist ablation results to a CSV file.

    The CSV includes a header row matching the fields of :class:`AblationResult`.
    Parent directories are created if they do not exist.

    Parameters
    ----------
    results:
        List of :class:`AblationResult` instances, one per (threshold, layer)
        combination.
    path:
        Destination file path (e.g. ``output_dir / "ablation_results.csv"``).
    """
    raise NotImplementedError
