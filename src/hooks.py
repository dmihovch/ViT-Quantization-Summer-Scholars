"""Forward hook machinery for collecting pre-GELU activation statistics.

Hooks reduce each activation tensor to scalar summary statistics immediately
upon capture — *no* raw tensors are retained in memory — which keeps the
profiling run viable at scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from src.exceptions import HookRegistrationError

logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    """Per-layer summary statistics of pre-GELU activations.

    Attributes
    ----------
    layer_name:
        Fully-qualified module name as returned by ``model.named_modules()``.
    maximum:
        Maximum observed activation value across all profiled batches.
    minimum:
        Minimum observed activation value across all profiled batches.
    std:
        Standard deviation of activations across all profiled batches.
    mean:
        Mean of activations across all profiled batches.
    """

    layer_name: str
    maximum: float
    minimum: float
    std: float
    mean: float


@dataclass
class HookHandle:
    """Container returned by ``register_profiling_hooks``.

    Attributes
    ----------
    handles:
        List of ``RemovableHandle`` objects, one per registered hook.
        Pass this to ``remove_hooks`` for cleanup.
    stats:
        Live-updated mapping from layer name to :class:`LayerStats`.
        Values are populated (and may be updated) as batches flow through
        the model; read only after all forward passes are complete.
    """

    handles: list[RemovableHandle]
    stats: dict[str, LayerStats]


def register_profiling_hooks(model: nn.Module) -> HookHandle:
    """Attach forward hooks to every ``nn.GELU`` submodule in ``model``.

    Each hook captures the *input* tensor to the GELU activation (i.e. the
    pre-GELU values), immediately reduces it to ``max``, ``min``, ``std``,
    and ``mean`` scalars, and stores them in the returned ``HookHandle.stats``
    dict.  Multiple batches accumulate a running update so the final stats
    reflect the full profiling dataset.

    Parameters
    ----------
    model:
        The model to profile.  Must contain at least one ``nn.GELU`` child.

    Returns
    -------
    HookHandle
        Contains all hook handles (for later removal) and a live reference
        to the accumulated stats dict.

    Raises
    ------
    HookRegistrationError
        If ``model`` contains no ``nn.GELU`` submodules.
    """
    raise NotImplementedError


def remove_hooks(handle: HookHandle) -> None:
    """Remove all registered hooks referenced by ``handle``.

    Should be called after profiling is complete to avoid memory leaks and
    unintended side-effects on subsequent model evaluations.

    Parameters
    ----------
    handle:
        The :class:`HookHandle` returned by ``register_profiling_hooks``.
    """
    raise NotImplementedError


def save_stats(stats: dict[str, LayerStats], path: Path) -> None:
    """Serialize a stats mapping to a JSON file.

    The JSON structure is a dict keyed by layer name, with each value being
    a dict of the :class:`LayerStats` fields.  Parent directories are created
    if they do not exist.

    Parameters
    ----------
    stats:
        Mapping produced by ``register_profiling_hooks`` after a full forward
        pass over the profiling dataset.
    path:
        Destination file path (e.g. ``output_dir / "layer_stats.json"``).
    """
    raise NotImplementedError


def load_stats(path: Path) -> dict[str, LayerStats]:
    """Deserialize a stats mapping previously saved by ``save_stats``.

    Parameters
    ----------
    path:
        Path to the JSON file written by ``save_stats``.

    Returns
    -------
    dict[str, LayerStats]
        Mapping from layer name to :class:`LayerStats` instances.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist on disk.
    """
    raise NotImplementedError
