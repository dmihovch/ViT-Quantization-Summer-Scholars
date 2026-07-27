"""Custom exception hierarchy for the ViT quantization project.

All project-specific errors subclass standard built-ins so callers can
catch either the concrete type or its parent class interchangeably.
"""

from __future__ import annotations


class DataDirectoryError(FileNotFoundError):
    """Raised when a required image directory is missing or contains no images.

    Examples
    --------
    Raised by ``data_loader.build_val_loader`` when ``data_dir`` does not
    exist on disk or yields zero samples after constructing the dataset.
    """


class HookRegistrationError(RuntimeError):
    """Raised when no ``nn.GELU`` modules are found to attach hooks to.

    Examples
    --------
    Raised by ``hooks.register_profiling_hooks`` when iterating over a model
    that contains no ``torch.nn.GELU`` submodules, which would silently
    produce an empty stats dict without this guard.
    """


class ShapeMismatchError(ValueError):
    """Raised when tensor shapes are incompatible for a given operation.

    Examples
    --------
    Raised by any function that requires two tensors to have matching shapes
    (e.g. element-wise comparisons during ablation or LUT application).
    """


class ProfilingError(RuntimeError):
    """Raised when the nnsight profiling trace fails or produces unexpected results.

    Examples
    --------
    Raised by ``profiler.profile_vit`` when the nnsight trace raises an
    exception, when the model has no ``blocks`` attribute, or when zero
    transformer blocks are found.
    """
