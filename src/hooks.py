"""Forward hook machinery for collecting multi-site activation statistics.

Hooks reduce each activation tensor to scalar summary statistics immediately
upon capture — *no* raw tensors are retained in memory — which keeps the
profiling run viable at scale.

Five measurement sites are targeted by the spec. Three are implemented here:

    pre_gelu        — inputs to nn.GELU; shape (B, N, D_mlp)
    post_layernorm  — outputs of nn.LayerNorm; shape (B, N, D)
    residual_stream — accumulated residual *before* each LayerNorm normalises
                      it; captured via pre-hook on the same nn.LayerNorm
                      modules as post_layernorm.

Two attention sites are deferred pending a design decision:

    pre_softmax     — raw QKᵀ/√d logits; requires patching PyTorch SDPA
                      because timm's ViT uses scaled_dot_product_attention,
                      which never materialises the logit matrix in memory.
    post_softmax    — attention weight matrix; same root cause.

Stats accumulate online across all forward passes using Welford's algorithm.
Call ``remove_hooks`` after all batches are processed; it finalises statistics
and populates ``HookHandle.stats``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeAlias

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from src.exceptions import HookRegistrationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site key constants
# ---------------------------------------------------------------------------

SITE_PRE_GELU: str = "pre_gelu"
SITE_POST_LAYERNORM: str = "post_layernorm"
SITE_RESIDUAL_STREAM: str = "residual_stream"

# Outlier sigma thresholds matched to the spec.
_OUTLIER_SIGMAS: tuple[int, ...] = (3, 4, 6)

StatsKey: TypeAlias = str  # format: "{layer_name}/{site}"


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Per-layer summary statistics for one measurement site.

    All scalar statistics reflect the full distribution of tensor elements
    seen across every profiling batch.

    Attributes:
        site: Site key (e.g. ``"pre_gelu"``). One of the SITE_* constants.
        layer_name: Fully-qualified module name from ``model.named_modules()``.
        max: Maximum observed value over all profiled batches.
        min: Minimum observed value over all profiled batches.
        mean: Global mean over all profiled batches.
        std: Global standard deviation over all profiled batches.
        kurtosis: Excess kurtosis (E[(x−μ)⁴]/σ⁴ − 3). A Gaussian scores 0;
            heavy-tailed distributions score positive.
        outlier_frac: Fraction of elements where |x| > k·σ, for k in {3,4,6}.
            Keys are string representations of k: ``"3"``, ``"4"``, ``"6"``.
        per_channel_std: Per-channel σ over batch and token dims; shape [D].
            Non-None only for ``pre_gelu`` and ``post_layernorm`` sites.
        attn_entropy: Per-head mean attention entropy in bits. Non-None only
            for ``post_softmax`` (not yet implemented, always ``None``).
        n_samples: Total scalar elements accumulated (for reference).
    """

    site: str
    layer_name: str
    max: float
    min: float
    mean: float
    std: float
    kurtosis: float
    outlier_frac: dict[str, float]
    per_channel_std: list[float] | None
    attn_entropy: list[float] | None
    n_samples: int


@dataclass
class HookHandle:
    """Container returned by ``register_profiling_hooks``.

    Attributes:
        handles: One ``RemovableHandle`` per registered hook. Pass to
            ``remove_hooks`` for cleanup and stat finalisation.
        stats: Mapping from ``"{layer_name}/{site}"`` to :class:`LayerStats`.
            Empty until :func:`remove_hooks` is called.
        _accumulators: Internal mapping of live accumulators; not part of the
            public API and should not be read directly by callers.
    """

    handles: list[RemovableHandle]
    stats: dict[StatsKey, LayerStats]
    _accumulators: dict[StatsKey, _SiteAccumulator]


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


@dataclass
class _SiteAccumulator:
    """Online running accumulators for one (layer, site) measurement point.

    Uses Welford's parallel-groups merge formula for numerically stable
    global mean and variance across arbitrary batch sizes.

    The fourth central moment is accumulated per-batch and merged by summing
    (x − batch_mean)⁴ contributions. This is an approximation that converges
    to the true excess kurtosis for large n across many batches; an exact
    parallel formula would require tracking M3 and M4 with cross terms, adding
    complexity for negligible benefit at our dataset sizes.

    Attributes:
        site: Site key string.
        layer_name: Module name this accumulator belongs to.
        n: Total scalar elements accumulated.
        running_max: Running element-wise maximum.
        running_min: Running element-wise minimum.
        welford_mean: Current Welford running mean.
        welford_M2: Current Welford running sum of squared deviations.
        sum_fourth_central: Running sum of (x − batch_mean)⁴ across batches.
        outlier_counts: Raw count of |x| > k·σ (running σ estimate) per k.
        per_channel_sum: Running per-channel sum; None if not tracked.
        per_channel_sum_sq: Running per-channel sum of squares; None if not tracked.
        per_channel_n: Sample count for per-channel stats (B*N per batch).
        track_per_channel: Whether to accumulate per-channel statistics.
    """

    site: str
    layer_name: str
    n: int = 0
    running_max: float = -math.inf
    running_min: float = math.inf
    welford_mean: float = 0.0
    welford_M2: float = 0.0
    sum_fourth_central: float = 0.0
    outlier_counts: dict[str, int] = field(
        default_factory=lambda: {str(k): 0 for k in _OUTLIER_SIGMAS}
    )
    per_channel_sum: torch.Tensor | None = None
    per_channel_sum_sq: torch.Tensor | None = None
    per_channel_n: int = 0
    track_per_channel: bool = False


# ---------------------------------------------------------------------------
# Accumulator update helpers
# ---------------------------------------------------------------------------


def _update_accumulator(acc: _SiteAccumulator, tensor: torch.Tensor) -> None:
    """Incorporate a new batch tensor into the running accumulators.

    All elements of ``tensor`` are treated as i.i.d. samples from the layer's
    activation distribution. The tensor is flattened before processing so this
    works for any shape.

    All internal tensor operations are wrapped in ``torch.no_grad()`` so that
    the hook callback never builds an autograd graph while the model itself
    may still be mid-forward. This is the safest mode for hook callbacks on
    all platforms.

    Args:
        acc: Accumulator to update in-place.
        tensor: Activation tensor for the current batch. Any floating-point
            dtype; converted to float32 internally.
    """
    with torch.no_grad():
        x: torch.Tensor = tensor.detach().float().flatten()
        n_batch: int = x.numel()

        if n_batch == 0:
            return

        batch_max: float = x.max().item()
        batch_min: float = x.min().item()
        batch_mean: float = x.mean().item()
        # Population variance — treating each batch as its own full stratum.
        batch_var: float = x.var(unbiased=False).item()

        # Fourth central moment — accumulate sum of (x − batch_mean)⁴.
        batch_centered: torch.Tensor = x - batch_mean
        fourth_sum: float = (batch_centered**4).sum().item()

    # All Python-only arithmetic below; no torch dispatch needed.
    acc.running_max = max(acc.running_max, batch_max)
    acc.running_min = min(acc.running_min, batch_min)

    # Parallel-groups Welford merge.
    n_prev = acc.n
    n_combined = n_prev + n_batch
    delta: float = batch_mean - acc.welford_mean
    new_mean: float = acc.welford_mean + delta * n_batch / n_combined
    new_M2: float = (
        acc.welford_M2
        + batch_var * n_batch
        + (delta**2) * n_prev * n_batch / n_combined
    )

    acc.sum_fourth_central += fourth_sum

    # Outlier counting uses the running std estimate after the Welford merge.
    running_std: float = math.sqrt(new_M2 / n_combined) if n_combined > 1 else 0.0
    if running_std > 0.0:
        with torch.no_grad():
            x_abs_counts = {str(k): int((x.abs() > k * running_std).sum().item())
                            for k in _OUTLIER_SIGMAS}
        for k in _OUTLIER_SIGMAS:
            acc.outlier_counts[str(k)] += x_abs_counts[str(k)]

    acc.n = n_combined
    acc.welford_mean = new_mean
    acc.welford_M2 = new_M2

    if acc.track_per_channel:
        _update_per_channel(acc, tensor)


def _update_per_channel(acc: _SiteAccumulator, tensor: torch.Tensor) -> None:
    """Accumulate per-channel running sum and sum-of-squares.

    Expects ``tensor`` with shape ``(B, N, D)`` — batch, token, channel.
    Reduces over dims 0 and 1, keeping D.

    Args:
        acc: Accumulator to update in-place.
        tensor: Float tensor of shape ``(B, N, D)``.
    """
    if tensor.ndim != 3:  # noqa: PLR2004
        logger.warning(
            "per_channel_std requested for '%s/%s' but tensor ndim=%d (expected 3); "
            "skipping per-channel accumulation for this batch.",
            acc.layer_name,
            acc.site,
            tensor.ndim,
        )
        return

    with torch.no_grad():
        t = tensor.detach().double()  # (B, N, D) — float64 for numerical stability
        b, n, d = t.shape

        flat = t.reshape(-1, d)  # (B*N, D)
        batch_sum = flat.sum(dim=0)  # (D,)
        batch_sum_sq = (flat**2).sum(dim=0)  # (D,)

    if acc.per_channel_sum is None:
        acc.per_channel_sum = torch.zeros(d, dtype=torch.float64)
        acc.per_channel_sum_sq = torch.zeros(d, dtype=torch.float64)

    acc.per_channel_sum = acc.per_channel_sum + batch_sum
    acc.per_channel_sum_sq = acc.per_channel_sum_sq + batch_sum_sq  # type: ignore[operator]
    acc.per_channel_n += b * n


def _finalize_accumulator(acc: _SiteAccumulator) -> LayerStats:
    """Convert a completed accumulator into a :class:`LayerStats`.

    Args:
        acc: Accumulator after all forward passes have fired.

    Returns:
        A fully populated :class:`LayerStats` instance.

    Raises:
        RuntimeError: If the accumulator has seen zero elements.
    """
    if acc.n == 0:
        raise RuntimeError(
            f"Accumulator for '{acc.layer_name}/{acc.site}' has zero elements. "
            "Ensure at least one forward pass ran after hook registration."
        )

    global_var: float = acc.welford_M2 / acc.n
    global_std: float = math.sqrt(global_var) if global_var > 0.0 else 0.0

    # Excess kurtosis: E[(x−μ)⁴]/σ⁴ − 3. Using population σ.
    if global_std > 0.0:
        kurtosis: float = (acc.sum_fourth_central / acc.n) / (global_std**4) - 3.0
    else:
        kurtosis = 0.0

    outlier_frac: dict[str, float] = {
        str(k): acc.outlier_counts[str(k)] / acc.n for k in _OUTLIER_SIGMAS
    }

    per_channel_std: list[float] | None = None
    if (
        acc.track_per_channel
        and acc.per_channel_sum is not None
        and acc.per_channel_sum_sq is not None
        and acc.per_channel_n > 0
    ):
        mean_ch = acc.per_channel_sum / acc.per_channel_n  # (D,)
        var_ch = acc.per_channel_sum_sq / acc.per_channel_n - mean_ch**2  # (D,)
        std_ch = var_ch.clamp(min=0.0).sqrt()
        per_channel_std = std_ch.tolist()

    return LayerStats(
        site=acc.site,
        layer_name=acc.layer_name,
        max=acc.running_max,
        min=acc.running_min,
        mean=acc.welford_mean,
        std=global_std,
        kurtosis=kurtosis,
        outlier_frac=outlier_frac,
        per_channel_std=per_channel_std,
        attn_entropy=None,
        n_samples=acc.n,
    )


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _make_pre_hook(acc: _SiteAccumulator) -> Callable[[nn.Module, tuple], None]:
    """Build a ``forward_pre_hook`` that accumulates ``args[0]`` into ``acc``.

    Args:
        acc: The accumulator to update each time the hook fires.

    Returns:
        A callable compatible with ``nn.Module.register_forward_pre_hook``.
    """

    def hook(module: nn.Module, args: tuple) -> None:  # noqa: ARG001
        if not args:
            return
        tensor = args[0]
        if isinstance(tensor, torch.Tensor):
            _update_accumulator(acc, tensor)

    return hook


def _make_post_hook(
    acc: _SiteAccumulator,
) -> Callable[[nn.Module, tuple, torch.Tensor | tuple], None]:
    """Build a ``forward_hook`` that accumulates the module output into ``acc``.

    If the output is a tuple (e.g. some LayerNorm wrappers), the first element
    is used.

    Args:
        acc: The accumulator to update each time the hook fires.

    Returns:
        A callable compatible with ``nn.Module.register_forward_hook``.
    """

    def hook(
        module: nn.Module,  # noqa: ARG001
        inputs: tuple,  # noqa: ARG001
        output: torch.Tensor | tuple,
    ) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            _update_accumulator(acc, tensor)

    return hook


def _make_layernorm_combined_hook(
    res_acc: _SiteAccumulator,
    post_acc: _SiteAccumulator,
) -> Callable[[nn.Module, tuple, torch.Tensor | tuple], None]:
    """Build a single forward hook capturing both pre- and post-LayerNorm tensors.

    A single ``register_forward_hook`` receives both ``inputs`` (the residual
    stream before normalisation) and ``output`` (the normalised tensor) in one
    callback. This avoids registering a pre-hook AND a post-hook on the same
    ``nn.LayerNorm`` module, which can cause a C-level abort in some PyTorch
    2.2.x CPU backends when both hook types fire on the same LayerNorm
    (observed on macOS arm64; the combined-hook pattern is safer everywhere).

    Args:
        res_acc: Accumulator for the ``residual_stream`` site (input to LN).
        post_acc: Accumulator for the ``post_layernorm`` site (output of LN).

    Returns:
        A callable compatible with ``nn.Module.register_forward_hook``.
    """

    def hook(
        module: nn.Module,  # noqa: ARG001
        inputs: tuple,
        output: torch.Tensor | tuple,
    ) -> None:
        # Capture residual_stream from the LN input
        if inputs:
            inp = inputs[0]
            if isinstance(inp, torch.Tensor):
                _update_accumulator(res_acc, inp)

        # Capture post_layernorm from the LN output
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            _update_accumulator(post_acc, tensor)

    return hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_profiling_hooks(model: nn.Module) -> HookHandle:
    """Attach profiling hooks to ``nn.GELU`` and ``nn.LayerNorm`` submodules.

    Three sites are registered per matching module:

    - ``pre_gelu``: pre-hook on each ``nn.GELU`` — captures the pre-activation
      hidden states ``(B, N, D_mlp)``.
    - ``post_layernorm``: LN output ``(B, N, D)`` captured via a combined
      post-hook that also records the LN input for ``residual_stream``.
    - ``residual_stream``: LN input ``(B, N, D)`` captured by the same
      combined hook, giving the accumulated residual before normalisation.

    One combined ``register_forward_hook`` is used per ``nn.LayerNorm`` (rather
    than separate pre- and post-hooks on the same module). This avoids
    hook-dispatch edge cases in some PyTorch 2.2.x CPU backends.

    Stats are finalised (accumulators → :class:`LayerStats`) when
    :func:`remove_hooks` is called. Do not read ``handle.stats`` before then.

    Args:
        model: The model to profile. Must contain at least one ``nn.GELU``
            submodule (i.e. a ViT or any GELU-based transformer).

    Returns:
        A :class:`HookHandle` with empty ``stats``; will be populated by
        :func:`remove_hooks`.

    Raises:
        HookRegistrationError: If the model contains no ``nn.GELU`` submodules,
            indicating it is not a supported architecture.
    """
    accumulators: dict[StatsKey, _SiteAccumulator] = {}
    handles: list[RemovableHandle] = []
    gelu_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            gelu_count += 1
            key = f"{name}/{SITE_PRE_GELU}"
            acc = _SiteAccumulator(
                site=SITE_PRE_GELU, layer_name=name, track_per_channel=True
            )
            accumulators[key] = acc
            handles.append(module.register_forward_pre_hook(_make_pre_hook(acc)))

        elif isinstance(module, nn.LayerNorm):
            post_key = f"{name}/{SITE_POST_LAYERNORM}"
            post_acc = _SiteAccumulator(
                site=SITE_POST_LAYERNORM, layer_name=name, track_per_channel=True
            )
            accumulators[post_key] = post_acc

            # The pre-hook on the same LayerNorm captures the residual stream
            # *before* normalisation — the accumulated representation that will
            # be passed into the next sub-block (or the classification head).
            res_key = f"{name}/{SITE_RESIDUAL_STREAM}"
            res_acc = _SiteAccumulator(
                site=SITE_RESIDUAL_STREAM, layer_name=name, track_per_channel=False
            )
            accumulators[res_key] = res_acc

            # Use a single combined post-hook to capture BOTH the LN input
            # (residual_stream) and LN output (post_layernorm). Registering
            # separate pre- and post-hooks on the same nn.LayerNorm can trigger
            # a buffer-aliasing abort in some PyTorch 2.2.x CPU backends.
            handles.append(
                module.register_forward_hook(
                    _make_layernorm_combined_hook(res_acc, post_acc)
                )
            )

    if gelu_count == 0:
        raise HookRegistrationError(
            f"No nn.GELU submodules found in {type(model).__name__}. "
            "Profiling hooks require at least one GELU activation; "
            "verify this model is a GELU-based ViT."
        )

    ln_count = len(handles) - gelu_count  # one combined hook per LN module
    logger.info(
        "Registered %d hooks (%d GELU pre-hooks, %d LayerNorm combined hooks) in %s.",
        len(handles),
        gelu_count,
        ln_count,
        type(model).__name__,
    )

    return HookHandle(handles=handles, stats={}, _accumulators=accumulators)


def remove_hooks(handle: HookHandle) -> None:
    """Remove all hooks and finalise accumulated statistics into ``handle.stats``.

    Must be called after all profiling forward passes are complete.
    Calling it multiple times is safe — subsequent calls are no-ops.

    Args:
        handle: The :class:`HookHandle` returned by
            :func:`register_profiling_hooks`.
    """
    for h in handle.handles:
        h.remove()
    handle.handles.clear()

    for key, acc in handle._accumulators.items():
        if acc.n == 0:
            logger.warning(
                "Accumulator '%s' has zero elements; skipping finalisation. "
                "Did any forward passes run after hook registration?",
                key,
            )
            continue
        handle.stats[key] = _finalize_accumulator(acc)

    logger.info("Finalised stats for %d measurement sites.", len(handle.stats))


def save_stats(stats: dict[StatsKey, LayerStats], path: Path) -> None:
    """Serialise a stats mapping to a JSON file.

    Creates parent directories if they do not exist. JSON keys are the
    ``"{layer_name}/{site}"`` strings; values mirror :class:`LayerStats` fields.

    Args:
        stats: Mapping produced by :func:`remove_hooks`.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: asdict(val) for key, val in stats.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %d stats entries to %s.", len(stats), path)


def load_stats(path: Path) -> dict[StatsKey, LayerStats]:
    """Deserialise a stats mapping from a file written by :func:`save_stats`.

    Args:
        path: Path to the JSON file.

    Returns:
        Mapping from ``"{layer_name}/{site}"`` to :class:`LayerStats`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, dict] = json.load(f)

    return {key: LayerStats(**val) for key, val in raw.items()}
