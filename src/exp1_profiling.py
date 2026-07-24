"""Phase 1 — Baseline Activation Profiling experiment entry point.

Orchestrates the full Phase 1 pipeline using the ``profiler.py`` Welford
multi-batch pipeline (Option C) for exact dataset-wide statistics across all
five measurement sites per encoder block.

Pipeline
--------
1. Load the pretrained ViT-B/16 model and preprocessing transform.
2. Build the ImageNet validation DataLoader.
3. Run ``run_profiling_dataset_pass`` inside ``torch.no_grad()`` to collect
   exact global statistics (mean, std, kurtosis, outlier fractions,
   per-channel std) via the Pébay (2008) parallel higher-moments merge.
4. Save the complete ``ProfilingResult`` to ``profiling_result.json``.
5. Generate and save log-scale activation histograms (reconstructed from
   N(μ, σ²)) and per-channel std heatmaps via ``plotting.*``.

Statistical notes
-----------------
All statistics are exact (population conventions, ddof=0).  Kurtosis is
computed via the Pébay (2008) M3/M4 parallel merge — no approximation.
Histograms are drawn from synthetic N(μ, σ²) samples (labelled as such)
because the Welford pipeline discards raw tensors for memory efficiency.
Real-activation histograms require the ``--spot-batch`` path (not yet
implemented — see ``open-issues.md`` Issue 6.2).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from nnsight import NNsight

from src.config import ProfilingConfig
from src.data_loader import build_val_loader
from src.model import load_vit
from src.plotting import plot_activation_histogram, plot_per_channel_std_heatmap
from src.profiler import (
    LayerStats,
    ProfilingResult,
    SiteId,
    run_profiling_dataset_pass,
    save_profiling_result,
)
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def run(config: ProfilingConfig) -> None:
    """Execute the Phase 1 profiling pipeline end-to-end.

    All artefacts (``profiling_result.json``, histogram PNGs, and per-channel
    σ heatmap) are written under ``config.output_dir``, which is created if it
    does not exist.

    Parameters
    ----------
    config:
        Fully-specified :class:`~config.ProfilingConfig` instance.

    Raises
    ------
    ProfilingError
        If the nnsight trace fails.
    RuntimeError
        If the DataLoader yields zero batches.
    """
    # 1. Load model and wrap with NNsight.
    model, transform = load_vit(config.device)
    wrapped = NNsight(model)

    # 2. Build the validation DataLoader.
    loader = build_val_loader(
        config.data_dir,
        transform,
        config.batch_size,
        config.num_images,
        config.device,
    )

    # 3. Dataset-wide profiling pass (all 5 sites, exact statistics).
    logger.info("Starting profiling pass over %d images...", config.num_images)
    with torch.no_grad():
        stats: dict[SiteId, LayerStats] = run_profiling_dataset_pass(
            wrapped, loader, config.device,
        )

    # 4. Save profiling result.
    ensure_dir(config.output_dir)

    # Peek at first batch shape for accurate metadata (not hardcoded).
    first_images, _ = next(iter(loader))
    actual_batch_shape = tuple(first_images.shape)

    inner = wrapped._model
    result = ProfilingResult(
        stats=stats,
        num_blocks=len(inner.blocks),
        batch_shape=actual_batch_shape,
    )
    json_path = config.output_dir / "profiling_result.json"
    save_profiling_result(result, json_path)
    logger.info("Stats for %d sites saved to %s", len(stats), json_path)

    # 5. Generate plots.
    _plot_histograms(stats, config)
    _plot_per_channel_heatmap(stats, config)
    logger.info("Phase 1 complete. Outputs in %s", config.output_dir)


def _plot_histograms(
    stats: dict[SiteId, LayerStats], config: ProfilingConfig,
) -> None:
    """Generate synthetic N(μ, σ²) histograms for every site.

    Histograms are drawn from synthetic Gaussian samples because the Welford
    pipeline discards raw activation tensors.  Every title contains
    ``[reconstructed N(μ,σ²)]`` as a mandatory label.

    Args:
        stats: Mapping from site_identifier to finalized global LayerStats.
        config: Profiling configuration (for output_dir).
    """
    hist_dir = config.output_dir / "histograms"
    ensure_dir(hist_dir)
    rng = np.random.default_rng(seed=0)

    for key, s in stats.items():
        synthetic = rng.normal(
            loc=s.mean, scale=max(s.std, 1e-8), size=50_000,
        ).astype(np.float32)
        safe_key = key.replace("/", "_").replace(".", "_")
        plot_activation_histogram(
            activations=synthetic,
            layer_name=f"{key}  [reconstructed N(μ,σ²)]",
            output_path=hist_dir / f"{safe_key}.png",
            log_scale=True,
        )

    logger.info("Wrote %d histogram PNGs to %s", len(stats), hist_dir)


def _plot_per_channel_heatmap(
    stats: dict[SiteId, LayerStats], config: ProfilingConfig,
) -> None:
    """Generate per-channel σ heatmap for sites that track channel statistics.

    Only includes sites where ``per_channel_std`` is not None (pre_gelu and
    post_layernorm_1/2).

    Args:
        stats: Mapping from site_identifier to finalized global LayerStats.
        config: Profiling configuration (for output_dir).
    """
    per_channel: dict[str, list[float]] = {
        key: s.per_channel_std
        for key, s in stats.items()
        if s.per_channel_std is not None
    }
    if not per_channel:
        logger.warning(
            "No per_channel_std data found in stats. "
            "Ensure _register_stat_saves is called with track_per_channel=True "
            "for pre_gelu and post_layernorm sites (Step 4b-iii)."
        )
        return
    out_path = config.output_dir / "per_channel_std_heatmap.png"
    plot_per_channel_std_heatmap(per_channel, out_path)
    logger.info("Per-channel σ heatmap written to %s", out_path)