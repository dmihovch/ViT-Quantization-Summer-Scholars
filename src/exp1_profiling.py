"""Phase 1 — Baseline Activation Profiling experiment entry point.

Orchestrates the full Phase 1 pipeline using the ``profiler.py`` Welford
multi-batch pipeline (Option C) for exact dataset-wide statistics across all
six measurement sites per encoder block.

Pipeline
--------
1. Load the pretrained ViT-B/16 model and preprocessing transform.
2. Build the ImageNet validation DataLoader (auto-shuffled for subsets).
3. Run ``run_profiling_dataset_pass`` inside ``torch.no_grad()`` to collect
   exact global statistics (mean, std, kurtosis, outlier fractions,
   per-channel std) via the Pébay (2008) parallel higher-moments merge.
4. Save the complete ``ProfilingResult`` to ``profiling_result.json``.
5. Run ``histogram_profile_vit`` on a shuffled batch to collect full
   activation tensors for blocks 0, 5, 11 at all six sites.
6. Generate and save log-scale activation histograms (real activations) and
   per-channel std heatmaps via ``plotting.*``.

When ``config.num_seeds > 1``, steps 2–6 are repeated for each seed
``config.seed``, ``config.seed+1``, ..., ``config.seed+num_seeds-1``.
Results are written to ``output_dir/seed_{s}/`` for each seed.

Statistical notes
-----------------
All statistics are exact (population conventions, ddof=0).  Kurtosis is
computed via the Pébay (2008) M3/M4 parallel merge — no approximation.
Histograms are generated from real activation tensors collected by
``histogram_profile_vit`` — not reconstructed Gaussians.
"""

from __future__ import annotations

import logging
from typing import Callable

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
    histogram_profile_vit,
    run_profiling_dataset_pass,
    save_profiling_result,
)
from src.utils import ensure_dir, seed_everything

logger = logging.getLogger(__name__)


def run(config: ProfilingConfig) -> None:
    """Execute the Phase 1 profiling pipeline end-to-end.

    All artefacts (``profiling_result.json``, histogram PNGs, and per-channel
    σ heatmap) are written under ``config.output_dir``, which is created if it
    does not exist.

    When ``config.num_seeds > 1``, the pipeline is repeated for each seed
    and results are saved to per-seed subdirectories.

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
    # 1. Load model and wrap with NNsight (shared across all seeds).
    model, transform = load_vit(config.device)
    wrapped = NNsight(model)

    seeds = [config.seed + s for s in range(config.num_seeds)]

    for run_idx, run_seed in enumerate(seeds):
        if config.num_seeds > 1:
            logger.info(
                "=== Seed %d/%d (seed=%d) ===",
                run_idx + 1, config.num_seeds, run_seed,
            )
            run_output_dir = config.output_dir / f"seed_{run_seed}"
        else:
            run_output_dir = config.output_dir

        seed_everything(run_seed)
        _run_single(wrapped, transform, config, run_output_dir)

    logger.info("Phase 1 complete. Outputs in %s", config.output_dir)


def _run_single(
    wrapped: NNsight,
    transform: Callable,
    config: ProfilingConfig,
    output_dir: Path,
) -> None:
    """Run the profiling pipeline for a single seed.

    Parameters
    ----------
    wrapped:
        NNsight-wrapped model (already loaded and on the correct device).
    transform:
        Preprocessing transform from ``load_vit``.
    config:
        Full profiling configuration.
    output_dir:
        Directory for this seed's outputs (created if needed).
    """
    # 2. Build the validation DataLoader.
    #    Auto-shuffle: True for subsets (class diversity), False for full dataset.
    loader = build_val_loader(
        config.data_dir,
        transform,
        config.batch_size,
        config.num_images,
        config.device,
    )

    # 3. Dataset-wide profiling pass (all 6 sites, exact statistics).
    logger.info("Starting profiling pass over %s images...",
                config.num_images if config.num_images is not None else "all")
    with torch.no_grad():
        stats: dict[SiteId, LayerStats] = run_profiling_dataset_pass(
            wrapped, loader, config.device,
        )

    # 4. Save profiling result.
    ensure_dir(output_dir)

    # Peek at first batch shape for accurate metadata (not hardcoded).
    first_images, _ = next(iter(loader))
    actual_batch_shape = tuple(first_images.shape)

    inner = wrapped._model
    result = ProfilingResult(
        stats=stats,
        num_blocks=len(inner.blocks),
        batch_shape=actual_batch_shape,
    )
    json_path = output_dir / "profiling_result.json"
    save_profiling_result(result, json_path)
    logger.info("Stats for %d sites saved to %s", len(stats), json_path)

    # 5. Generate plots.
    _plot_histograms(wrapped, transform, config, output_dir)
    _plot_per_channel_heatmap(stats, output_dir)
    logger.info("Seed complete. Outputs in %s", output_dir)


def _plot_histograms(
    wrapped: NNsight,
    transform: Callable,
    config: ProfilingConfig,
    output_dir: Path,
    block_indices: tuple[int, ...] = (0, 5, 11),
) -> None:
    """Generate real-activation histograms for selected blocks.

    Runs one additional forward pass using ``histogram_profile_vit`` to
    collect full activation tensors at all six sites for ``block_indices``.
    Histograms show the true distribution including heavy tails.

    Args:
        wrapped: NNsight-wrapped model (already profiled by the Welford pass).
        transform: The preprocessing transform returned by ``load_vit``.
            Passed explicitly so a shuffled loader can be constructed here.
        config: Profiling config (data_dir, device, batch_size, num_images).
        output_dir: Directory where histogram PNGs are written.
        block_indices: Encoder blocks to generate histograms for.
    """
    # Shuffled loader samples from the FULL dataset (not the Welford subset)
    # so the histogram batch spans all classes.  Pass num_images=None to use
    # the entire dataset, then the DataLoader's shuffle+batching gives us a
    # random batch_size sample.
    histogram_loader = build_val_loader(
        config.data_dir, transform, config.batch_size,
        None, config.device, shuffle=True,
    )
    images, _ = next(iter(histogram_loader))
    with torch.no_grad():
        raw_tensors = histogram_profile_vit(
            wrapped, images.to(config.device), block_indices,
        )
    hist_dir = output_dir / "histograms"
    ensure_dir(hist_dir)
    for key, tensor in raw_tensors.items():
        activations = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        safe_key = key.replace("/", "_").replace(".", "_")
        plot_activation_histogram(
            activations=activations,
            layer_name=key,
            output_path=hist_dir / f"{safe_key}.png",
            log_scale=True,
        )
    logger.info("Wrote %d real-activation histogram PNGs to %s", len(raw_tensors), hist_dir)


def _plot_per_channel_heatmap(
    stats: dict[SiteId, LayerStats], output_dir: Path,
) -> None:
    """Generate per-channel σ heatmap for sites that track channel statistics.

    Only includes sites where ``per_channel_std`` is not None (pre_gelu and
    post_layernorm_1/2).  Sites with different channel dimensions (e.g.
    pre_gelu at 3072 vs layernorm at 768) are plotted in separate heatmaps.

    Args:
        stats: Mapping from site_identifier to finalized global LayerStats.
        output_dir: Directory where heatmap PNGs are written.
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

    # Group sites by channel dimension so each heatmap has homogeneous columns.
    by_dim: dict[int, dict[str, list[float]]] = {}
    for key, stds in per_channel.items():
        d = len(stds)
        by_dim.setdefault(d, {})[key] = stds

    for d, group in sorted(by_dim.items()):
        suffix = f"_d{d}" if len(by_dim) > 1 else ""
        out_path = output_dir / f"per_channel_std_heatmap{suffix}.png"
        plot_per_channel_std_heatmap(group, out_path)
        logger.info("Per-channel σ heatmap (D=%d) written to %s", d, out_path)