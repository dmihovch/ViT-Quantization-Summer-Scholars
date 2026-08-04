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

import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
from nnsight import NNsight
from PIL import Image

from src.config import ProfilingConfig
from src.data_loader import build_val_loader
from src.model import load_vit
from src.plotting import (
    plot_activation_histogram,
)
from src.profiler import (
    LayerStats,
    ProfilingResult,
    RunMetadata,
    SiteId,
    generate_summary_table,
    histogram_profile_vit,
    run_outlier_counting_pass,
    run_profiling_dataset_pass,
    save_profiling_result,
    save_summary_table,
)
from src.utils import collect_system_metadata, ensure_dir, seed_everything

logger = logging.getLogger(__name__)


def run(config: ProfilingConfig) -> None:
    """Execute the Phase 1 profiling pipeline end-to-end.

    Saves ``profiling_result.json``, ``summary_table.csv``, and activation
    histograms to ``config.output_dir / seed_{seed}``.  All other plots are
    generated offline via ``scripts/regenerate_plots.py``.

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

        seed_everything(run_seed)
        _run_single(wrapped, transform, config, run_output_dir)

    logger.info("Phase 1 complete. Outputs in %s", config.output_dir)


def _run_single(
    wrapped: NNsight,
    transform: Callable[[Image.Image], torch.Tensor],
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
    #    Auto-shuffle: always True (class-diverse batches for representative
    #    per-batch σ, reducing the outlier-fraction overestimate).
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

    # 3b. Global-σ outlier recount (F2).
    if not config.approximate_outliers:
        logger.info("Starting outlier recount pass (global-σ fractions)...")
        with torch.no_grad():
            corrected_fractions = run_outlier_counting_pass(
                wrapped, loader, config.device, stats
            )
        # Patch stats in-place: replace per-batch outlier fractions with global-σ ones.
        for site_id, fracs in corrected_fractions.items():
            stats[site_id] = dataclasses.replace(
                stats[site_id], outlier_fractions=fracs
            )
        logger.info("Outlier recount complete.")
    else:
        logger.warning(
            "Approximate outliers mode (--approximate-outliers). "
            "Outlier fractions in output are per-batch-σ approximations — "
            "systematically over-estimated relative to the correct global-σ definition."
        )

    # 4. Save profiling result.
    ensure_dir(output_dir)

    # Peek at first batch shape for accurate metadata (not hardcoded).
    first_images, _ = next(iter(loader))
    actual_batch_shape = tuple(first_images.shape)

    inner = wrapped._model

    # Collect system metadata for reproducibility.
    sys_meta = collect_system_metadata()
    metadata = RunMetadata(
        python_version=str(sys_meta["python_version"]),
        pytorch_version=str(sys_meta["pytorch_version"]),
        timm_version=str(sys_meta["timm_version"]),
        nnsight_version=str(sys_meta["nnsight_version"]),
        cuda_available=bool(sys_meta["cuda_available"]),
        cuda_version=str(sys_meta["cuda_version"]) if sys_meta["cuda_version"] is not None else None,
        gpu_name=str(sys_meta["gpu_name"]) if sys_meta["gpu_name"] is not None else None,
        gpu_memory_gb=float(sys_meta["gpu_memory_gb"]) if sys_meta["gpu_memory_gb"] is not None else None,
        model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        dataset="ImageNet-1K validation",
        num_images=config.num_images,
        batch_size=config.batch_size,
        seed=config.seed,
        num_seeds=config.num_seeds,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    result = ProfilingResult(
        stats=stats,
        num_blocks=len(inner.blocks),  # type: ignore[arg-type]
        batch_shape=actual_batch_shape,
        metadata=metadata,
    )
    json_path = output_dir / "profiling_result.json"
    save_profiling_result(result, json_path)
    logger.info("Stats for %d sites saved to %s", len(stats), json_path)

    # 4b. Generate summary table (F4).
    table_rows = generate_summary_table(result)
    table_path = output_dir / "summary_table.csv"
    save_summary_table(table_rows, table_path)
    logger.info("Summary table (%d rows) written to %s", len(table_rows), table_path)

    # 5. Generate activation histograms (only plot that requires live model).
    _plot_histograms(wrapped, transform, config, output_dir)

    logger.info(
        "Seed complete.  Data saved to %s.  Run regenerate_plots.py for all other plots.",
        output_dir,
    )


def _plot_histograms(
    wrapped: NNsight,
    transform: Callable[[Image.Image], torch.Tensor],
    config: ProfilingConfig,
    output_dir: Path,
    block_indices: tuple[int, ...] = tuple(range(12)),
) -> None:
    """Generate real-activation histograms for all 12 encoder blocks.

    Runs one additional forward pass using ``histogram_profile_vit`` to
    collect full activation tensors at all six sites for every block.
    Histograms show the true distribution including heavy tails.

    Defaults to all 12 blocks (0-11).  Wei et al. (2022, arXiv:2209.13325,
    §3.1) show that outlier emergence is progressive through the network —
    histograms for every block are needed to characterize this fully.

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