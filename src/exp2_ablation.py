"""Phase 2 — Outlier Ablation (Zeroing) experiment entry point.

Orchestrates the full Phase 2 pipeline across three measurement sites:
``pre_gelu``, ``pre_softmax``, and ``residual_stream``.

All intervention is performed inside nnsight trace contexts via
:func:`ablation.zero_outliers_in_trace` — no raw PyTorch hooks are used.

Pipeline
--------
1. Load the pretrained ViT-B/16 model, wrap with NNsight.
2. Load Phase 1 per-layer statistics from ``config.layer_stats_path``
   (``profiling_result.json``).  These provide exact global ``σ`` and ``μ``
   for all six measurement sites — no single-batch estimation needed.
3. Build the validation DataLoader (shuffle=False for deterministic eval).
4. Measure **baseline** accuracy (no intervention) via
   ``model.evaluate_accuracy``.
5. For each site in ``{pre_gelu, pre_softmax, residual_stream}``:
     For each ``k`` in ``config.sigma_thresholds``:
       For each batch in the val loader:
         a. **Outlier pass**: zero_outliers_in_trace with mean-centered
            threshold.  Collect logits and per-layer %-zeroed.
         b. **Random control** (pre_gelu and residual_stream only):
            zero_outliers_in_trace with random_fractions set to the
            exact per-batch per-layer %-zeroed from step (a).  This
            ensures the random control zeros exactly the same fraction
            of elements as the outlier condition on every batch.
       Record :class:`~ablation.AblationResult` per layer for both
       outlier and random conditions.
6. Save all results to ``ablation_results.csv`` and ``entropy_deltas.csv`` via
   ``ablation.save_ablation_results`` and ``ablation.save_entropy_deltas``.
   Plots are generated offline via ``scripts/regenerate_plots.py``.

Statistical notes
-----------------
All zeroing thresholds use exact global σ and μ from Phase 1's
``run_profiling_dataset_pass``.  The threshold definition is mean-centered
(``|x − μ| > k·σ``), consistent with Phase 1's outlier definition and the
standard statistical convention (Wei et al. 2022, §3.1; Bondarenko et al.
2021, §4.1).  No batch-derived estimates are used.

The random-zeroing control zeros the **exact same fraction** of elements as
the outlier-threshold condition on a **per-batch** basis.  After each outlier
forward pass, the per-layer %-zeroed is collected and used as the target
fraction for the random pass on the same batch.  This eliminates the
confound of differing zeroing rates between conditions.

Entropy deltas are computed per-head using ``torch.special.entr``
(consistent with Phase 1 after T-017 fix) and compared against Phase 1
baseline entropy values from ``profiling_result.json``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch.nn as nn
from nnsight import NNsight
from PIL import Image

from src.ablation import (
    AblationResult,
    save_ablation_results,
    save_entropy_deltas,
    zero_outliers_in_trace,
)
from src.config import AblationConfig
from src.data_loader import build_val_loader
from src.model import evaluate_accuracy, load_vit
from src.profiler import LayerStats, load_profiling_result
from src.utils import ensure_dir, seed_everything

logger = logging.getLogger(__name__)


def _site_matches(site_id: str, site: str) -> bool:
    """Check whether a site_identifier belongs to the given measurement site.

    Parameters
    ----------
    site_id:
        Fully-qualified site identifier, e.g. ``"blocks.3/pre_gelu"``.
    site:
        Measurement site name: ``"pre_gelu"``, ``"pre_softmax"``, or
        ``"residual_stream"``.

    Returns
    -------
    bool
        True if ``site_id`` ends with ``"/{site}"`` or matches the
        ``patch_embed/residual_stream`` special case.
    """
    if site_id == "patch_embed/residual_stream" and site == "residual_stream":
        return True
    return site_id.endswith(f"/{site}")


def _build_layer_results(
    pct_sum: dict[str, float],
    pct_count: dict[str, int],
    entropy_cls_sum: dict[str, list[float]],
    entropy_patch_sum: dict[str, list[float]],
    site: str,
    sigma_k: float,
    layer_stats: dict[str, LayerStats],
    baseline_top1: float,
    baseline_top5: float,
    top1: float,
    top5: float,
    seed: int,
    is_random: bool,
    granularity: str = "global",
    ablation_mode: str = "outlier",
) -> list[AblationResult]:
    """Build AblationResult records from accumulated per-layer statistics.

    Parameters
    ----------
    pct_sum:
        Running sum of per-layer %-zeroed across batches.
    pct_count:
        Number of batches contributing to each layer's sum.
    entropy_cls_sum:
        Per-batch CLS entropy lists, keyed by site_identifier.
    entropy_patch_sum:
        Per-batch patch entropy lists, keyed by site_identifier.
    site:
        Measurement site name.
    sigma_k:
        Threshold multiplier.
    layer_stats:
        Phase 1 per-site statistics (for baseline entropy).
    baseline_top1, baseline_top5:
        Unablated accuracy percentages.
    top1, top5:
        Accuracy percentages for this condition.
    seed:
        Random seed used for this run (for multi-seed aggregation).
    is_random:
        Whether this is the random control condition.
    granularity:
        Zeroing granularity (``"global"`` or ``"per_channel"``).
    ablation_mode:
        Per-channel ablation variant (``"outlier"``, ``"mean_only"``, or
        ``"var_only"``).  Ignored when ``granularity == "global"``.

    Returns
    -------
    list[AblationResult]
    """
    results: list[AblationResult] = []
    for sid in sorted(pct_sum.keys()):
        mean_pct = pct_sum[sid] / pct_count[sid] if pct_count[sid] > 0 else 0.0

        cls_entropy: list[float] = []
        patch_entropy: list[float] = []
        baseline_cls: list[float] = []
        baseline_patch: list[float] = []

        if sid in entropy_cls_sum and sid in layer_stats:
            num_heads = len(entropy_cls_sum[sid][0])
            cls_entropy = [
                sum(batch[h] for batch in entropy_cls_sum[sid]) / len(entropy_cls_sum[sid])
                for h in range(num_heads)
            ]
            patch_entropy = [
                sum(batch[h] for batch in entropy_patch_sum[sid]) / len(entropy_patch_sum[sid])
                for h in range(num_heads)
            ]
            ps_stats = layer_stats[sid]
            # Phase 1 stores attention entropy on post_softmax sites, but
            # ablation site_identifiers use pre_softmax.  Resolve the
            # corresponding post_softmax site for baseline entropy lookup.
            entropy_sid = sid.replace("/pre_softmax", "/post_softmax")
            entropy_stats = layer_stats.get(entropy_sid)
            if entropy_stats is not None:
                if entropy_stats.attention_entropy_cls is not None:
                    baseline_cls = list(entropy_stats.attention_entropy_cls)
                if entropy_stats.attention_entropy_patches is not None:
                    baseline_patch = list(entropy_stats.attention_entropy_patches)

        results.append(AblationResult(
            site=site,
            sigma_threshold=sigma_k,
            site_identifier=sid,
            pct_zeroed=mean_pct,
            top1_accuracy=top1,
            top5_accuracy=top5,
            baseline_top1=baseline_top1,
            baseline_top5=baseline_top5,
            seed=seed,
            is_random=is_random,
            granularity=granularity,
            ablation_mode=ablation_mode,
            cls_entropy=cls_entropy,
            patch_entropy=patch_entropy,
            baseline_cls_entropy=baseline_cls,
            baseline_patch_entropy=baseline_patch,
        ))

    return results


def run(config: AblationConfig) -> None:
    """Execute the Phase 2 ablation pipeline end-to-end.

    When ``config.num_seeds > 1``, the pipeline is repeated for each seed
    ``config.seed``, ``config.seed+1``, ..., ``config.seed+num_seeds-1``.
    Results are written to ``config.output_dir / seed_{s}/`` for each seed.

    Saves ``ablation_results.csv`` and ``entropy_deltas.csv`` to
    ``config.output_dir / seed_{s}/``.  All plots are generated offline via
    ``scripts/regenerate_plots.py``.

    Parameters
    ----------
    config:
        Fully-specified :class:`~config.AblationConfig` instance.
    """
    # 1. Load model and wrap with NNsight (shared across all seeds).
    model, transform = load_vit(config.device)
    wrapped = NNsight(model)

    # 2. Load Phase 1 stats (shared across all seeds).
    profiling_result = load_profiling_result(config.layer_stats_path)
    layer_stats: dict[str, LayerStats] = profiling_result.stats
    logger.info(
        "Loaded Phase 1 stats: %d sites from %s",
        len(layer_stats), config.layer_stats_path,
    )

    seeds = [config.seed + s for s in range(config.num_seeds)]

    for run_idx, run_seed in enumerate(seeds):
        if config.num_seeds > 1:
            logger.info(
                "=== Seed %d/%d (seed=%d) ===",
                run_idx + 1, config.num_seeds, run_seed,
            )
        run_output_dir = config.output_dir / f"seed_{run_seed}"

        seed_everything(run_seed)
        _run_single(wrapped, model, transform, config, run_seed, run_output_dir, layer_stats)

    logger.info("Phase 2 complete. Outputs in %s", config.output_dir)


def _run_single(
    wrapped: NNsight,
    model: nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    config: AblationConfig,
    run_seed: int,
    output_dir: Path,
    layer_stats: dict[str, LayerStats],
) -> None:
    """Run the ablation pipeline for a single seed.

    Parameters
    ----------
    wrapped:
        NNsight-wrapped model (already loaded and on the correct device).
    model:
        Underlying VisionTransformer (for baseline accuracy eval).
    transform:
        Preprocessing transform from ``load_vit``.
    config:
        Full ablation configuration.
    run_seed:
        Seed value for this run.
    output_dir:
        Directory for this seed's outputs (created if needed).
    layer_stats:
        Phase 1 per-site statistics.
    """

    # 3. Build val loader (deterministic order for fair comparisons).
    loader = build_val_loader(
        config.data_dir, transform, config.batch_size,
        config.num_images, config.device, shuffle=False,
    )

    # 4. Measure baseline accuracy (no intervention).
    logger.info("Measuring baseline accuracy (seed=%d)...", run_seed)
    baseline_top1, baseline_top5 = evaluate_accuracy(model, loader, config.device)
    logger.info("Baseline (seed=%d): top-1=%.2f%%, top-5=%.2f%%", run_seed, baseline_top1, baseline_top5)

    # 5. Sweep sites × thresholds.
    #    For pre_gelu and residual_stream, each batch is processed twice:
    #    first with outlier thresholding, then with random zeroing using
    #    the exact per-batch per-layer %-zeroed from the outlier pass.
    #    In per_channel mode, only pre_gelu is ablated (per-channel μ_c, σ_c
    #    thresholds are only meaningful for the channel-structured MLP hidden dim).
    is_per_channel = config.granularity == "per_channel"
    if is_per_channel:
        sites: tuple[str, ...] = ("pre_gelu",)
        logger.info("Per-channel granularity: only ablating pre_gelu site.")
    else:
        sites = ("pre_gelu", "pre_softmax", "residual_stream")
    all_results: list[AblationResult] = []

    for site in sites:
        logger.info("=== Ablating site: %s ===", site)
        do_random = site in ("pre_gelu", "residual_stream") and not is_per_channel

        for k in config.sigma_thresholds:
            logger.info("  Threshold k=%.1f ...", k)

            # --- Accumulators for outlier pass ---
            out_pct_sum: dict[str, float] = defaultdict(float)
            out_pct_count: dict[str, int] = defaultdict(int)
            out_entropy_cls: dict[str, list[float]] = defaultdict(list)
            out_entropy_patch: dict[str, list[float]] = defaultdict(list)
            out_correct_top1 = 0
            out_correct_top5 = 0
            out_total = 0

            # --- Accumulators for random control ---
            rnd_pct_sum: dict[str, float] = defaultdict(float)
            rnd_pct_count: dict[str, int] = defaultdict(int)
            rnd_correct_top1 = 0
            rnd_correct_top5 = 0
            rnd_total = 0

            for images, labels in loader:
                images = images.to(config.device)
                labels = labels.to(config.device)

                # --- Outlier pass ---
                logits, batch_pct, batch_entropy = zero_outliers_in_trace(
                    wrapped, images, site, k, layer_stats,
                    per_channel=is_per_channel,
                    ablation_mode=config.ablation_mode,
                    layer_range=config.layer_range,
                )

                for sid, pct in batch_pct.items():
                    out_pct_sum[sid] += pct
                    out_pct_count[sid] += 1
                for sid, ent in batch_entropy.items():
                    out_entropy_cls[sid].append(ent["cls"])
                    out_entropy_patch[sid].append(ent["patch"])

                top5_preds = logits.topk(5, dim=1).indices
                out_correct_top1 += (top5_preds[:, 0] == labels).sum().item()
                out_correct_top5 += (
                    top5_preds == labels.unsqueeze(1)
                ).any(dim=1).sum().item()
                out_total += labels.size(0)

                # --- Random control (per-batch matched fractions) ---
                if do_random:
                    # Convert per-layer %-zeroed (0-100) to fractions (0-1).
                    random_fractions = {
                        sid: pct / 100.0 for sid, pct in batch_pct.items()
                    }
                    rnd_logits, rnd_pct, _rnd_entropy = zero_outliers_in_trace(
                        wrapped, images, site, k, layer_stats,
                        random_fractions=random_fractions,
                        random_seed=run_seed,
                    )

                    for sid, pct in rnd_pct.items():
                        rnd_pct_sum[sid] += pct
                        rnd_pct_count[sid] += 1

                    rnd_top5 = rnd_logits.topk(5, dim=1).indices
                    rnd_correct_top1 += (rnd_top5[:, 0] == labels).sum().item()
                    rnd_correct_top5 += (
                        rnd_top5 == labels.unsqueeze(1)
                    ).any(dim=1).sum().item()
                    rnd_total += labels.size(0)

            # --- Record outlier results ---
            if out_total == 0:
                logger.warning("No samples evaluated for site=%s k=%.1f", site, k)
                continue

            out_top1 = 100.0 * out_correct_top1 / out_total
            out_top5 = 100.0 * out_correct_top5 / out_total
            logger.info(
                "    site=%s k=%.1f [outlier]: top-1=%.2f%%, top-5=%.2f%%",
                site, k, out_top1, out_top5,
            )

            all_results.extend(_build_layer_results(
                out_pct_sum, out_pct_count,
                out_entropy_cls, out_entropy_patch,
                site, k, layer_stats,
                baseline_top1, baseline_top5,
                out_top1, out_top5,
                seed=run_seed,
                is_random=False,
                granularity=config.granularity,
                ablation_mode=config.ablation_mode,
            ))

            # --- Record random control results ---
            if do_random and rnd_total > 0:
                rnd_top1 = 100.0 * rnd_correct_top1 / rnd_total
                rnd_top5 = 100.0 * rnd_correct_top5 / rnd_total
                logger.info(
                    "    site=%s k=%.1f [random]: top-1=%.2f%%, top-5=%.2f%%",
                    site, k, rnd_top1, rnd_top5,
                )

                all_results.extend(_build_layer_results(
                    rnd_pct_sum, rnd_pct_count,
                    defaultdict(list), defaultdict(list),  # no entropy for random
                    site, k, layer_stats,
                    baseline_top1, baseline_top5,
                    rnd_top1, rnd_top5,
                    seed=run_seed,
                    is_random=True,
                    granularity=config.granularity,
                    ablation_mode=config.ablation_mode,
                ))

    # 6. Save results.
    ensure_dir(output_dir)
    csv_path = output_dir / "ablation_results.csv"
    save_ablation_results(all_results, csv_path)

    entropy_path = output_dir / "entropy_deltas.csv"
    save_entropy_deltas(all_results, entropy_path)

    logger.info(
        "Seed %d complete.  Data saved to %s.  "
        "Run 'python scripts/regenerate_plots.py --phase2-csv %s --output-dir %s' for plots.",
        run_seed, output_dir, csv_path, output_dir,
    )