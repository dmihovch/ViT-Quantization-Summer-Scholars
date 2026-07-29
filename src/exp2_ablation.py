"""Phase 2 — Outlier Ablation (Zeroing) experiment entry point.

Orchestrates the full Phase 2 pipeline across three measurement sites:
``pre_gelu``, ``pre_softmax``, and ``residual_stream``.

Pipeline
--------
1. Load the pretrained ViT-B/16 model and preprocessing transform.
2. Load Phase 1 per-layer statistics from ``config.layer_stats_path``
   (``profiling_result.json`` produced by ``profiler.save_profiling_result``).
   These provide exact global ``σ`` for all six measurement sites.
3. **Sample attention-site σ:** build a single batch of
   ``config.attn_profile_num_images`` images using fixed seed
   ``config.attn_profile_seed``, run ``profiler.profile_vit`` on that
   batch, and extract the per-layer ``pre_softmax`` std.  This batch-derived
   estimate is used as the scale parameter for the pre-softmax threshold
   ``τ = k · σ``.  It is **not** a globally exact statistic — log a
   warning to that effect at runtime.
4. Merge Phase 1 and sampled stats into a unified ``layer_stats`` dict.
5. For each site in ``{pre_gelu, pre_softmax, residual_stream}``:
     For each ``k`` in ``config.sigma_thresholds``:
       a. Register site-selective zeroing pre-hooks
          (``ablation.patch_model_for_ablation``).
       b. Build the validation DataLoader.
       c. Evaluate top-1 and top-5 accuracy (``model.evaluate_accuracy``).
       d. Record per-layer percentage-zeroed and accuracy as
          :class:`~ablation.AblationResult`.
       e. Remove hooks (``ablation.remove_hooks``) before next iteration.
     For ``pre_softmax`` only: also record
     ``ablation.compute_entropy_delta`` across the threshold sweep.
6. Save all :class:`~ablation.AblationResult` records to
   ``ablation_results.csv`` via ``ablation.save_ablation_results``.
7. Generate accuracy-vs-threshold and per-layer %-zeroed plots via
   ``plotting.*``.

Statistical notes
-----------------
Zeroing thresholds for ``pre_gelu`` and ``residual_stream`` use exact global
σ from Phase 1.  The ``pre_softmax`` threshold uses a batch-derived σ
estimate (see step 3); this is adequate for a threshold scale parameter but
must be disclosed as approximate in any publication.
"""

from __future__ import annotations

import logging

from src.config import AblationConfig

logger = logging.getLogger(__name__)


def run(config: AblationConfig) -> None:
    """Execute the Phase 2 ablation pipeline end-to-end.

    All artefacts (``ablation_results.csv`` and plot PNGs) are written under
    ``config.output_dir``, which is created if it does not exist.

    Parameters
    ----------
    config:
        Fully-specified :class:`~config.AblationConfig` instance.

    Raises
    ------
    NotImplementedError
        Always; stub implementation pending Phase 2 development.
    """
    raise NotImplementedError
