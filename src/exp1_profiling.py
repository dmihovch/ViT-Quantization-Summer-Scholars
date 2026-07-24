"""Phase 1 — Baseline Activation Profiling experiment entry point.

Orchestrates the full Phase 1 pipeline using the ``hooks.py`` Welford
accumulator pipeline for dataset-wide statistics.  ``profiler.py`` (nnsight)
is reserved for single-batch spot-checks and attention-site previews; it is
not used here because finalized per-batch scalars cannot be correctly merged
across batches.

Pipeline
--------
1. Load the pretrained ViT-B/16 model and preprocessing transform.
2. Build the ImageNet validation DataLoader.
3. Register Welford profiling hooks (``hooks.register_profiling_hooks``) on
   all ``nn.GELU`` and ``nn.LayerNorm`` submodules.  Covers three sites:
   ``residual_stream``, ``post_layernorm_1/2``, and ``pre_gelu``.
4. Run all images through the model inside ``torch.no_grad()``.
5. Remove hooks (``hooks.remove_hooks``).
6. Finalize accumulators and save ``LayerStats`` dict to
   ``layer_stats.json`` via ``hooks.save_stats``.
7. Generate and save log-scale activation histograms and per-channel
   std heatmaps via ``plotting.*``.

Statistical notes
-----------------
Mean, std, and outlier fractions are computed exactly via Welford's
parallel-groups merge formula.  Kurtosis is an approximation: each batch
contributes ``(x − batch_mean)⁴`` using the local batch mean as a proxy for
the global mean (Chan et al., 1983).  This approximation is adequate for
layer ranking but should be labelled approximate in any publication.
"""

from __future__ import annotations

import logging

from src.config import ProfilingConfig

logger = logging.getLogger(__name__)


def run(config: ProfilingConfig) -> None:
    """Execute the Phase 1 profiling pipeline end-to-end.

    All artefacts (``layer_stats.json`` and per-layer histogram PNGs) are
    written under ``config.output_dir``, which is created if it does not exist.

    Parameters
    ----------
    config:
        Fully-specified :class:`~config.ProfilingConfig` instance.

    Raises
    ------
    NotImplementedError
        Always; stub implementation pending Phase 1 development.
    """
    raise NotImplementedError
