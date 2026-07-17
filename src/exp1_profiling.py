"""Phase 1 — Baseline Pre-GELU Profiling experiment entry point.

Orchestrates the full Phase 1 pipeline:
  1. Load the pretrained ViT model and preprocessing transform.
  2. Build the ImageNet validation DataLoader.
  3. Register profiling hooks on every ``nn.GELU`` submodule.
  4. Run all images through the model in evaluation mode.
  5. Remove hooks and save collected statistics to JSON.
  6. Generate and save log-scale activation histograms for each layer.
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
