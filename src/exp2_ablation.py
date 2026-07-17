"""Phase 2 — Outlier Ablation (Zeroing) experiment entry point.

Orchestrates the full Phase 2 pipeline:
  1. Load the pretrained ViT model and preprocessing transform.
  2. Load per-layer statistics produced by Phase 1.
  3. For each sigma threshold ``k`` in ``config.sigma_thresholds``:
       a. Register outlier-zeroing pre-hooks on all GELU layers.
       b. Build the validation DataLoader.
       c. Evaluate top-1 and top-5 accuracy.
       d. Collect per-layer percentage-zeroed statistics.
       e. Remove hooks before the next threshold iteration.
  4. Save all :class:`~ablation.AblationResult` records to a CSV file.
  5. Generate accuracy-vs-threshold and per-layer % zeroed plots.
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
