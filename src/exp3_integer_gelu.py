"""Phase 3 — Integer GELU Exploration experiment entry point.

Orchestrates the full Phase 3 pipeline:
  1. Load per-layer statistics from Phase 1 to derive quantisation scales.
  2. For each GELU layer, build a :class:`~integer_gelu.GELULut` calibrated
     with that layer's activation range.
  3. Compare each LUT against FP32 GELU over all 256 INT8 input points and
     collect error metrics (max abs error, mean abs error, RMSE).
  4. Save comparison plots (FP32 curve vs LUT step function) for each layer.
  5. Persist all error metrics to a JSON summary file.
"""

from __future__ import annotations

import logging

from src.config import IntegerGELUConfig

logger = logging.getLogger(__name__)


def run(config: IntegerGELUConfig) -> None:
    """Execute the Phase 3 integer GELU exploration pipeline end-to-end.

    All artefacts (per-layer LUT comparison PNGs and ``lut_metrics.json``)
    are written under ``config.output_dir``, which is created if it does not
    exist.

    Parameters
    ----------
    config:
        Fully-specified :class:`~config.IntegerGELUConfig` instance.

    Raises
    ------
    NotImplementedError
        Always; stub implementation pending Phase 3 development.
    """
    raise NotImplementedError
