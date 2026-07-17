"""LUT-based integer GELU approximation for Phase 3.

Each GELU layer gets its own look-up table (LUT) that maps 256 possible INT8
inputs to INT8 outputs.  This mirrors the approach taken in many integer-only
inference engines where GELU cannot be efficiently computed on-the-fly and is
pre-tabulated during a calibration step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GELULut:
    """Look-up table approximating GELU for a single transformer layer.

    Attributes
    ----------
    layer_name:
        Fully-qualified name of the GELU module this LUT corresponds to.
    scale_in:
        Input quantisation scale.  Derived from the layer's activation range
        as ``scale_in = T / 127`` where ``T`` is the calibrated maximum.
    scale_out:
        Output quantisation scale; determined by the output range of
        ``GELU(x)`` over the calibration set.
    lut:
        256-element list of INT8 values.  Index ``i`` corresponds to the
        integer input value ``i - 128``, covering the range ``[-128, 127]``.
    """

    layer_name: str
    scale_in: float
    scale_out: float
    lut: list[int]


def build_lut(scale_in: float, scale_out: float) -> list[int]:
    """Pre-compute a 256-entry GELU look-up table for given quantisation scales.

    For each integer index ``i`` in ``[0, 255]``, the corresponding INT8
    input value is ``i - 128`` (covering the full signed INT8 range).  The
    FP32 dequantised value is ``(i - 128) * scale_in``.  GELU is applied in
    FP32 and the result is requantised and clipped:

    .. code-block:: text

        LUT[i] = clip(round(GELU((i - 128) * scale_in) / scale_out), -128, 127)

    Parameters
    ----------
    scale_in:
        Dequantisation scale for input INT8 values.
    scale_out:
        Quantisation scale for the output.

    Returns
    -------
    list[int]
        256-element list with values in ``[-128, 127]``.
    """
    raise NotImplementedError


def apply_lut(tensor: torch.Tensor, lut: GELULut) -> torch.Tensor:
    """Apply a pre-computed GELU LUT to an integer-valued tensor.

    This is a pure function; the input tensor is not modified.

    Parameters
    ----------
    tensor:
        Integer tensor whose values represent quantised activations in the
        range ``[-128, 127]``.  Typically of dtype ``torch.int8``.
    lut:
        The :class:`GELULut` to use for the table lookup.  ``lut.lut[i]`` is
        accessed at index ``value + 128`` for each element value.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``tensor``, with INT8 GELU
        approximation applied element-wise.
    """
    raise NotImplementedError


def compare_lut_vs_fp32(lut: GELULut, scale_in: float) -> dict[str, float]:
    """Quantify the approximation error of the LUT relative to FP32 GELU.

    Evaluates both FP32 GELU and the LUT approximation on all 256 possible
    INT8 input values and returns summary error metrics.

    Parameters
    ----------
    lut:
        The :class:`GELULut` to evaluate (provides the table and scales).
    scale_in:
        Dequantisation scale used when converting INT8 inputs to FP32 for
        the reference GELU computation; should match ``lut.scale_in``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:

        - ``"max_abs_error"``: maximum absolute difference over all 256 points.
        - ``"mean_abs_error"``: mean absolute difference.
        - ``"rmse"``: root-mean-square error.
    """
    raise NotImplementedError
