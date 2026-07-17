"""Tests for the hook registration machinery in :mod:`src.hooks`.

Real ViT model weights are never loaded here — all tests operate on small
synthetic modules so the suite stays fast.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.exceptions import HookRegistrationError
from src.hooks import register_profiling_hooks


def test_register_hooks_raises_on_model_with_no_gelu() -> None:
    """register_profiling_hooks must raise HookRegistrationError for GELU-free models.

    An nn.Linear has no nn.GELU children, so attaching profiling hooks would
    silently produce an empty stats dict without this guard.
    """
    model = nn.Linear(4, 4)
    with pytest.raises(HookRegistrationError):
        register_profiling_hooks(model)
