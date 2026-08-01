"""Smoke test: verify nnsight 0.7.0 supports tensor replacement at intervention points.

This is the single most critical verification before Phase 2 implementation.
If nnsight cannot replace intermediate tensors inside a trace, the entire
nnsight-based ablation approach must be rethought.

Tests three intervention sites:
1. pre_gelu: block.mlp.act.input replacement
2. residual_stream: block.norm1.input replacement
3. pre_softmax: attn.softmax.input replacement

Run: python scripts/smoke_test_nnsight_intervention.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from nnsight import NNsight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import disable_fused_attn, load_vit
from src.utils import get_device, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _make_dummy_batch(device: torch.device) -> torch.Tensor:
    """Create a single dummy image batch (B=2, 3×224×224)."""
    return torch.randn(2, 3, 224, 224, device=device)


def test_pre_gelu_intervention(
    wrapped: NNsight,
    inner_model,
    device: torch.device,
) -> bool:
    """Verify we can zero block 0's pre-GELU activations inside a trace.

    Strategy: run two traces on the same input — one without intervention
    (baseline logits) and one with block.0.mlp.act.input zeroed.  If the
    logits differ, the intervention worked.
    """
    batch = _make_dummy_batch(device)

    # Baseline: no intervention.  Save logits inside the trace.
    with torch.no_grad():
        with wrapped.trace(batch):
            baseline = wrapped.output.save()
        # nnsight >=0.3: .save() returns a concrete tensor directly.
        baseline_logits = baseline.clone()

    # Intervention: zero block 0's pre-GELU activations.
    with torch.no_grad():
        with wrapped.trace(batch):
            gelu_input = wrapped.blocks[0].mlp.act.input
            wrapped.blocks[0].mlp.act.input = torch.zeros_like(gelu_input)
            intervened = wrapped.output.save()
        intervened_logits = intervened.clone()

    diff = (baseline_logits - intervened_logits).abs().max().item()
    logger.info("pre_gelu intervention: max logit diff = %.6f", diff)

    if diff < 1e-6:
        logger.error(
            "pre_gelu intervention FAILED: logits unchanged. "
            "nnsight may not support tensor replacement at mlp.act.input."
        )
        return False

    logger.info("pre_gelu intervention: PASSED (logits changed, intervention works)")
    return True


def test_residual_stream_intervention(
    wrapped: NNsight,
    inner_model,
    device: torch.device,
) -> bool:
    """Verify we can zero the residual stream entering block 1."""
    batch = _make_dummy_batch(device)

    with torch.no_grad():
        with wrapped.trace(batch):
            baseline = wrapped.output.save()
        baseline_logits = baseline.clone()

    with torch.no_grad():
        with wrapped.trace(batch):
            residual = wrapped.blocks[1].norm1.input
            wrapped.blocks[1].norm1.input = torch.zeros_like(residual)
            intervened = wrapped.output.save()
        intervened_logits = intervened.clone()

    diff = (baseline_logits - intervened_logits).abs().max().item()
    logger.info("residual_stream intervention: max logit diff = %.6f", diff)

    if diff < 1e-6:
        logger.error("residual_stream intervention FAILED: logits unchanged.")
        return False

    logger.info("residual_stream intervention: PASSED")
    return True


def test_post_softmax_intervention(
    wrapped: NNsight,
    inner_model,
    device: torch.device,
) -> bool:
    """Verify we can intervene on post-softmax attention weights.

    Note: timm's Attention module computes softmax inline (no module boundary).
    The closest accessible intervention point is attn.attn_drop.input,
    which is the post-softmax attention weight matrix.

    For pre_softmax intervention in Phase 2, we will need to reconstruct
    QKᵀ/√d from qkv.output (same as Phase 1's _register_pre_softmax_saves),
    apply the zeroing mask, and replace the attention computation.  This
    requires a more invasive trace pattern — verified separately.
    """
    batch = _make_dummy_batch(device)

    with torch.no_grad():
        with wrapped.trace(batch):
            baseline = wrapped.output.save()
        baseline_logits = baseline.clone()

    with torch.no_grad():
        with wrapped.trace(batch):
            attn_weights = wrapped.blocks[0].attn.attn_drop.input
            wrapped.blocks[0].attn.attn_drop.input = torch.zeros_like(attn_weights)
            intervened = wrapped.output.save()
        intervened_logits = intervened.clone()

    diff = (baseline_logits - intervened_logits).abs().max().item()
    logger.info("post_softmax intervention: max logit diff = %.6f", diff)

    if diff < 1e-6:
        logger.error(
            "post_softmax intervention FAILED: logits unchanged. "
            "nnsight may not support tensor replacement at attn.attn_drop.input."
        )
        return False

    logger.info("post_softmax intervention: PASSED")
    return True


def main() -> None:
    """Run all three intervention smoke tests."""
    seed_everything(42)
    device = get_device()
    logger.info("Device: %s", device)

    logger.info("Loading ViT-B/16...")
    model, _transform = load_vit(device)
    wrapped = NNsight(model)
    inner_model = wrapped._model

    results: dict[str, bool] = {}

    logger.info("=== Test 1: pre_gelu intervention ===")
    results["pre_gelu"] = test_pre_gelu_intervention(wrapped, inner_model, device)

    logger.info("=== Test 2: residual_stream intervention ===")
    results["residual_stream"] = test_residual_stream_intervention(wrapped, inner_model, device)

    logger.info("=== Test 3: post_softmax intervention ===")
    results["post_softmax"] = test_post_softmax_intervention(wrapped, inner_model, device)

    logger.info("=" * 50)
    all_passed = all(results.values())
    for site, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info("  %s: %s", site, status)

    if all_passed:
        logger.info("ALL SMOKE TESTS PASSED — nnsight intervention is viable for Phase 2.")
    else:
        logger.error(
            "SOME TESTS FAILED — nnsight intervention may not work for all sites. "
            "Investigate failed sites before Phase 2 implementation."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()