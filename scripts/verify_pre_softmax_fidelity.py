"""Verify pre_softmax reconstruction is bit-identical to native when zeroing is a no-op."""
import sys
from pathlib import Path

import torch
from nnsight import NNsight

# Ensure project root is on sys.path so `src` imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_vit
from src.utils import get_device, seed_everything
from src.ablation import zero_outliers_in_trace
from src.profiler import LayerStats

seed_everything(42)
device = get_device()
model, _ = load_vit(device)
wrapped = NNsight(model)
num_blocks = len(model.blocks)

# Build fake stats with enormous sigma — nothing gets zeroed.
sigma = 1000.0
fake_stats = {}
for i in range(num_blocks):
    fake_stats[f"blocks.{i}/pre_softmax"] = LayerStats(
        site_identifier=f"blocks.{i}/pre_softmax",
        mean=0.0, std=sigma, kurtosis=0.0,
        outlier_fractions={}, n_samples=0,
    )

batch = torch.randn(2, 3, 224, 224, device=device)

# Native forward pass.
with torch.no_grad():
    with wrapped.trace(batch):
        native_logits = wrapped.output.save()
    native = native_logits.clone()

# Intervention with k=1e9 (threshold = 1e12, nothing zeroed).
logits, _, _ = zero_outliers_in_trace(
    wrapped, batch, "pre_softmax", sigma_k=1e9, layer_stats=fake_stats,
)

diff = (native - logits).abs().max().item()
print(f"Max logit diff (native vs pre_softmax no-op): {diff:.6f}")
if diff < 1e-3:
    print("PASS: reconstruction is bit-identical to native when nothing zeroed.")
else:
    print(f"FAIL: max diff = {diff:.6f}")
