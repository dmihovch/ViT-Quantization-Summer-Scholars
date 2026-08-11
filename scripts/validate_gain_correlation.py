"""Validate the effective-gain vs σ_c correlation across all 5 seeds.

Computes Pearson and Spearman r for Blocks 8, 9, 10 across seeds 42–46.
Effective gain (‖fc1.weight ⊙ γ‖₂) is identical across seeds (static weights),
so we load fc1 once and correlate against per_seed per_channel_std.

Usage:
    python scripts/validate_gain_correlation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx = np.sqrt(((x - mx) ** 2).sum())
    sy = np.sqrt(((y - my) ** 2).sum())
    if sx == 0 or sy == 0:
        return 0.0
    return float(cov / (sx * sy))


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(x, y)
    return float(r)


def effective_gain(fc1_weight: np.ndarray, ln_gamma: np.ndarray) -> np.ndarray:
    """‖fc1_weight[c,:] ⊙ ln_gamma‖₂ per channel, shape (3072,)."""
    weighted = fc1_weight * ln_gamma[np.newaxis, :]
    return np.linalg.norm(weighted, axis=1)


def main() -> None:
    base = Path("outputs/5-seed-full-run-2026-08-05/phase1-profiling")
    seeds = [42, 43, 44, 45, 46]

    # 1. Load model weights (once — same across seeds).
    print("Loading model weights …")
    from src.model import load_vit
    from src.utils import get_device, seed_everything

    seed_everything(42)
    device = get_device()
    model, _ = load_vit(device)

    fc1_weights: dict[int, np.ndarray] = {}
    for bidx in range(12):
        fc1_weights[bidx] = (
            model.blocks[bidx].mlp.fc1.weight.detach().cpu().numpy()
        )
    del model
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()

    # 2. For each seed, compute effective_gain and correlate with σ_c.
    blocks_of_interest = [8, 9, 10]

    for bidx in blocks_of_interest:
        print(f"\n{'='*70}")
        print(f"Block {bidx}")
        print(f"{'Seed':>6s}  {'Pearson r':>10s}  {'Spearman r':>10s}  {'pc_std min':>10s}  {'pc_std max':>10s}  {'gain min':>10s}  {'gain max':>10s}")
        print(f"{'':->6s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}")

        pearson_vals: list[float] = []
        spearman_vals: list[float] = []

        for seed in seeds:
            json_path = base / f"seed_{seed}" / "profiling_result.json"
            with open(json_path, "r") as f:
                profiling = json.load(f)
            stats = profiling["stats"]

            ln2_sid = f"blocks.{bidx}/post_layernorm_2"
            gelu_sid = f"blocks.{bidx}/pre_gelu"

            ln_gamma = np.array(stats[ln2_sid]["layernorm_gamma"], dtype=np.float64)
            pc_std = np.array(stats[gelu_sid]["per_channel_std"], dtype=np.float64)

            eg = effective_gain(fc1_weights[bidx], ln_gamma)

            p = pearson_r(eg, pc_std)
            s = spearman_r(eg, pc_std)
            pearson_vals.append(p)
            spearman_vals.append(s)

            print(
                f"{seed:6d}  {p:+10.4f}  {s:+10.4f}  "
                f"{pc_std.min():10.4f}  {pc_std.max():10.4f}  "
                f"{eg.min():10.4f}  {eg.max():10.4f}"
            )

        mean_p = np.mean(pearson_vals)
        std_p = np.std(pearson_vals, ddof=1)
        mean_s = np.mean(spearman_vals)
        std_s = np.std(spearman_vals, ddof=1)
        print(f"{'Mean':>6s}  {mean_p:+10.4f}  {mean_s:+10.4f}")
        print(f"{'±1σ':>6s}  {'±'+f'{std_p:.4f}':>10s}  {'±'+f'{std_s:.4f}':>10s}")

    print("\nDone.")


if __name__ == "__main__":
    main()