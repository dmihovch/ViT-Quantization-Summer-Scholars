"""Regenerate plots from an existing profiling_result.json without re-profiling.

Use this when you've already run Phase 1 profiling and just want updated
visualizations (e.g., after fixing sort ordering, plot styling, or adding
new plot types).

Usage:
    python scripts/regenerate_plots.py \
        --input outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/phase1-profiling/seed_42/ \
        --plots heatmaps entropy histograms

    # Heatmaps only (fast, no model needed):
    python scripts/regenerate_plots.py \
        --input outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/phase1-profiling/seed_42/ \
        --plots heatmaps entropy

    # Histograms only (requires model + GPU):
    python scripts/regenerate_plots.py \
        --input outputs/phase1-profiling/seed_42/profiling_result.json \
        --output-dir outputs/phase1-profiling/seed_42/ \
        --plots histograms
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nnsight import NNsight

from src.data_loader import build_val_loader
from src.model import load_vit
from src.plotting import (
    plot_activation_histogram,
    plot_attention_entropy_heatmap,
    plot_per_channel_std_heatmap,
)
from src.profiler import (
    LayerStats,
    SiteId,
    histogram_profile_vit,
    load_profiling_result,
)
from src.utils import ensure_dir, get_device, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _regenerate_heatmaps(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate per-channel std heatmaps from existing stats."""
    per_channel: dict[str, list[float]] = {
        key: s.per_channel_std
        for key, s in stats.items()
        if s.per_channel_std is not None
    }
    if not per_channel:
        logger.warning("No per_channel_std data found; skipping heatmaps.")
        return

    by_dim: dict[int, dict[str, list[float]]] = {}
    for key, stds in per_channel.items():
        d = len(stds)
        by_dim.setdefault(d, {})[key] = stds

    for d, group in sorted(by_dim.items()):
        suffix = f"_d{d}" if len(by_dim) > 1 else ""
        out_path = output_dir / f"per_channel_std_heatmap{suffix}.png"
        plot_per_channel_std_heatmap(group, out_path)
        logger.info("Per-channel σ heatmap (D=%d) written to %s", d, out_path)


def _regenerate_entropy_heatmaps(
    stats: dict[SiteId, LayerStats],
    output_dir: Path,
) -> None:
    """Regenerate attention entropy heatmaps from existing stats."""
    cls_entropies: dict[str, list[float]] = {
        key: s.attention_entropy_cls
        for key, s in stats.items()
        if s.attention_entropy_cls is not None
    }
    patch_entropies: dict[str, list[float]] = {
        key: s.attention_entropy_patches
        for key, s in stats.items()
        if s.attention_entropy_patches is not None
    }
    if cls_entropies:
        plot_attention_entropy_heatmap(
            cls_entropies,
            output_dir / "attention_entropy_cls_heatmap.png",
            title="CLS query attention entropy per head (nats)",
        )
        logger.info("CLS entropy heatmap written.")
    if patch_entropies:
        plot_attention_entropy_heatmap(
            patch_entropies,
            output_dir / "attention_entropy_patches_heatmap.png",
            title="Patch query mean attention entropy per head (nats)",
        )
        logger.info("Patch entropy heatmap written.")
    if not cls_entropies and not patch_entropies:
        logger.warning("No entropy data found; skipping entropy heatmaps.")


def _regenerate_histograms(
    data_dir: Path,
    batch_size: int,
    block_indices: tuple[int, ...],
    output_dir: Path,
) -> None:
    """Regenerate activation histograms (requires model load)."""
    device = get_device()
    model, transform = load_vit(device)
    wrapped = NNsight(model)

    loader = build_val_loader(
        data_dir, transform, batch_size, None, device, shuffle=True,
    )
    images, _ = next(iter(loader))

    with torch.no_grad():
        raw_tensors = histogram_profile_vit(
            wrapped, images.to(device), block_indices,
        )

    hist_dir = output_dir / "histograms"
    ensure_dir(hist_dir)
    for key, tensor in raw_tensors.items():
        activations = tensor.detach().cpu().numpy().ravel().astype("float32")
        safe_key = key.replace("/", "_").replace(".", "_")
        plot_activation_histogram(
            activations=activations,
            layer_name=key,
            output_path=hist_dir / f"{safe_key}.png",
            log_scale=True,
        )
    logger.info("Wrote %d histogram PNGs to %s", len(raw_tensors), hist_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Phase 1 plots from existing profiling_result.json.",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to profiling_result.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write regenerated plots.",
    )
    parser.add_argument(
        "--plots", type=str, nargs="+",
        default=["heatmaps", "entropy"],
        choices=["heatmaps", "entropy", "histograms"],
        help="Which plots to regenerate (default: heatmaps entropy).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="ImageNet val directory (only needed for --plots histograms).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for histogram pass.",
    )
    parser.add_argument(
        "--all-blocks", action="store_true",
        help="Generate histograms for all 12 blocks (default: 0, 5, 11 only).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    ensure_dir(args.output_dir)

    result = load_profiling_result(args.input)
    logger.info("Loaded %d sites from %s", len(result.stats), args.input)

    plot_types = set(args.plots)

    if "heatmaps" in plot_types:
        _regenerate_heatmaps(result.stats, args.output_dir)
    if "entropy" in plot_types:
        _regenerate_entropy_heatmaps(result.stats, args.output_dir)
    if "histograms" in plot_types:
        block_indices = tuple(range(12)) if args.all_blocks else (0, 5, 11)
        _regenerate_histograms(
            args.data_dir, args.batch_size, block_indices, args.output_dir,
        )

    logger.info("Done. Plots in %s", args.output_dir)


if __name__ == "__main__":
    main()
