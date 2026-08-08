"""Shared utilities for the plotting modules.

Provides canonical sort keys, regex patterns, label formatters, and
colour conventions used by both :mod:`src.plotting` and
:mod:`src.plotting_poster`.
"""

from __future__ import annotations

import re

# Regex for extracting numeric block index from site identifiers.
_BLOCK_RE: re.Pattern[str] = re.compile(r"blocks\.(\d+)")


def site_sort_key(site_id: str) -> tuple[int, int]:
    """Return a numeric sort key for a site identifier.

    Sorts ``patch_embed/...`` first ``(0, 0)``, then ``blocks.{N}/...`` by
    numeric N ``(1, N)``.  Unknown prefixes sort last ``(2, 0)``.

    Parameters
    ----------
    site_id:
        Site identifier string, e.g. ``"blocks.5/pre_gelu"``.

    Returns
    -------
    tuple[int, int]
        Sort key for use with ``sorted(key=...)``.
    """
    if site_id.startswith("patch_embed"):
        return (0, 0)
    m = _BLOCK_RE.search(site_id)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


def block_sort_key(site_id: str) -> tuple[int, int | str]:
    """Return a sort key for the block portion of a site identifier."""
    if site_id.startswith("patch_embed"):
        return (0, 0)
    if site_id.startswith("blocks."):
        try:
            n = int(site_id.split(".", 1)[1].split("/")[0])
            return (1, n)
        except (ValueError, IndexError):
            pass
    return (2, site_id)


def format_site_label(site_id: str) -> str:
    """Convert a raw site identifier to a human-readable label.

    ``"blocks.3/pre_gelu"`` → ``"Block 3 / pre-GELU"``
    ``"blocks.0/post_softmax"`` → ``"Block 0 / post-softmax"``
    ``"patch_embed/residual_stream"`` → ``"Patch Embed / residual stream"``

    Parameters
    ----------
    site_id:
        Raw site identifier string.

    Returns
    -------
    str
        Human-readable label.
    """
    if "/" not in site_id:
        return site_id.replace("_", " ")

    prefix, site = site_id.split("/", 1)

    # Format the block/prefix part.
    if prefix.startswith("blocks."):
        block_num = prefix.split(".", 1)[1]
        prefix_label = f"Block {block_num}"
    elif prefix == "patch_embed":
        prefix_label = "Patch Embed"
    else:
        prefix_label = prefix

    # Format the site part.
    site_label = site.replace("_", " ")

    return f"{prefix_label} / {site_label}"


def extract_block_index(site_id: str) -> int | None:
    """Extract the numeric block index from a site identifier.

    Returns ``None`` for non-block sites (e.g. ``patch_embed/...``).

    Parameters
    ----------
    site_id:
        Site identifier string.

    Returns
    -------
    int or None
        Block index, or ``None`` if not a block site.
    """
    m = _BLOCK_RE.search(site_id)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Canonical colour conventions
# ---------------------------------------------------------------------------

# Analytical plots (src/plotting.py).
ANALYTICAL_COLORS: dict[str, str] = {
    "global": "coral",
    "per_channel": "teal",
    "baseline": "gray",
    "positive": "teal",
    "negative": "coral",
    "random": "#BBBBBB",
    "histogram": "steelblue",
}

# Poster plots (src/plotting_poster.py) — Paul Tol-inspired.
POSTER_PALETTE: dict[str, str] = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "gray": "#BBBBBB",
    "dark": "#222222",
    "coral": "#CC3311",
    "teal": "#009988",
}

# Canonical axis labels.
LABELS: dict[str, str] = {
    "sigma_threshold": "Sigma threshold k",
    "pct_zeroed": "% Elements Zeroed",
    "accuracy": "Top-1 Accuracy (%)",
    "block": "Block",
    "channel_index": "Channel index",
    "head_index": "Head index",
    "entropy": "Mean entropy (nats)",
    "delta_entropy": "Δ Entropy (nats)",
    "delta_accuracy": "Δ Top-1 Accuracy (%)",
    "per_channel_std": "Per-channel STD",
    "effective_channels": "Effective Channels",
    "degradation": "Accuracy loss per 1% sparsity (pp / %)",
    "ln2_ratio": "‖LN2(x)‖₂ / ‖x_skip‖₂",
}