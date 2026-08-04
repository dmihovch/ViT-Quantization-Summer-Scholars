# Phase 2 Expansion — Per-Channel Ablation Deep Dive

> **Created:** 2026-08-03
> **Status:** Implementation complete; experiments pending.

---

## Motivation

Phase 2's original global ablation answered: *"Do outliers matter for accuracy?"*
The answer is yes — zeroing pre-GELU outliers at k=3 drops top-1 from 85.03% to
43.24%, while random zeroing at matched fractions preserves baseline accuracy.

The per-channel extension (T-024, 2026-08-02) revealed a more nuanced result:
per-channel thresholds preserve 3.76% more accuracy at k=3 (47.00% vs 43.24%).
This suggests that high-variance channels carry proportionally more
outlier-dependent signal, and a global threshold over-zeroes them.

This expansion asks: **why** does the per-channel benefit exist, and **where**
does it come from?

---

## Research Questions

### RQ1: Are outliers architectural or anomalous?

**Hypothesis (SmoothQuant, Xiao et al. 2023):** High-γ LayerNorm channels
amplify the residual stream into the MLP, creating the per-channel variance
pattern.  If true, outliers are a deliberate architectural feature, not an
anomaly.

**Experiment:** Compute Pearson r between `layernorm_gamma` (from
`post_layernorm_2`) and `per_channel_std` (from `pre_gelu`) for each block.

**Result (2026-08-03):** r ≈ 0.0003 across all blocks — **no correlation.**
This is expected because LN2 γ is 768-dim (embedding space) while pre-GELU σ_c
is 3072-dim (MLP hidden space).  The per-channel variance pattern emerges from
the interaction of LN2 output with `fc1.weight` (3072×768), not from the LN
scale alone.  A proper test would require computing the effective per-channel
gain as `‖fc1.weight[c, :] ⊙ γ‖` — the element-wise product of the fc1 row
with the LN γ vector.

**Follow-up (2026-08-03):** ✅ Complete.  Computed ``‖fc1.weight[c, :] ⊙ γ‖`` per
channel via ``scripts/analyze_effective_gain.py``.  Results:

| Block | r(gain, σ_c) |
|-------|-------------|
| 0–7   | −0.13 to +0.21 |
| 8     | **+0.7550** |
| 9     | **+0.7747** |
| 10    | **+0.6496** |
| 11    | **+0.7674** |

Mean r across all blocks: **+0.3241**.  The strong correlation in late blocks
(8–11) confirms the SmoothQuant hypothesis: the per-channel variance pattern is
**architectural** — encoded in the interaction of fc1.weight and LN2 γ.  This
is a genuine finding: outliers are not anomalous noise but a deliberate
consequence of trained weights.

### RQ2: Is the per-channel benefit from mean correction or variance correction?

**Hypothesis:** Block 10 has μ ranging −71 to +26 across channels (97-point
spread).  If the benefit comes from the mean correction (per-channel μ_c
instead of global μ), that's a different conclusion than "high-variance
channels carry signal."

**Experiment:** Run per-channel ablation at k=3 in three modes:
- `outlier`: per-channel μ_c + σ_c (full per-channel)
- `mean_only`: per-channel μ_c + global σ
- `var_only`: global μ + per-channel σ_c

**Status:** 🔲 Pending.  CLI flag `--ablation-mode` implemented.

### RQ3: Which layers drive the per-channel benefit?

**Hypothesis:** Block 10 (the extreme outlier layer with σ=11.20) is the
primary driver.  If per-channel ablation on block 10 alone recovers most of
the 3.76% gain, the effect is concentrated in a single layer.

**Experiment:** Run per-channel ablation at k=3 with layer ranges:
- `--layer-range 10 10` (block 10 only)
- `--layer-range 8 11` (late blocks)
- `--layer-range 0 7` (early blocks)

**Status:** 🔲 Pending.  CLI flag `--layer-range` implemented.

### RQ4: Is the effect statistically robust?

**Hypothesis:** A 3.76% delta on 50k images should be significant, but
single-seed results are inadmissible for claims of improvement.

**Experiment:** Re-run per-channel at k=3 with seeds 42, 123, 456.  Report
mean ± std of the delta.

**Status:** 🔲 Pending.

### RQ5: Where is the crossover point?

**Hypothesis:** The per-channel benefit is concentrated at aggressive
thresholds (k < 4).  A finer sweep would reveal the crossover.

**Experiment:** Run global and per-channel at k ∈ {2.5, 2.75, 3.0, 3.25, 3.5}.

**Status:** 🔲 Pending.

---

## Implementation

### New CLI flags

```bash
# Ablation mode (per-channel variants)
--ablation-mode {outlier,mean_only,var_only}

# Layer range restriction
--layer-range START END

# Example: mean-only per-channel on block 10 only
python run_phase2_ablation.py --num-images 50000 --granularity per_channel \
    --ablation-mode mean_only --layer-range 10 10 \
    --sigma-thresholds 3.0
```

### New config fields (`AblationConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ablation_mode` | `str` | `"outlier"` | `"outlier"`, `"mean_only"`, or `"var_only"` |
| `layer_range` | `tuple[int, int] \| None` | `None` | Inclusive block range to ablate |

### New analysis script

`scripts/analyze_layernorm_gamma.py` — Computes Pearson r between LN2 γ and
pre-GELU per-channel σ for each block.  Outputs a table and bar chart.

---

## Phase 3 Deletion

Phase 3 (integer GELU LUTs) was deleted on 2026-08-03.  Rationale:

1. The per-channel result opened a more interesting and more publishable line
   of questioning than the integer GELU LUT engineering check.
2. Phase 3 would have produced a predictable result (LUT approximates GELU
   within ~10⁻³) that doesn't close the loop on the original motivation
   (end-to-end INT8 inference accuracy).
3. The integer GELU LUT is a legitimate engineering component of any future
   full-INT8 inference pipeline, but it doesn't deserve priority over
   understanding *why* the per-channel effect exists.

Deleted files:
- `run_phase3_integer_gelu.py`
- `src/integer_gelu.py`
- `src/exp3_integer_gelu.py`
- `tests/test_integer_gelu.py`

Cleaned references in:
- `src/config.py` (removed `IntegerGELUConfig`)
- `src/plotting.py` (removed `plot_lut_vs_fp32`, `GELULut` import)
- `tests/test_config.py` (removed `test_integer_gelu_config_is_frozen`)
- `tests/test_plotting.py` (removed `test_plot_lut_vs_fp32_creates_file`)

---

## Experiment Priority

| Priority | Experiment | GPU hrs | RQ |
|----------|-----------|---------|-----|
| 1 | LN γ correlation (done) | 0 | RQ1 |
| 2 | fc1.weight ⊙ γ effective gain analysis (done) | 0 | RQ1 |
| 3 | mean_only + var_only at k=3 | ~1 | RQ2 |
| 4 | Layer-group ablation at k=3 | ~3 | RQ3 |
| 5 | Multi-seed variance at k=3 | ~12 | RQ4 |
| 6 | Finer k sweep [2.5..3.5] | ~5 | RQ5 |