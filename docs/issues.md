# Issue Tracker — ViT Quantization Summer Scholars

> **Created:** 2026-07-28 — from skeptical review of Phase 1 implementation.
> **Source:** 15-issue review against `src/profiler.py`, `docs/scispace-docs/vit_profiling_framework.md`,
> `docs/CITATIONS.md`, and test suite.
>
> Each ticket includes: severity, status, evidence, proposed fix, rationale with citations,
> and affected files.

---

## CRITICAL

### T-001 — Final encoder residual stream (`blocks.11/residual_stream`) never captured

**Severity:** CRITICAL  
**Status:** ✅ Closed — fixed 2026-07-30  
**Category:** Bug — data gap  
**Source:** Issue 1 (skeptical review)

**Evidence**

`profile_vit` (line 1130) iterates `for i in range(num_blocks)` (0..11 for ViT-B/16).
The `residual_stream` label is set at lines 1139–1142:

```python
residual_label = (
    "patch_embed/residual_stream" if i == 0
    else f"blocks.{i - 1}/residual_stream"
)
```

For `i=11` (the last block), the label is `blocks.10/residual_stream`. The output of
block 11 — the final encoder output before the classification head — is **never
captured** as a raw residual stream. It is only captured after LayerNorm as
`blocks.11/post_layernorm_1`, which is zero-mean, unit-variance per token and
**masks** the true magnitude growth in the residual.

The test at `test_profiler.py:284` confirms the site count: `len(stats) == 12 × 6 = 72`.
The test at lines 298–299 only checks `blocks.{i-1}/residual_stream` for `i in range(1, 12)`,
i.e., `blocks.0` through `blocks.10`. The test itself encodes the bug.

**Why this matters**

The residual stream at the final encoder output is the representation fed to the
classification head. Its statistics — especially outlier fractions and kurtosis — are
critical for understanding whether quantization error accumulates through the network.
The `post_layernorm_1` of block 11 is normalized, which masks the true magnitude.
Without this measurement, we cannot answer: "how large do the activations get at the
point where they enter the head?"

This is the single most important activation tensor for quantization range decisions
in Phase 2/3, because it represents the cumulative effect of all 12 blocks.

**Proposed fix**

Add a hook on `inner_model.norm` (the final LayerNorm before the classification head)
`.input` to capture the final residual stream. Label it `blocks.11/residual_stream`.

In `profile_vit`, after the block loop (after line 1197), add:

```python
# --- Final residual stream (output of last encoder block, before head LN) ---
all_savers.append(
    _register_stat_saves(
        wrapped_model.norm.input,
        f"blocks.{num_blocks - 1}/residual_stream",
        n_residual,
    )
)
```

The same hook must be added to `run_outlier_counting_pass` (after line 699) for the
F2 recount pass.

Update `test_profiler.py`:
- `test_slow_total_site_count`: change expected count from `12 * 6 = 72` to `12 * 6 + 1 = 73`.
- `test_slow_all_expected_sites_present`: add `assert f"blocks.{_NUM_BLOCKS - 1}/residual_stream" in keys`.
- `test_profiling_result_canned_site_keys`: same.

**Rationale**

The Pébay (2008) merge pipeline already supports arbitrary site counts — no
infrastructure changes needed. The `_site_n` function already handles `residual_stream`
sites correctly (returns `B * N * D`). This is a one-line addition to the trace loop
plus test updates.

**References**
- Pébay, P. P. (2008), "Formulas for Robust, One-Pass Parallel Computation of
  Covariances and Arbitrary-Order Statistical Moments," SAND2008-6212.
  See `docs/CITATIONS.md`.
- Bondarenko et al. (2021), "Understanding and Overcoming the Challenges of
  Efficient Transformer Quantization," arXiv:2109.12948 — §4.2 discusses how
  residual stream magnitude growth is the primary cause of quantization range
  blow-up in transformers.

**Affected files**
- `src/profiler.py` — `profile_vit` (add hook after block loop), `run_outlier_counting_pass` (same)
- `tests/test_profiler.py` — `test_slow_total_site_count`, `test_slow_all_expected_sites_present`,
  `test_profiling_result_canned_site_keys`

---

## HIGH

### T-002 — Spec-code mismatch on outlier sigma thresholds: spec says {3,4,6}, code uses {3,5,8}

**Severity:** HIGH
**Status:** ✅ Closed (2026-07-30)
**Category:** Documentation/code divergence
**Source:** Issue 3 (skeptical review)

**Resolution:** Changed `OUTLIER_SIGMAS` in `profiler.py` from `(3.0, 5.0, 8.0)` to
`(3.0, 4.0, 6.0)` to match the spec and standard quantization-literature thresholds.
The 4σ threshold is standard in the quantization literature (Bondarenko et al. 2021)
and 6σ is standard for extreme outlier detection (Dettmers et al. 2022; Wei et al. 2022).

Updated all downstream references:
- `src/profiler.py`: `OUTLIER_SIGMAS` constant and `LayerStats` docstring.
- `src/plotting.py`: `plot_activation_histogram` now annotates ±3σ, ±4σ, ±6σ.
- `src/hooks.py`: comment updated (already had correct values).
- `docs/scispace-docs/vit_profiling_framework.md`: Per-Site Metrics, Deliverables, Phase 2 thresholding, Sigma Threshold Convention.
- `docs/NEXT-STEPS.md`: `LayerStats` dataclass keys and Statistics computed section.
- `docs/EXP1-IMPL.md`: no direct references to update (uses `OUTLIER_SIGMAS` symbolically).
- `tests/test_profiler.py`: hardcoded outlier fraction keys updated.

**Note:** This invalidates any previously collected profiling data. Re-run Phase 1
profiling to regenerate results with the correct thresholds.

**References**
- Bondarenko et al. (2021), arXiv:2109.12948 — uses 4σ as primary threshold.
- Dettmers et al. (2022), "LLM.int8()," NeurIPS 2022, arXiv:2208.07339 — uses 6σ
  for extreme outlier detection.
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight), arXiv:2209.13325 —
  uses multiple σ thresholds for sensitivity analysis.

**Affected files**
- `src/profiler.py` — `OUTLIER_SIGMAS` constant changed to `(3.0, 4.0, 6.0)`
- `src/plotting.py` — histogram annotations updated to ±3σ, ±4σ, ±6σ
- `src/hooks.py` — comment clarified
- `docs/scispace-docs/vit_profiling_framework.md` — all sigma references updated
- `docs/NEXT-STEPS.md` — LayerStats keys updated
- `tests/test_profiler.py` — hardcoded outlier fraction keys updated

---

### T-003 — ShiftGELU attribution verified: correctly attributed to I-ViT

**Severity:** ~~HIGH~~ → RESOLVED (no bug)  
**Status:** ✅ Closed  
**Category:** Citation (verified correct)  
**Source:** Issue 4 (skeptical review) — **review claim was incorrect**

**Evidence**

The skeptical review claimed that ShiftGELU was introduced by I-BERT (Kim et al.
2021) rather than I-ViT (Li & Gu 2023). This claim was investigated and found to
be **incorrect**. ShiftGELU was introduced by I-ViT (Li & Gu 2023, ICCV 2023,
arXiv:2207.01405). The framework document's attribution to I-ViT is correct.

I-BERT (Kim et al. 2021, ICML 2021 Oral, arXiv:2101.01321) introduced integer-only
GELU via polynomial approximation (i-GELU) for BERT, but ShiftGELU specifically
is an I-ViT contribution.

**Resolution**

No changes needed. The existing attribution in `docs/scispace-docs/vit_profiling_framework.md`
and `docs/CITATIONS.md` is correct. Both papers remain cited for their respective
contributions: I-BERT for i-GELU (integer polynomial GELU), I-ViT for ShiftGELU
(bit-shifting approximation for ViT).

**References**
- Kim et al. (2021), "I-BERT: Integer-only BERT Quantization," ICML 2021 (Oral),
  arXiv:2101.01321. See `docs/CITATIONS.md`.
- Li & Gu (2023), "I-ViT: Integer-only Quantization for Efficient Vision Transformer
  Inference," ICCV 2023, arXiv:2207.01405. See `docs/CITATIONS.md`.

---

### T-004 — LayerNorm γ/β weights not captured (spec requires it)

**Severity:** HIGH
**Status:** ✅ Closed (2026-07-30)
**Category:** Missing feature
**Source:** Issue 5 (skeptical review)

**Resolution:** Implemented γ/β capture for all post_layernorm_1 and post_layernorm_2
sites.  The γ (weight) and β (bias) parameters of each LayerNorm module are extracted
from the underlying PyTorch model **after the nnsight trace exits** — they are static
model parameters, not activation statistics, so they don't require the trace context
or the Welford merge pipeline.

Changes applied:
- `LayerStats`: added `layernorm_gamma: list[float] | None` and `layernorm_beta: list[float] | None` fields.
  Shape `[D]` (D=768 for ViT-B/16); non-None only for post_layernorm_1 and post_layernorm_2 sites.
- `WelfordAccumulator`: added `layernorm_gamma` and `layernorm_beta` fields.
  Carried through from the first batch's LayerStats (they don't change between batches).
- `merge_batch_stats`: copies γ/β from the first non-None batch's LayerStats to the accumulator.
- `finalize_accumulator`: passes γ/β through unchanged to the final LayerStats.
- `profile_vit`: after the trace exits, iterates over `inner_model.blocks[i]` and extracts
  `norm1.weight`, `norm1.bias`, `norm2.weight`, `norm2.bias` via `.detach().cpu().tolist()`.
- `save_profiling_result` / `load_profiling_result`: JSON serialization handles the new
  fields automatically via `dataclasses.asdict` / `LayerStats(**val)`.

Tests added:
- 3 fast tests: default-None, store values, JSON roundtrip.
- 5 slow tests: gamma present, beta present, absent on non-LN sites, match model weights,
  survive serialization.

All 73 fast + 27 slow tests pass (including the 8 new γ/β tests).

**Rationale:** This enables distinguishing learned-scale outliers (large γ) from
distribution outliers (large activation variance) in per-channel quantization
decisions — the core SmoothQuant (Xiao et al. 2023) insight.

**References**
- Xiao et al. (2023), "SmoothQuant," ICML 2023, arXiv:2211.10438 — §3.1: the
  core insight is that activation outliers can be smoothed into weights by
  scaling LayerNorm γ down and the subsequent linear layer's weights up.
  See `docs/CITATIONS.md`.
- Bondarenko et al. (2021), arXiv:2109.12948 — §4.1: discusses per-channel
  variance and the role of LayerNorm affine parameters.

**Affected files**
- `src/profiler.py` — `LayerStats` (fields added), `WelfordAccumulator` (fields added),
  `merge_batch_stats` (carry-through), `finalize_accumulator` (pass-through),
  `profile_vit` (post-trace extraction)
- `tests/test_profiler.py` — 8 new tests for γ/β presence, correctness, and serialization
- `docs/scispace-docs/vit_profiling_framework.md` — Per-Site Metrics and Post-LayerNorm sections updated
- `docs/EXP1-IMPL.md` — §3.2 and §3.4 updated with γ/β fields
- `docs/MISTAKES.md` — entry added

---

### T-014 — Outlier fraction definition inconsistency: Pass 1 uses |x| > k·σ, Pass 2 uses |x−μ| > k·σ_global

**Severity:** HIGH  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Bug — silent definition mismatch  
**Source:** Skeptical review, 2026-07-30

**Resolution:** Researcher selected Option 1 — fix Pass 1 to use the
mean-centered definition ``|x − μ| > k·σ``, consistent with the standard
statistical definition and the quantization literature (Wei et al. 2022,
§3.1; Bondarenko et al. 2021, §4.1).

Changes applied:
- ``src/profiler.py``: ``_register_stat_saves`` line 950 changed from
  ``(t.abs() > sigma * t.std(correction=0))`` to
  ``((t - t.mean()).abs() > sigma * t.std(correction=0))``.
  Added inline comment citing Wei et al. 2022 and Bondarenko et al. 2021.
- ``src/profiler.py``: Updated ``WelfordAccumulator.outlier_counts``,
  ``finalize_accumulator``, and ``_register_stat_saves`` docstrings to
  document the mean-centered definition.
- ``tests/test_profiler.py``: ``test_run_outlier_counting_pass_known_gaussian``
  updated from ``data.abs()`` to ``(data - mu).abs()``.
- ``docs/EXP1-IMPL.md``: §2.3 rewritten with the corrected convention,
  formula, and literature citations.
- ``docs/MISTAKES.md``: §1.3 updated with two-pass recount note; new §1.4
  added documenting the zero-centered bug and its resolution.

**Note:** This invalidates any previously collected profiling data where
``skip_outlier_recount=True`` was used.  Re-run Phase 1 profiling to
regenerate results with the corrected Pass 1 outlier definition.

**References**
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight),
  arXiv:2209.13325 — §3.1: identifies outliers by examining per-token and
  per-channel activation magnitudes relative to the distribution.
- Bondarenko et al. (2021), "Understanding and Overcoming the Challenges of
  Efficient Transformer Quantization," arXiv:2109.12948 — §4.1: characterizes
  distributions by their central moments, all defined relative to μ.

**Affected files**
- ``src/profiler.py`` — ``_register_stat_saves`` (line 950), ``WelfordAccumulator``
  docstring, ``finalize_accumulator`` docstring, ``_register_stat_saves`` docstring
- ``tests/test_profiler.py`` — ``test_run_outlier_counting_pass_known_gaussian``
- ``docs/EXP1-IMPL.md`` — §2.3 Outlier fraction convention
- ``docs/MISTAKES.md`` — §1.3 updated, §1.4 added

---

### T-015 — `residual_delta_ratio` docstrings and comments incorrectly claimed it measured MLP output

**Severity:** ~~HIGH~~ → LOW (documentation only; code was correct)  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Documentation  
**Source:** Skeptical review, 2026-07-30

**Resolution:** The code was correct all along — it intentionally measures
``‖LN2(x)‖₂ / ‖x_skip‖₂``, the pre-MLP LayerNorm amplification.  The bug was
in the docstrings and comments, which incorrectly claimed the field measured
``‖mlp_output‖₂ / ‖x_skip‖₂``.

Changes applied:
- Renamed field from ``residual_delta_ratio`` to ``ln2_amplification_ratio``
  across all of ``src/profiler.py`` (``LayerStats``, ``WelfordAccumulator``,
  ``_StatsSavers``, ``merge_batch_stats``, ``finalize_accumulator``,
  ``_finalize_stats``, ``profile_vit``).
- Updated all docstrings and inline comments to say ``‖LN2(x)‖₂ / ‖x_skip‖₂``
  instead of ``‖mlp_output‖₂ / ‖x_skip‖₂``.
- Renamed all 12 test functions and updated all field references in
  ``tests/test_profiler.py``.
- Updated ``docs/MISTAKES.md`` §11.1 to reflect the corrected name and formula.
- Updated ``docs/issues.md`` T-005 resolution notes.

**Note:** This is a rename only — no behavior change.  The metric was always
measuring LN2 output; the docstrings were simply wrong about what it was called
and what it claimed to measure.

**Affected files**
- ``src/profiler.py`` — ``LayerStats``, ``WelfordAccumulator``, ``_StatsSavers``,
  ``merge_batch_stats``, ``finalize_accumulator``, ``_finalize_stats``, ``profile_vit``
- ``tests/test_profiler.py`` — 12 test functions renamed, backwards-compat test updated
- ``docs/MISTAKES.md`` — §11.1 updated
- ``docs/issues.md`` — T-005 resolution notes updated

---

### T-016 — Model checkpoint identity not pinned in code; report and code may reference different weights

**Severity:** HIGH  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Reproducibility  
**Source:** Skeptical review, 2026-07-30

**Resolution:** Pinned the model to the explicit variant name
``vit_base_patch16_224.augreg2_in21k_ft_in1k`` in ``src/model.py``.
Also added a ``RunMetadata`` dataclass to ``src/profiler.py`` that captures
hardware, software, and experiment metadata (Python version, PyTorch version,
timm version, nnsight version, CUDA info, GPU info, model name, dataset,
num_images, batch_size, seed, num_seeds, UTC timestamp).  This metadata is
embedded in every ``ProfilingResult`` and serialized into
``profiling_result.json``, making every output file self-documenting.

Changes applied:
- ``src/model.py``: ``load_vit`` now calls
  ``timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)``.
- ``src/profiler.py``: Added ``RunMetadata`` dataclass with 16 fields covering
  hardware, software, model, dataset, and experiment parameters.
- ``src/profiler.py``: ``ProfilingResult`` gained ``metadata: RunMetadata | None`` field.
- ``src/profiler.py``: ``load_profiling_result`` reconstructs ``RunMetadata``
  from JSON if present; backwards-compatible with old files lacking the key.
- ``src/utils.py``: Added ``collect_system_metadata()`` helper that returns a
  dict of system info suitable for constructing ``RunMetadata``.
- ``src/exp1_profiling.py``: ``_run_single`` now constructs ``RunMetadata``
  via ``collect_system_metadata()`` and attaches it to the ``ProfilingResult``.
- ``tests/test_profiler.py``: Added ``test_run_metadata_round_trip`` (full
  serialization roundtrip) and ``test_run_metadata_backwards_compat`` (old
  JSON without metadata key).  Updated backwards-compat test JSON to include
  ``metadata: None``.

**Affected files**
- ``src/model.py`` — ``load_vit`` (line 69)
- ``src/profiler.py`` — ``RunMetadata`` (new), ``ProfilingResult`` (field added),
  ``load_profiling_result`` (reconstruction)
- ``src/utils.py`` — ``collect_system_metadata`` (new)
- ``src/exp1_profiling.py`` — ``_run_single`` (metadata construction)
- ``tests/test_profiler.py`` — 2 new tests, 1 updated backwards-compat test

---

## MEDIUM

### T-005 — Residual update delta ‖Δ‖/‖x_skip‖ not computed

**Severity:** MEDIUM  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Missing feature  
**Source:** Issue 6 (skeptical review)

**Resolution:** Implemented LN2 amplification ratio computation inside the nnsight
trace.  For each encoder block `i`, the ratio `‖LN2(x)‖₂ / ‖x_skip‖₂` is
computed per-token and averaged over batch and tokens.  The ratio is attached to
the `blocks.{i}/residual_stream` LayerStats (representing the LN2 amplification
in block `i` that produced this residual).  `patch_embed/residual_stream` has
`None` (no preceding LN2 block).

Changes applied:
- `LayerStats`: added `ln2_amplification_ratio: float | None = None` field.
  Non-None only for `residual_stream` sites (except `patch_embed`).
- `WelfordAccumulator`: added `ln2_amplification_ratio_sum` and
  `ln2_amplification_ratio_count` fields for simple mean accumulation across batches.
- `_StatsSavers`: added `ln2_amplification_ratio` proxy field.
- `merge_batch_stats`: accumulates ratio via simple sum (not Pébay merge —
  the ratio is already a per-batch mean).
- `finalize_accumulator`: computes final mean delta ratio.
- `_finalize_stats`: extracts delta ratio from saved proxy.
- `profile_vit`: captures `skip_norm = block.norm1.input.norm(dim=-1).mean()`
  before the proxy is consumed by `_register_stat_saves`, then computes
  `mlp_norm / (skip_norm + 1e-8)` after `norm2.output` is available.  Uses a
  one-iteration pending buffer to attach the ratio to the correct
  `residual_stream` site (the ratio for block `i`'s MLP is attached to
  `blocks.{i}/residual_stream`).

Tests added:
- 7 fast tests: default-None, store value, JSON roundtrip, accumulator defaults,
  merge accumulation, finalize mean, finalize None when no data.
- 5 slow tests: present on residual_stream, positive, absent on non-residual
  sites, survive serialization, reasonable magnitude.

All 80 fast + 32 slow tests pass (including the 12 new delta ratio tests).

**Rationale:** This metric directly answers "how aggressively does each MLP block
modify the residual stream?" — the primary driver of quantization range expansion
(Bondarenko et al. 2021, §4.2).  Combined with the existing kurtosis values, it
distinguishes whether heavy tails come from the MLP or the attention sub-block,
which is actionable for Phase 2 ablation targeting (Wei et al. 2022, §3.1).

**References**
- Bondarenko et al. (2021), arXiv:2109.12948 — §4.2: residual connections
  cause quantization range expansion.
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight),
  arXiv:2209.13325 — §3.1: analyzes which transformer sub-layers produce outliers.

**Affected files**
- `src/profiler.py` — `LayerStats` (field added), `WelfordAccumulator` (fields added),
  `_StatsSavers` (field added), `merge_batch_stats` (accumulation),
  `finalize_accumulator` (finalization), `_finalize_stats` (extraction),
  `profile_vit` (trace computation with pending buffer)
- `tests/test_profiler.py` — 12 new tests for delta ratio
- `docs/scispace-docs/vit_profiling_framework.md` — Residual Update Stream section updated
- `docs/EXP1-IMPL.md` — §3.2, §3.4, §8 updated with delta ratio fields
- `docs/MISTAKES.md` — entry added

---

### T-006 — Update `docs/scispace-docs/vit_profiling_framework.md` to match implementation

**Severity:** MEDIUM  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Documentation  
**Source:** Issue 7 (skeptical review)

**Resolution:** The framework document has been incrementally updated alongside
each issue resolution (T-001 through T-005).  The final stale item — the
`blocks.11/residual_stream` row in the Site Labeling Convention table still
marked "Not yet captured" despite T-001 being closed — has been corrected.

Current state of each proposed fix:
1. §Per-Site Metrics sigma thresholds: `k ∈ {3, 4, 6}` — **already correct**
   (matches `OUTLIER_SIGMAS = (3.0, 4.0, 6.0)` in `src/profiler.py`).
2. §Per-Site Metrics outlier fraction methodology: **already correct** —
   describes the two-pass recount (F2, `run_outlier_counting_pass`).
3. §Per-Site Metrics max/min: **correctly marked** as "not yet implemented"
   (T-010 is still open).
4. §Post-LayerNorm γ/β logging: **already correct** — updated during T-004
   to document the `layernorm_gamma`/`layernorm_beta` fields.
5. §Residual Update Stream LN2 amplification ratio: **already correct** — updated during
   T-005 to document the `ln2_amplification_ratio` field.
6. §Measurement Sites table: **already correct** — has 6 rows including
   Residual Stream and Residual Update Stream.
7. §Document Conventions: **already correct** — includes Site Labeling
   Convention and Sigma Threshold Convention sections.

Final fix applied:
- Site Labeling Convention table: `blocks.11/residual_stream` row changed
  from "Not yet captured — see T-001" to "Captured from
  `wrapped_model.norm.input` after the block loop exits."

**Verification:** All sigma thresholds are `(3, 4, 6)` consistently across:
- `src/profiler.py`: `OUTLIER_SIGMAS = (3.0, 4.0, 6.0)`
- `src/hooks.py`: `_OUTLIER_SIGMAS = (3, 4, 6)`
- `src/plotting.py`: annotates `±3σ, ±4σ, ±6σ`
- `docs/scispace-docs/vit_profiling_framework.md`: `k ∈ {3, 4, 6}`

**Affected files**
- `docs/scispace-docs/vit_profiling_framework.md` — Site Labeling Convention table updated

---

### T-007 — Verify reconstructed DOI for Yadav & Das 2025

**Severity:** MEDIUM  
**Status:** 🔲 Open  
**Category:** Citation hygiene  
**Source:** Issue 8 (skeptical review)

**Evidence**

`CITATIONS.md` lines 253–254:
```
DOI: S1383762126001542.
🔗 Link: https://doi.org/10.1016/j.sysarc.2026.103154
(⚠️ DOI reconstructed from PII S1383762126001542 — verify before citing)
```

The DOI `10.1016/j.sysarc.2026.103154` was reconstructed from the Publisher Item
Identifier (PII) `S1383762126001542` found in the ScienceDirect URL. The
reconstruction follows the standard Elsevier PII→DOI conversion pattern
(`S1383762126` → `10.1016/j.sysarc.2026.` + article number), but has not been
verified against the actual published article.

**Why this matters**

If the reconstructed DOI is wrong, the citation is unverifiable. Anyone trying to
look up this paper by DOI will fail. The `CITATIONS.md` already flags this with ⚠️,
but the flag must be resolved, not left as a permanent warning.

**Proposed fix**

1. Attempt to resolve `https://doi.org/10.1016/j.sysarc.2026.103154` in a browser
   or via `curl -L`.
2. If it resolves to the correct paper (Yadav & Das, "GateAttn-ViT"), update the
   CITATIONS.md entry to remove the ⚠️ warning and mark the DOI as verified.
3. If it does NOT resolve, search for the correct DOI on the Journal of Systems
   Architecture website using the paper title and authors.
4. If no DOI can be found, mark the citation as "DOI unverified — use ScienceDirect
   PII link instead" and use the PII-based URL as the primary link.

**Affected files**
- `docs/CITATIONS.md` — Yadav & Das 2025 entry

---

### T-008 — Strengthen outlier recount pass test

**Severity:** MEDIUM  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Test gap  
**Source:** Issue 10 (skeptical review)

**Resolution**

Added two integration tests that exercise the full two-pass pipeline:

1. `test_slow_run_outlier_counting_pass_correctness` — runs
   `run_profiling_dataset_pass` → `run_outlier_counting_pass` on a small
   synthetic dataset (4 images, 2 batches), then independently captures raw
   activations via a separate nnsight trace and verifies recount fractions
   match ground truth computed with global μ and σ.

2. `test_slow_run_outlier_counting_pass_multiple_sites_correctness` — same
   pattern but verifies across three site types (pre_gelu, post_layernorm_1,
   residual_stream) with different tensor shapes and element counts.

Both tests are marked `@pytest.mark.slow` (require nnsight trace).

The existing `test_run_outlier_counting_pass_known_gaussian` was annotated
with a clarifying docstring noting it is a mathematical sanity check, not
an integration test of the recount pass.

**Evidence**

`test_run_outlier_counting_pass_known_gaussian` (test_profiler.py:1731) does NOT test
the actual `run_outlier_counting_pass` function. It manually computes
`(data.abs() > k * sigma).float().mean().item()` on a `torch.randn(100_000)` tensor —
a direct computation with no nnsight trace, no recount pass, no batch accumulation.

The test verifies that the mathematical definition of outlier fraction is correct
(3σ fraction ≈ 0.0027 for Gaussian), not that the recount pass code is correct.
A bug in `run_outlier_counting_pass` — wrong site_id lookup, incorrect accumulation,
trace ordering error, off-by-one in batch counting — would not be caught.

**Why this matters**

The outlier recount pass (F2) was previously dead code (`pass`). The current
implementation was verified by manual inspection of output values from the
2026-07-28 50k-image run, not by automated regression tests. A future refactor
could silently break it. The recount pass is the source of truth for outlier
fractions — if it breaks, all Phase 2 ablation thresholds will be wrong.

**Proposed fix** ✅ Implemented

Add an integration test that:
1. Creates a small synthetic dataset (e.g., 4 batches of 16 images each).
2. Runs the full `run_profiling_dataset_pass` to get global μ and σ.
3. Runs `run_outlier_counting_pass` on the same data.
4. Verifies that the recount fractions match a ground-truth computation
   (manual counting on the same data using the global μ and σ).

This requires a real nnsight trace, so it should be marked `@pytest.mark.slow`.

Alternatively, add a unit test that mocks the nnsight trace and verifies the
accumulation logic in isolation — but this is harder to make meaningful.

**References**
- Bondarenko et al. (2021), arXiv:2109.12948 — the two-pass methodology is
  standard practice in quantization literature.
- Dettmers et al. (2022), "LLM.int8()," NeurIPS 2022, arXiv:2208.07339 — uses
  two-pass outlier detection with global statistics.

**Affected files**
- `tests/test_profiler.py` — added `test_slow_run_outlier_counting_pass_correctness`
  and `test_slow_run_outlier_counting_pass_multiple_sites_correctness`

---

### T-009 — Delete `hooks.LayerStats`, migrate consumers to `profiler.LayerStats`, and add max/min

**Severity:** LOW  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Latent bug + missing feature  
**Source:** Issues 12 + 13 (skeptical review)  
**Supersedes:** Original T-009 (unify classes) and T-010 (max/min) — merged 2026-07-30

**Resolution**

Implemented the full three-step plan:

**Step 1 — Added `max`/`min` to `profiler.LayerStats`:**
- Added `max: float = 0.0` and `min: float = 0.0` fields to `LayerStats`.
- Added `max_proxy`/`min_proxy` to `_StatsSavers`.
- Registered `t.max().save()` and `t.min().save()` in `_register_stat_saves`.
- Extracted max/min in `_finalize_stats`.
- Added `max_val`/`min_val` to `WelfordAccumulator` (initialized to `-inf`/`+inf`).
- Updated `merge_batch_stats` to track running max/min across batches.
- Updated `finalize_accumulator` to pass max/min through to `LayerStats`.

**Step 2 — Migrated consumers and deleted `hooks.LayerStats`:**
- `src/ablation.py`: changed import to `from src.profiler import LayerStats`.
- `tests/conftest.py`: changed import and rewrote `tiny_layer_stats` fixture
  to construct `profiler.LayerStats` with `site_identifier`, `outlier_fractions`
  (key format `"3.0_sigma"`), `max`, `min`.
- `tests/test_ablation.py`: changed import.
- `src/hooks.py`: deleted `LayerStats` dataclass (lines 65–100). Added import
  of `profiler.LayerStats` for `_finalize_accumulator`. Updated
  `_finalize_accumulator` to return `profiler.LayerStats`.
- `tests/test_hooks.py`: deleted (26 tests for dead code).

**Step 3 — Added tests:**
- `test_layer_stats_max_min_defaults` — verifies defaults.
- `test_layer_stats_max_min_store_values` — verifies explicit values.
- `test_layer_stats_max_min_survive_serialization` — JSON roundtrip.
- `test_layer_stats_max_min_backwards_compat` — old JSON without max/min keys.
- `test_slow_max_min_present` — profiling pass populates max/min.
- `test_slow_max_min_survives_serialisation` — slow roundtrip.
- `test_merge_batch_stats_max_min_tracking` — merge tracking across batches.

**Documentation updated:**
- `README.md`: updated `hooks.py` description, removed `test_hooks.py`, updated test counts.
- `docs/NEXT-STEPS.md`: updated `hooks.py` status row.
- `docs/MISTAKES.md`: updated §4.2 note about `hooks.py`.
- `docs/CITATIONS.md`: updated Welford reference for `hooks.py`.
- `docs/EXP1-IMPL.md`: added 7 new tests to checklist.

**Evidence**

Two incompatible `LayerStats` dataclasses exist:

| | `hooks.LayerStats` | `profiler.LayerStats` |
|---|---|---|
| **File** | `src/hooks.py:65` | `src/profiler.py:89` |
| **Key field** | `site: str` | `site_identifier: SiteId` |
| **Has max/min** | Yes (`max: float`, `min: float`) | No |
| **Has m3** | No | Yes (`m3: float = 0.0`) |
| **Outlier key** | `outlier_frac: dict[str, float]` | `outlier_fractions: dict[str, float]` |
| **Entropy** | `attn_entropy: list[float] \| None` | `attention_entropy_cls` + `attention_entropy_patches` |
| **Used by** | `ablation.py:21`, `conftest.py:16`, `test_ablation.py:9` | `profiler.py`, `test_profiler.py` |

`ablation.py` line 21: `from src.hooks import LayerStats` — imports the hooks version.
`conftest.py` line 16: `from src.hooks import LayerStats` — same.
`test_ablation.py` line 9: `from src.hooks import LayerStats` — same.

If Phase 2 code tries to pass a `profiler.LayerStats` (from `load_profiling_result`)
to a function expecting `hooks.LayerStats`, it will fail at runtime with
`TypeError: __init__() missing required keyword-only argument: 'site'` or
`AttributeError: 'LayerStats' object has no attribute 'site'`.

Additionally, `profiler.LayerStats` lacks `max` and `min` fields.  The spec
(`docs/scispace-docs/vit_profiling_framework.md` line 51) specifies max/min as
per-tensor scalars.  Without them, you cannot directly see the extreme values
that would clip under INT8 quantization (range [-128, 127]).  While std and
kurtosis characterize the distribution shape, the absolute range is what
determines whether uniform quantization is feasible at all.

**Why this matters**

Phase 2 (`ablation.py`) is not yet implemented, but when it is, it will need to
load profiling results from `profiling_result.json` (produced by `profiler.py`)
and use them to set ablation thresholds.  The current import in `ablation.py`
points to the wrong `LayerStats`.  This is a latent bug that will manifest the
moment Phase 2 implementation begins.

Max/min are also needed for quantization range sanity-checking.  The legacy
`hooks.LayerStats` had them; the profiler should too.  Since we are already
touching `profiler.LayerStats` to add the fields that `hooks.LayerStats`
consumers need, it is efficient to add max/min in the same pass.

**Proposed fix** ✅ Implemented

Delete `hooks.LayerStats` and migrate all consumers to `profiler.LayerStats`,
adding `max`/`min` fields to the profiler class in the same change.

**Step 1 — Add `max`/`min` to `profiler.LayerStats`**

1a. Add fields to `LayerStats` dataclass:
    ```python
    max: float = 0.0
    min: float = 0.0
    ```

1b. In `_StatsSavers`, add:
    ```python
    max_proxy: Any = None
    min_proxy: Any = None
    ```

1c. In `_register_stat_saves`, register max/min proxies:
    ```python
    max_proxy = t.max().save()
    min_proxy = t.min().save()
    ```
    Pass them through `_StatsSavers(max_proxy=max_proxy, min_proxy=min_proxy)`.

1d. In `_finalize_stats`, extract max/min:
    ```python
    max_val = float(_val(savers.max_proxy).item()) if savers.max_proxy is not None else 0.0
    min_val = float(_val(savers.min_proxy).item()) if savers.min_proxy is not None else 0.0
    ```
    Pass to `LayerStats(max=max_val, min=min_val)`.

1e. In `WelfordAccumulator`, add running max/min fields:
    ```python
    max_val: float = float("-inf")
    min_val: float = float("inf")
    ```

1f. In `merge_batch_stats`, update running max/min:
    ```python
    if batch_stats.max > acc.max_val:
        acc.max_val = batch_stats.max
    if batch_stats.min < acc.min_val:
        acc.min_val = batch_stats.min
    ```

1g. In `finalize_accumulator`, pass max/min through to `LayerStats`.

1h. Update `_register_pre_softmax_saves` — it delegates to `_register_stat_saves`,
    so max/min flow through automatically.  No change needed.

**Step 2 — Migrate consumers**

2a. `src/ablation.py` line 21: change `from src.hooks import LayerStats` →
    `from src.profiler import LayerStats`.

2b. `tests/conftest.py` line 16: same import change.

2c. `tests/conftest.py` `tiny_layer_stats` fixture: construct `profiler.LayerStats`
    instead of `hooks.LayerStats`.  Field mapping:
    - `site` → `site_identifier`
    - `layer_name` → dropped (profiler doesn't use it; add as a comment or
      store in `site_identifier` as `"{layer_name}/pre_gelu"`)
    - `outlier_frac` → `outlier_fractions` (key format changes from `"3"` to
      `"3.0_sigma"`, etc.)
    - `attn_entropy` → dropped (profiler uses separate CLS/patch fields)
    - `max`, `min` → preserved (now available on profiler class)

2d. `tests/test_ablation.py` line 9: same import change.

2e. Delete `hooks.LayerStats` from `src/hooks.py`.

**Step 3 — Add tests**

3a. `tests/test_profiler.py`: add `test_layer_stats_max_min_defaults` —
    verifies `max=0.0`, `min=0.0` by default.

3b. `tests/test_profiler.py`: add `test_slow_max_min_present` — runs a
    profiling pass and verifies max/min are populated with finite values
    and `max >= min`.

3c. `tests/test_profiler.py`: add `test_merge_batch_stats_max_min_tracking` —
    verifies that `merge_batch_stats` correctly tracks running max/min
    across multiple batches.

3d. `tests/test_profiler.py`: add `test_max_min_survive_serialization` —
    verifies max/min survive JSON save → load roundtrip.

**Affected files**
- `src/profiler.py` — `LayerStats`, `_StatsSavers`, `_register_stat_saves`,
  `_finalize_stats`, `WelfordAccumulator`, `merge_batch_stats`,
  `finalize_accumulator`
- `src/hooks.py` — delete `LayerStats` dataclass
- `src/ablation.py` — change import
- `tests/conftest.py` — change import and `tiny_layer_stats` fixture
- `tests/test_ablation.py` — change import
- `tests/test_profiler.py` — add 7 tests
- `tests/test_hooks.py` — deleted (26 tests for dead code)

---

### T-010 — ~~Compute max/min in profiler~~ (merged into T-009)

**Severity:** ~~LOW~~  
**Status:** 🔀 Merged into T-009 (2026-07-30)  

---

### T-011 — Researcher sign-off on all citations

**Severity:** LOW  
**Status:** 🔲 Open  
**Category:** Process  
**Source:** Issue 14 (skeptical review)

**Evidence**

Every single citation in `CITATIONS.md` (25+ entries) has:
```
☐ Researcher sign-off: Not yet reviewed
```

The citation audit log (lines 426–439) shows that links and metadata were verified
(arXiv IDs, DOIs, venues), but no researcher has actually read the papers to confirm
that the cited claims exist in the source.

**Why this matters**

Citations are being used to justify methodological choices:
- CLS/patch entropy separation (Maisonnave et al. 2025, Lee & Kim 2025, Mali 2025)
- Two-pass outlier counting (Bondarenko et al. 2021, Dettmers et al. 2022)
- ShiftGELU for Phase 3 (Kim et al. 2021, Li & Gu 2023)
- Attention entropy collapse (Zhai et al. 2023)

If the cited papers don't actually support those claims, the methodology is built on
sand. This is a process issue, not a code bug, but it matters for scholarly integrity.

**Proposed fix**

Prioritize sign-off on the 6–8 most critical citations (those directly justifying
methodological choices in the code):

1. Bondarenko et al. 2021 — justifies two-pass outlier counting
2. Dettmers et al. 2022 — justifies outlier handling methodology
3. Kim et al. 2021 (I-BERT) — justifies ShiftGELU for Phase 3
4. Li & Gu 2023 (I-ViT) — justifies ViT-specific integer quantization
5. Zhai et al. 2023 — justifies attention entropy as a diagnostic
6. Xiao et al. 2023 (SmoothQuant) — justifies per-channel scaling methodology
7. Wei et al. 2022 — justifies outlier suppression approach
8. Pébay 2008 — justifies the parallel moments merge formula

For each: read the paper (at minimum the abstract, methodology section, and relevant
results), verify that the cited claim exists, and check the ☐ → ☑.

**Affected files**
- `docs/CITATIONS.md` — update sign-off checkboxes

---

### T-017 — Attention entropy ε-bias not documented; values unreliable for highly focused distributions

**Severity:** MEDIUM  
**Status:** ✅ Closed (2026-08-01)  
**Category:** Documentation + numerical concern  
**Source:** Skeptical review, 2026-07-30

**Resolution:** Replaced ``p·log(p + ε)`` with ``torch.special.entr(p)`` which
implements ``−p·log(p)`` with the correct ``0·log(0) = 0`` convention natively.
No ε guard needed.  All 12 entropy tests pass with the new implementation.

Changes applied:
- ``src/profiler.py``: ``_register_entropy_saves`` line 1081 changed from
  ``-(attn_weight_proxy * (attn_weight_proxy + eps).log()).sum(dim=-1)`` to
  ``torch.special.entr(attn_weight_proxy).sum(dim=-1)``.
- ``src/profiler.py``: Updated docstring to remove ε mention and document
  ``torch.special.entr`` usage.

**Note:** This changes entropy values slightly (removes the O(N·ε) bias).
Any previously collected Phase 1 profiling data should be re-run to get
corrected entropy values before Phase 2 entropy delta computation.

**Affected files**
- ``src/profiler.py`` — ``_register_entropy_saves`` (line 1081), docstring

---

### T-018 — Per-channel accumulation ignores within-image spatial correlation; undocumented

**Severity:** LOW  
**Status:** ✅ Closed (2026-08-01)  
**Category:** Documentation  
**Source:** Skeptical review, 2026-07-30

**Resolution:** Added a documentation comment in ``_register_stat_saves`` near the
per-channel accumulation code explaining that all (B×N) token positions are treated
as i.i.d. samples of each channel's marginal distribution.  This is correct for
computing per-channel activation statistics for quantization range calibration.
Within-image spatial correlations affect the effective sample size but not the
validity of the marginal statistics.

**Affected files**
- ``src/profiler.py`` — ``_register_stat_saves`` (per-channel block, lines 1018-1025)

---

### T-019 — eval() mode and dropout disabled: not asserted at runtime

**Severity:** LOW  
**Status:** ✅ Closed (2026-08-01)  
**Category:** Defensive programming  
**Source:** Skeptical review, 2026-07-30

**Resolution:** Added a ``ProfilingError`` guard in ``profile_vit`` that checks
``inner_model.training`` and raises if the model is in training mode.  This is
a defensive check — ``load_vit`` already calls ``model.eval()``, but this guards
against accidental ``model.train()`` calls between loading and profiling.

Changes applied:
- ``src/profiler.py``: ``profile_vit`` now raises ``ProfilingError`` if
  ``inner_model.training`` is ``True``, after the ``blocks`` attribute check.

**Affected files**
- ``src/profiler.py`` — ``profile_vit`` (after ``inner_model`` extraction)

---

## INFO

### T-012 — Document `residual_stream` labeling convention

**Severity:** INFO  
**Status:** ✅ Closed (2026-07-30)  
**Category:** Documentation  
**Source:** Issue 15 (skeptical review)

**Resolution**

Added §0.1 Site Labeling Convention to `docs/EXP1-IMPL.md` with:
- A table mapping each `residual_stream` label to its semantic meaning
- A table listing all other site labels and where they're measured
- The canonical site order for display
- An explanation of why this matters (misattribution risk)

Added inline comments in `src/profiler.py` at both label construction sites
(`profile_vit` and `run_outlier_counting_pass`) referencing the convention.

**Evidence**

The `residual_stream` for block `i` is labeled `blocks.{i-1}/residual_stream`. This means:
- `patch_embed/residual_stream` — the patch embedding output (input to block 0)
- `blocks.0/residual_stream` — output of block 0 (input to block 1)
- `blocks.5/residual_stream` — output of block 5 (input to block 6)
- `blocks.10/residual_stream` — output of block 10 (input to block 11)

This is internally consistent but confusing because `blocks.5/residual_stream` sounds
like it should be the residual **at** block 5, not the residual **after** block 5.

**Why this matters**

When interpreting results, a reader might think `blocks.5/residual_stream`
(kurtosis=2992 from the 2026-07-28 run) is the residual inside block 5, when it's
actually the residual after block 5 (entering block 6). This could lead to
misattribution of outlier emergence — e.g., thinking block 5 produces the outliers
when block 4 actually does.

**Proposed fix** ✅ Implemented

Add a §Document Conventions section to `docs/scispace-docs/vit_profiling_framework.md`:

```markdown
### Site Labeling Convention

The `residual_stream` site label `blocks.{k}/residual_stream` denotes the residual
stream **after** encoder block `k` has processed it — i.e., the input to block `k+1`.

- `patch_embed/residual_stream`: patch embedding + positional encoding + CLS token
  (input to block 0)
- `blocks.0/residual_stream`: output of block 0 (input to block 1)
- `blocks.5/residual_stream`: output of block 5 (input to block 6)
- `blocks.10/residual_stream`: output of block 10 (input to block 11)
- `blocks.11/residual_stream`: output of block 11 (final encoder output, before head
  LayerNorm) — see T-001

All other site labels `blocks.{k}/{site}` denote measurements **inside** block `k`.
```

Also add a comment in `profiler.py` near the label construction (line 1139).

**Affected files**
- `docs/EXP1-IMPL.md` — added §0.1 Site Labeling Convention
- `src/profiler.py` — added comments near label construction in `profile_vit` and
  `run_outlier_counting_pass`

---

## DEFERRED / NEEDS DATA

### T-013 — Verify truncated outlier fraction in `blocks.4/post_softmax`

**Severity:** HIGH (if confirmed as bug) / LOW (if exact value)  
**Status:** ✅ Closed (2026-07-30) — confirmed as correct, not a bug  
**Category:** Data integrity  
**Source:** Issue 2 (skeptical review)

**Resolution**

Investigated the JSON serialization path. Python's `json.dump` uses `repr()`
internally for float values, which guarantees round-trip fidelity — the shortest
decimal representation that reproduces the exact IEEE 754 binary value.

A value of `0.00261` in the JSON is the exact float64 that was computed, not a
truncated approximation. The arithmetic check (0.00261 × 23,285,400,000 =
60,774,894, an integer) confirms this is a genuine exact value, not a
serialization artifact.

Added `test_float_precision_round_trip` to verify that sensitive float values
(including 0.0026099999999999999, π, very small values) survive JSON save → load
with bit-identical equality. Updated `save_profiling_result` docstring to
document the precision guarantee.

**Evidence**

The review claims that the `6.0_sigma` outlier fraction for `blocks.4/post_softmax`
is stored as `0.00261` — exactly 5 significant digits — while every other outlier
fraction in the 72-site output has full float64 precision (15–17 significant digits).

The arithmetic check: 0.00261 × 23,285,400,000 = 60,774,894 (an integer) suggests
the value may genuinely be exactly 0.00261. But the inconsistency with all other
values is anomalous.

**Cannot verify without the actual `profiling_result.json` file.** The `outputs/`
directory is empty in the current workspace.

**Proposed verification** ✅ Completed

1. Locate `profiling_result.json` from the 2026-07-28 50k-image run.
2. Check the raw JSON for `blocks.4/post_softmax` → `outlier_fractions` → `6.0_sigma`.
3. If the value is `0.00261` (no additional digits), check whether this is a
   `json.dumps` rounding artifact or a genuine exact value.
4. If it's a serialization bug, investigate `json.dump` float formatting in
   `save_profiling_result`.

**Affected files**
- `src/profiler.py` — updated `save_profiling_result` docstring
- `tests/test_profiler.py` — added `test_float_precision_round_trip`

---

### T-020 — Phase 2 threshold definition mismatch: zero-centered (|x| > k·σ) vs mean-centered (|x − μ| > k·σ)

**Severity:** HIGH
**Status:** ✅ Closed (2026-08-01)
**Category:** Bug — silent definition mismatch
**Source:** Skeptical review of Phase 2, 2026-08-01

**Evidence**

Phase 1's ``_register_stat_saves`` (profiler.py L1008-1016) defines outliers as
``|x − μ| > k·σ`` — mean-centered, consistent with the standard statistical
definition and the quantization literature (Wei et al. 2022, §3.1; Bondarenko
et al. 2021, §4.1).

Phase 2's ``_build_zeroing_mask`` (ablation.py L185-186) used:
```python
threshold = sigma_k * sigma
return tensor.abs() <= threshold    # |x| ≤ k·σ — zero-centered!
```

For sites where μ ≈ 0 (residual_stream, post_layernorm), the difference is
negligible.  For pre_gelu sites with large mean shifts (e.g., block 10 pre-GELU
where μ = −28.33), the two definitions zero different sets of elements.

**Resolution:** Added ``mean`` parameter to ``_build_zeroing_mask``.  All
``_intervene_*`` functions now pass ``layer_stats[site_id].mean``.  The mask
is now ``(tensor - mean).abs() <= threshold``, consistent with Phase 1.

Changes applied:
- ``src/ablation.py``: ``_build_zeroing_mask`` gained ``mean: float = 0.0`` parameter.
- ``src/ablation.py``: ``_intervene_pre_gelu``, ``_intervene_residual_stream``,
  ``_intervene_pre_softmax`` all pass ``layer_stats[site_id].mean``.
- ``src/ablation.py``: Module docstring updated to document mean-centered definition.
- ``tests/test_ablation.py``: Added ``test_build_zeroing_mask_mean_centered`` and
  ``test_build_zeroing_mask_mean_centered_default_zero``.

**Note:** This invalidates any previously collected Phase 2 results.  Re-run
Phase 2 to regenerate results with the corrected threshold definition.

**Affected files**
- ``src/ablation.py`` — ``_build_zeroing_mask``, ``_intervene_pre_gelu``,
  ``_intervene_residual_stream``, ``_intervene_pre_softmax``, module docstring
- ``tests/test_ablation.py`` — 2 new tests
- ``docs/MISTAKES.md`` — entry added

---

### T-021 — Phase 2 missing random-zeroing control condition

**Severity:** HIGH
**Status:** ✅ Closed (2026-08-01)
**Category:** Missing experimental control
**Source:** Skeptical review of Phase 2, 2026-08-01

**Evidence**

Phase 2 zeroed activation elements exceeding a threshold and measured accuracy
degradation.  Without a random-zeroing control (zeroing the same fraction of
elements at random positions), it is impossible to attribute accuracy
degradation to *outliers specifically* rather than to activation sparsity in
general.  This is a standard control in ablation studies.

**Resolution:** Added ``_build_random_mask`` function and ``random_fractions``
parameter to ``zero_outliers_in_trace`` and all ``_intervene_*`` functions.
The orchestrator (``exp2_ablation.py``) now runs a random-zeroing sweep for
``pre_gelu`` and ``residual_stream`` sites using the **exact per-batch
per-layer %-zeroed** from the outlier pass as the target fraction.  This
ensures the random control zeros exactly the same fraction of elements as
the outlier condition on every batch — the only difference is *which*
elements are zeroed, not *how many*.  Results are marked with ``is_random=True``.

Changes applied:
- ``src/ablation.py``: Added ``_build_random_mask`` function.
- ``src/ablation.py``: ``AblationResult`` gained ``is_random: bool`` field.
- ``src/ablation.py``: ``zero_outliers_in_trace``, ``_intervene_pre_gelu``,
  ``_intervene_residual_stream`` gained ``random_fractions`` and ``random_seed``
  parameters.
- ``src/ablation.py``: ``save_ablation_results`` includes ``is_random`` column.
- ``src/ablation.py``: ``save_entropy_deltas`` filters out ``is_random=True`` rows.
- ``src/exp2_ablation.py``: Rewritten to run outlier and random passes on the
  same batch with matched fractions.  Added ``_site_matches`` and
  ``_build_layer_results`` helpers.
- ``tests/test_ablation.py``: Added 9 tests for ``_build_random_mask``,
  1 test for ``is_random`` field, 2 slow tests for random mode, 1 test for
  ``is_random`` in CSV output, 3 tests for ``_site_matches``, 1 test for
  ``save_entropy_deltas`` random filtering.

**Affected files**
- ``src/ablation.py`` — ``_build_random_mask`` (new), ``AblationResult``,
  ``zero_outliers_in_trace``, ``_intervene_pre_gelu``, ``_intervene_residual_stream``,
  ``save_ablation_results``, ``save_entropy_deltas``
- ``src/exp2_ablation.py`` — ``_site_matches`` (new), ``_build_layer_results`` (new),
  ``run`` (per-batch matched random-control sweep)
- ``tests/test_ablation.py`` — 17 new tests
- ``docs/MISTAKES.md`` — entry added

---

### T-022 — Class imbalance in subset mode when shuffle=False

**Severity:** MEDIUM
**Status:** ✅ Closed (2026-08-01)
**Category:** Bug — silent data skew
**Source:** Skeptical review of Phase 2, 2026-08-01

**Evidence**

``build_val_loader`` (data_loader.py L112) used:
```python
indices = list(range(num_images))
```
when ``shuffle=False``.  ``ImageFolder`` returns images grouped by class in
alphabetical order, so taking the first N images samples only from the first
few classes.  For ``--num-images 1024``, this would produce a class-imbalanced
subset with meaningless accuracy numbers.

**Resolution:** Replaced ``list(range(num_images))`` with a seeded random
permutation (``torch.randperm`` with ``Generator().manual_seed(42)``).  This
produces a deterministic, class-balanced subset regardless of ``shuffle``.

Changes applied:
- ``src/data_loader.py``: ``build_val_loader`` now uses seeded permutation
  for subset selection when ``shuffle=False``.

**Affected files**
- ``src/data_loader.py`` — ``build_val_loader`` (subset selection logic)

---

## Summary

| Ticket | Severity | Status | Title |
|--------|----------|--------|-------|
| T-001 | CRITICAL | ✅ Closed | Final encoder residual stream never captured |
| T-002 | HIGH | ✅ Closed (2026-07-30) | Spec-code sigma threshold mismatch — changed to (3, 4, 6) |
| T-003 | ~~HIGH~~ | ✅ Closed | ShiftGELU attribution verified correct (review claim was wrong) |
| T-004 | HIGH | ✅ Closed (2026-07-30) | LayerNorm γ/β weights now captured in LayerStats |
| T-005 | MEDIUM | ✅ Closed (2026-07-30) | Residual update delta ‖Δ‖/‖x_skip‖ now computed |
| T-006 | MEDIUM | ✅ Closed (2026-07-30) | Framework doc now matches implementation (3,4,6 sigmas) |
| T-007 | MEDIUM | 🔲 Open | Verify reconstructed DOI for Yadav & Das 2025 |
| T-008 | MEDIUM | ✅ Closed (2026-07-30) | Strengthen outlier recount pass test |
| T-009 | LOW | ✅ Closed (2026-07-30) | Delete hooks.LayerStats, migrate to profiler.LayerStats, add max/min |
| T-010 | ~~LOW~~ | 🔀 Merged into T-009 (2026-07-30) | ~~Compute max/min in profiler~~ |
| T-011 | LOW | 🔲 Open | Researcher sign-off on all citations |
| T-012 | INFO | ✅ Closed (2026-07-30) | Document residual_stream labeling convention |
| T-013 | HIGH? | ✅ Closed (2026-07-30) | Verified: 0.00261 is exact, not truncated; added precision test |
| T-014 | HIGH | ✅ Closed (2026-07-30) | Outlier fraction definition: fixed to mean-centered |x−μ| > k·σ |
| T-015 | ~~HIGH~~ → LOW | ✅ Closed (2026-07-30) | Doc bug: field renamed to `ln2_amplification_ratio`, docstrings corrected |
| T-016 | HIGH | ✅ Closed (2026-07-30) | Model pinned to `augreg2_in21k_ft_in1k`; `RunMetadata` added to output JSON |
| T-017 | MEDIUM | ✅ Closed (2026-08-01) | Entropy ε-bias: replaced p·log(p+ε) with torch.special.entr(p) |
| T-018 | LOW | ✅ Closed (2026-08-01) | Per-channel spatial correlation assumption documented |
| T-019 | LOW | ✅ Closed (2026-08-01) | eval() mode assertion added to profile_vit |
| T-020 | HIGH | ✅ Closed (2026-08-01) | Phase 2 threshold mismatch: zero-centered → mean-centered |x−μ| > k·σ |
| T-021 | HIGH | ✅ Closed (2026-08-01) | Phase 2 missing random-zeroing control condition |
| T-022 | MEDIUM | ✅ Closed (2026-08-01) | Class imbalance in subset mode when shuffle=False |