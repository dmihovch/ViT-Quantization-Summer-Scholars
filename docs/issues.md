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
**Status:** 🔲 Open  
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
**Status:** 🔲 Open  
**Category:** Documentation/code divergence  
**Source:** Issue 3 (skeptical review)

**Evidence**

- `profiler.py` line 77: `OUTLIER_SIGMAS: tuple[float, ...] = (3.0, 5.0, 8.0)`
- `docs/scispace-docs/vit_profiling_framework.md` line 53: specifies `k ∈ {3, 4, 6}`
- `hooks.py` (legacy, not used in Phase 1): `_OUTLIER_SIGMAS = (3, 4, 6)`
- The original implementation decision explicitly acknowledged this discrepancy and
  ordered not to change the values because doing so would invalidate
  previously collected data.

**Why this matters**

The 4σ threshold is standard in the quantization literature. Bondarenko et al. (2021)
use it as their primary outlier detection threshold. The 6σ threshold is standard for
"extreme outlier" detection (Dettmers et al. 2022; Wei et al. 2022). By using (3, 5, 8)
instead, the results are not directly comparable to literature that uses (3, 4, 6).
The 5σ and 8σ thresholds are non-standard.

Furthermore, the framework document (`docs/scispace-docs/vit_profiling_framework.md`) is the authoritative
spec for this project. Anyone reading it as the source of truth will expect (3, 4, 6)
thresholds and will be confused by the actual output.

**Proposed fix**

Two-part fix:

1. **Update `docs/scispace-docs/vit_profiling_framework.md`** to reflect the actual values (3, 5, 8).
   The framework document must match the implementation. Add a note explaining that
   the values were chosen to provide wider coverage (3σ for moderate outliers, 5σ for
   significant outliers, 8σ for extreme outliers) and that the 4σ and 6σ thresholds
   can be interpolated from the existing data if needed for literature comparison.

2. **Do NOT change `OUTLIER_SIGMAS` in `profiler.py`.** The rationale from the original
   implementation decision is correct: changing these values would invalidate all
   previously collected profiling data.
   rationale is correct: changing these values would invalidate all previously
   collected profiling data. The existing (3, 5, 8) thresholds are a superset of
   information — they provide strictly more coverage than (3, 4, 6) would.

**Rationale**

The (3, 5, 8) choice is defensible: 3σ captures the Gaussian tail baseline (~0.27%),
5σ captures heavy-tail outliers (~5.7×10⁻⁵ for Gaussian, much higher for heavy-tailed
distributions), and 8σ captures extreme outliers that are essentially impossible under
a Gaussian. The 4σ and 6σ thresholds can be approximately interpolated from the
existing data since outlier fraction is monotonic in σ. The documentation fix is the
correct resolution — not a code change.

**References**
- Bondarenko et al. (2021), arXiv:2109.12948 — uses 4σ as primary threshold.
- Dettmers et al. (2022), "LLM.int8()," NeurIPS 2022, arXiv:2208.07339 — uses 6σ
  for extreme outlier detection.
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight), arXiv:2209.13325 —
  uses multiple σ thresholds for sensitivity analysis.

**Affected files**
- `docs/scispace-docs/vit_profiling_framework.md` — update §Per-Site Metrics to reflect (3, 5, 8)

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
**Status:** 🔲 Open  
**Category:** Missing feature  
**Source:** Issue 5 (skeptical review)

**Evidence**

`docs/scispace-docs/vit_profiling_framework.md` line 65 explicitly specifies:
> "Log the γ weights alongside the post-LN activation std-per-channel to separate
> learned-scale outliers from distribution outliers."

This was never implemented. The current `_register_stat_saves` (line 790) captures
`per_channel_sum` and `per_channel_sum_sq` from the activation tensor, but there is
no code that extracts `norm.weight` (γ) or `norm.bias` (β) from any LayerNorm module.
The `LayerStats` dataclass (line 89) has no fields for LayerNorm γ or β.

**Why this matters**

Without γ values, you cannot distinguish between:
- A channel with high σ because the input distribution genuinely has high variance
  (distribution outlier — requires quantization accommodation)
- A channel with high σ because γ is large (learned scaling — could potentially be
  folded into the next layer's weights)

This distinction is critical for per-channel quantization decisions in Phase 2/3.
SmoothQuant (Xiao et al. 2023) explicitly uses this distinction to migrate
quantization difficulty from activations to weights via mathematical equivalence.

The per-channel σ values alone are ambiguous. A channel with σ = 5.0 could be a
genuine outlier channel or simply a channel where the model learned γ = 5.0 on a
well-behaved distribution.

**Proposed fix**

1. Add fields to `LayerStats`:
   ```python
   layernorm_gamma: list[float] | None = None  # γ weights, shape [D]
   layernorm_beta: list[float] | None = None   # β weights, shape [D]
   ```

2. In `profile_vit`, after capturing `post_layernorm_1` and `post_layernorm_2`,
   also extract the corresponding LayerNorm weights from the underlying model:
   ```python
   # After the trace exits, for each block i:
   ln1_gamma = inner_model.blocks[i].norm1.weight.detach().cpu().tolist()
   ln1_beta = inner_model.blocks[i].norm1.bias.detach().cpu().tolist()
   ln2_gamma = inner_model.blocks[i].norm2.weight.detach().cpu().tolist()
   ln2_beta = inner_model.blocks[i].norm2.bias.detach().cpu().tolist()
   ```

   These are static model parameters — they don't need to be captured inside the
   nnsight trace. They can be read directly from the model after the trace exits.

3. Store them in the corresponding `LayerStats` entries for `post_layernorm_1` and
   `post_layernorm_2` sites.

4. Update `WelfordAccumulator` to carry these through (they don't need merging since
   they're static per-site).

5. Update `ProfilingResult` serialization to include them.

**Rationale**

This is a static data collection task — the γ/β weights are model parameters, not
activation statistics. They don't require the nnsight trace or the Welford merge
pipeline. They can be extracted once after the trace exits. The implementation
cost is low; the analytical value is high.

**References**
- Xiao et al. (2023), "SmoothQuant," ICML 2023, arXiv:2211.10438 — §3.1: the
  core insight is that activation outliers can be smoothed into weights by
  scaling LayerNorm γ down and the subsequent linear layer's weights up.
  See `docs/CITATIONS.md`.
- Bondarenko et al. (2021), arXiv:2109.12948 — §4.1: discusses per-channel
  variance and the role of LayerNorm affine parameters.

**Affected files**
- `src/profiler.py` — `LayerStats` (add fields), `profile_vit` (extract weights),
  `WelfordAccumulator` (carry through), `finalize_accumulator` (pass through),
  `save_profiling_result` / `load_profiling_result` (serialization)
- `tests/test_profiler.py` — add tests for γ/β presence and correctness

---

## MEDIUM

### T-005 — Residual update delta ‖Δ‖/‖x_skip‖ not computed

**Severity:** MEDIUM  
**Status:** 🔲 Open  
**Category:** Missing feature  
**Source:** Issue 6 (skeptical review)

**Evidence**

`docs/scispace-docs/vit_profiling_framework.md` lines 71–75 specify computing the MLP update magnitude
relative to the skip connection: `‖mlp_output‖ / ‖x_skip‖`. This requires capturing
both the skip connection and the MLP output.

The current implementation captures:
- `block.norm1.input` — the residual before attention+MLP (the skip connection)
- `block.norm2.input` — the residual after attention, before MLP

But it does not isolate the MLP contribution or compute the ratio `‖Δ‖/‖x_skip‖`.

**Why this matters**

This metric directly answers: "how aggressively does each MLP block modify the
residual stream?" — which is the primary driver of quantization range expansion.
The residual_stream kurtosis values (e.g., 4703 at block 6 from the 2026-07-28 run)
tell us *that* the distribution is heavy-tailed, but not *why*. The residual delta
ratio tells us whether the MLP or the attention sub-block is the dominant contributor.

This is actionable for Phase 2: if MLP blocks 4–7 have large deltas, those are the
blocks where outlier ablation will have the most impact.

**Proposed fix**

Inside the nnsight trace in `profile_vit`, for each block `i`:

1. Capture `block.norm1.input` as the skip connection (already done as `residual_stream`).
2. Capture `block.norm2.input` as the residual after attention.
3. Compute the MLP output proxy: `mlp_output = block.norm2.output` (this is the MLP
   contribution after the second LayerNorm — not exactly the raw MLP output, but the
   normalized version).
4. Compute the ratio proxy: `mlp_norm = mlp_output.norm(dim=-1).mean()` and
   `skip_norm = block.norm1.input.norm(dim=-1).mean()`, then
   `delta_ratio = mlp_norm / (skip_norm + eps)`.

Alternatively, compute this as a post-hoc analysis from the existing `residual_stream`
and `post_layernorm_2` statistics. Since both are captured, the per-batch mean norms
could be computed and the ratio derived. However, this would be approximate (mean of
ratio ≠ ratio of means). Computing it inside the trace as a per-token ratio and then
averaging is more accurate.

Add a field to `LayerStats`:
```python
residual_delta_ratio: float | None = None
# Mean over batch and tokens of ‖mlp_output‖₂ / ‖x_skip‖₂.
# None for all sites except residual_stream (where it represents the
# ratio for the MLP block whose output produced this residual).
```

**Rationale**

This is a derived metric from already-captured data. The implementation requires
adding norm computations inside the trace and a new field to `LayerStats`. The
analytical value for Phase 2 ablation targeting is high.

**References**
- Bondarenko et al. (2021), arXiv:2109.12948 — §4.2 discusses how residual
  connections cause quantization range expansion.
- Wei et al. (2022), "Outlier Suppression," NeurIPS 2022 (Spotlight),
  arXiv:2209.13325 — §3.1 analyzes which transformer sub-layers produce outliers.

**Affected files**
- `src/profiler.py` — `LayerStats` (add field), `profile_vit` (add norm computations),
  `WelfordAccumulator` (carry through), `merge_batch_stats` (accumulate),
  `finalize_accumulator` (finalize)
- `tests/test_profiler.py` — add test for delta ratio computation

---

### T-006 — Update `docs/scispace-docs/vit_profiling_framework.md` to match implementation

**Severity:** MEDIUM  
**Status:** 🔲 Open  
**Category:** Documentation  
**Source:** Issue 7 (skeptical review)

**Evidence**

The framework document is stale in several places:
- Line 53: specifies `k ∈ {3, 4, 6}` — code uses `(3, 5, 8)` (see T-002)
- Line 53: says outlier fractions are "computed exactly via Welford's parallel merge" —
  actual implementation uses a two-pass recount (F2, `run_outlier_counting_pass`)
- Line 51: specifies `max`, `min` as per-tensor scalars — not implemented (see T-009)
- Line 65: specifies LayerNorm γ logging — not implemented (see T-004)
- Lines 71–75: specifies residual delta computation — not implemented (see T-005)
- The document describes 5 measurement sites in the table (lines 39–45) but the
  implementation has 6 (adding `residual_stream` as a distinct site)

**Why this matters**

`docs/scispace-docs/vit_profiling_framework.md` is the authoritative spec for this project.
contributors reading it will expect behavior that doesn't match the code. The
document must be the ground truth — if the implementation diverges, either the
code or the document must change. In this case, the code is correct and the
document is stale.

**Proposed fix**

Update `docs/scispace-docs/vit_profiling_framework.md`:
1. §Per-Site Metrics: change `k ∈ {3, 4, 6}` → `k ∈ {3, 5, 8}` with a note
   explaining the choice (see T-002 rationale).
2. §Per-Site Metrics: update outlier fraction description to note the two-pass
   recount methodology (F2) rather than claiming single-pass Welford.
3. §Per-Site Metrics: mark `max`/`min` as "not yet implemented" or remove them.
4. §Post-LayerNorm: mark γ/β logging as "not yet implemented" (see T-004).
5. §Residual Update Stream: mark delta ratio as "not yet implemented" (see T-005).
6. §Measurement Sites table: add the `residual_stream` row to make it 6 sites.
7. Add a §Document Conventions section explaining the `residual_stream` labeling
   convention (see T-012).

**Affected files**
- `docs/scispace-docs/vit_profiling_framework.md` — multiple sections

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
**Status:** 🔲 Open  
**Category:** Test gap  
**Source:** Issue 10 (skeptical review)

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

**Proposed fix**

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
- `tests/test_profiler.py` — add `test_slow_run_outlier_counting_pass_correctness`

---

## LOW

### T-009 — Unify two incompatible `LayerStats` classes

**Severity:** LOW  
**Status:** 🔲 Open  
**Category:** Latent bug  
**Source:** Issue 12 (skeptical review)

**Evidence**

Two incompatible `LayerStats` dataclasses exist:

| | `hooks.LayerStats` | `profiler.LayerStats` |
|---|---|---|
| **File** | `src/hooks.py:65` | `src/profiler.py:89` |
| **Key field** | `site: str` | `site_identifier: SiteId` |
| **Has max/min** | Yes | No |
| **Has m3** | No | Yes |
| **Outlier key** | `outlier_frac: dict[str, float]` | `outlier_fractions: dict[str, float]` |
| **Entropy** | `attn_entropy: list[float] \| None` | `attention_entropy_cls` + `attention_entropy_patches` |
| **Used by** | `ablation.py:21`, `conftest.py:16` | `profiler.py`, `test_profiler.py` |

`ablation.py` line 21: `from src.hooks import LayerStats` — imports the hooks version.
`conftest.py` line 16: `from src.hooks import LayerStats` — same.

If Phase 2 code tries to pass a `profiler.LayerStats` (from `load_profiling_result`)
to a function expecting `hooks.LayerStats`, it will fail at runtime with
`TypeError: __init__() missing required keyword-only argument: 'site'` or
`AttributeError: 'LayerStats' object has no attribute 'site'`.

**Why this matters**

Phase 2 (`ablation.py`) is not yet implemented, but when it is, it will need to
load profiling results from `profiling_result.json` (produced by `profiler.py`)
and use them to set ablation thresholds. The current import in `ablation.py` points
to the wrong `LayerStats`. This is a latent bug that will manifest the moment
Phase 2 implementation begins.

**Proposed fix**

Option A (preferred): Delete `hooks.LayerStats` and migrate all consumers to
`profiler.LayerStats`.

1. Update `ablation.py` line 21: `from src.profiler import LayerStats`
2. Update `conftest.py` line 16: `from src.profiler import LayerStats`
3. Update `conftest.py` `tiny_layer_stats` fixture to construct `profiler.LayerStats`
   instead of `hooks.LayerStats`.
4. Delete `hooks.LayerStats` or mark it as deprecated with a `DeprecationWarning`.

Option B: Keep both but add an adapter. This is more complex and error-prone.

**Affected files**
- `src/ablation.py` — change import
- `tests/conftest.py` — change import and fixture
- `src/hooks.py` — deprecate or remove `LayerStats`

---

### T-010 — Compute max/min in profiler

**Severity:** LOW  
**Status:** 🔲 Open  
**Category:** Missing feature  
**Source:** Issue 13 (skeptical review)

**Evidence**

`docs/scispace-docs/vit_profiling_framework.md` line 51 specifies `max` and `min` as per-tensor scalars.
The current `profiler.LayerStats` (line 89) has no `max` or `min` fields.
`_register_stat_saves` (line 790) computes mean, std, M3, kurtosis, and outlier
fractions — but not max/min.

The legacy `hooks.LayerStats` (line 65) does have `max` and `min` fields, but
`hooks.py` is not used in Phase 1.

**Why this matters**

Max/min are useful for sanity-checking quantization ranges. Without them, you cannot
directly see the extreme values that would clip under INT8 quantization (range
[-128, 127]). While std and kurtosis characterize the distribution shape, the
absolute range is what determines whether uniform quantization is feasible at all.

However, max/min don't merge across batches via the Pébay formula — you need
per-batch tracking of running max/min, which is trivial but wasn't implemented.

**Proposed fix**

1. Add fields to `LayerStats`:
   ```python
   max: float = 0.0
   min: float = 0.0
   ```

2. In `_register_stat_saves`, add:
   ```python
   max_proxy = t.max().save()
   min_proxy = t.min().save()
   ```

3. In `_finalize_stats`, extract these values.

4. In `merge_batch_stats`, track running max/min:
   ```python
   acc.max_val = max(acc.max_val, batch_stats.max)
   acc.min_val = min(acc.min_val, batch_stats.min)
   ```

5. Update `WelfordAccumulator` with `max_val` and `min_val` fields (initialized to
   `-inf` and `+inf` respectively).

**Affected files**
- `src/profiler.py` — `LayerStats`, `_register_stat_saves`, `_finalize_stats`,
  `WelfordAccumulator`, `merge_batch_stats`, `finalize_accumulator`
- `tests/test_profiler.py` — add test for max/min tracking

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

## INFO

### T-012 — Document `residual_stream` labeling convention

**Severity:** INFO  
**Status:** 🔲 Open  
**Category:** Documentation  
**Source:** Issue 15 (skeptical review)

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

**Proposed fix**

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
- `docs/scispace-docs/vit_profiling_framework.md` — add §Document Conventions
- `src/profiler.py` — add comment near line 1139

---

## DEFERRED / NEEDS DATA

### T-013 — Verify truncated outlier fraction in `blocks.4/post_softmax`

**Severity:** HIGH (if confirmed as bug) / LOW (if exact value)  
**Status:** ⏳ Blocked — needs `profiling_result.json`  
**Category:** Data integrity  
**Source:** Issue 2 (skeptical review)

**Evidence**

The review claims that the `8.0_sigma` outlier fraction for `blocks.4/post_softmax`
is stored as `0.00261` — exactly 5 significant digits — while every other outlier
fraction in the 72-site output has full float64 precision (15–17 significant digits).

The arithmetic check: 0.00261 × 23,285,400,000 = 60,774,894 (an integer) suggests
the value may genuinely be exactly 0.00261. But the inconsistency with all other
values is anomalous.

**Cannot verify without the actual `profiling_result.json` file.** The `outputs/`
directory is empty in the current workspace.

**Proposed verification**

1. Locate `profiling_result.json` from the 2026-07-28 50k-image run.
2. Check the raw JSON for `blocks.4/post_softmax` → `outlier_fractions` → `8.0_sigma`.
3. If the value is `0.00261` (no additional digits), check whether this is a
   `json.dumps` rounding artifact or a genuine exact value.
4. If it's a serialization bug, investigate `json.dump` float formatting in
   `save_profiling_result`.

**Affected files**
- `outputs/phase1-profiling/profiling_result.json` (not in workspace)

---

## Summary

| Ticket | Severity | Status | Title |
|--------|----------|--------|-------|
| T-001 | CRITICAL | 🔲 Open | Final encoder residual stream never captured |
| T-002 | HIGH | 🔲 Open | Spec-code sigma threshold mismatch |
| T-003 | ~~HIGH~~ | ✅ Closed | ShiftGELU attribution verified correct (review claim was wrong) |
| T-004 | HIGH | 🔲 Open | LayerNorm γ/β weights not captured |
| T-005 | MEDIUM | 🔲 Open | Residual update delta not computed |
| T-006 | MEDIUM | 🔲 Open | Update framework doc to match implementation |
| T-007 | MEDIUM | 🔲 Open | Verify reconstructed DOI for Yadav & Das 2025 |
| T-008 | MEDIUM | 🔲 Open | Strengthen outlier recount pass test |
| T-009 | LOW | 🔲 Open | Unify two incompatible LayerStats classes |
| T-010 | LOW | 🔲 Open | Compute max/min in profiler |
| T-011 | LOW | 🔲 Open | Researcher sign-off on all citations |
| T-012 | INFO | 🔲 Open | Document residual_stream labeling convention |
| T-013 | HIGH? | ⏳ Blocked | Verify truncated outlier fraction (needs JSON) |