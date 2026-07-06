# Unified Research Plan: Reconciling Warnings, Fixes, and Narrative

**Date:** 2026-07-06
**Status:** Strategic planning document. No implementation has begun.
**Depends on:** `docs/literature-survey.md`, `docs/course-correction-report.md`
**Audience:** The researcher (Dan) and any agent implementing the plan.

---

## 1. Where We Are Now

### Original research question

> Can the LLM.int8() mixed-precision decomposition framework (Dettmers et al. 2022),
> designed for LLMs, be applied to ViT-B/16 for edge deployment?

### What we have built

| Component | Status | File(s) |
|-----------|--------|---------|
| Outlier mapping pipeline (two-pass, exact stats) | **Complete** | `src/hooks.py`, `run_experiment1_mapping.py` |
| Quantization simulation (symmetric, 3 granularities) | **Complete** | `src/quantization.py` |
| Experiment 2: accuracy by granularity (4 configs) | **Script ready, not run** | `run_exp2_granularity.py` |
| Experiment 3: per-layer sensitivity | **Script ready, not run** | `run_experiment3_sensitivity.py` |
| Experiment 4: mixed-precision decomposition | **Not implemented** | — |
| Path B: SmoothQuant for blocks 9–10 | **Not implemented** | — |
| Data pipeline (ImageNet val, labeled + unlabeled) | **Complete** | `src/data_loader.py` |
| Test suite (43 fast + 1 slow) | **Complete** | `tests/` |

### Headline finding from Experiment 1

LLM.int8() column-wise routing works for 47/49 layers (routing fraction < 1%).
It fails catastrophically on `blocks.9.mlp.fc1` and `blocks.10.mlp.fc1`
(97–100% routing fraction). At block 10, σ = 11.68 — nearly 2× the INT8-safe
threshold of 6.0.

### The original July roadmap

- **Week 1–2:** Run Experiments 2 and 3.
- **Week 3:** Decision — Path A (validate + write up) vs. Path B (SmoothQuant).
- **Week 3–4 (Path A):** Run Experiment 4 (mixed-precision decomposition).
- **Week 3–4 (Path B):** Implement SmoothQuant for blocks 9–10 fc1.

### The problem with the course-correction report

The report treats six warnings as independent action items. In reality, they
interact — some are prerequisites for others, some would invalidate others if
implemented naively, and some address problems that don't exist in the current
experimental design. Implementing all six as written would explode scope without
a coherent narrative.

---

## 2. The Six Warnings: Consolidated Assessment

### Warning 1: Symmetric Quantization Is Wrong for Post-GELU Activations

| Disruption | Strengthening | Verdict |
|:----------:|:-------------:|---------|
| **Low** | **Medium** | **Do it. Correctness prerequisite, not a contribution.** |

Post-GELU activations (entering `mlp.fc2`, `head`) are predominantly positive
with a long tail. Symmetric quantization wastes half the bins on negative values
that almost never occur. PTQ4ViT (Yuan et al. 2022) already established this.

**Fix:** Add `quantize_per_tensor_asymmetric()` to `src/quantization.py` and a
`layer_type` parameter to `make_activation_quant_hook()`. Default behavior unchanged.

**Caveat:** The report's 0.5–1.5% accuracy estimate is optimistic — PTQ4ViT's
numbers include weight optimizations too. The isolated gain is probably smaller.

---

### Warning 2: Operation-Specific Quantizers for Sensitivity Rankings

| Disruption | Strengthening | Verdict |
|:----------:|:-------------:|---------|
| **Very Low** | **Low** | **Defer. The problem doesn't apply to current Experiment 3.** |

The report itself acknowledges (lines 208–214) that Experiment 3 only quantizes
*weights*, not activations. Weight quantization uses symmetric per-tensor, which
is standard. The operation-specific quantizer concern applies to *activation*
quantization, which Experiment 3 doesn't do. The proposed `--quantizer-aware`
flag would only matter if Experiment 3 were extended to quantize activations.

**Decision: Defer indefinitely.** Revisit only if Experiment 3 is extended.

---

### Warning 3: SmoothQuant May Not Transfer to Dense-Outlier Regimes

| Disruption | Strengthening | Verdict |
|:----------:|:-------------:|---------|
| **Medium (positive)** | **High** | **Critical. Gate Path B with a feasibility check.** |

SmoothQuant was designed for LLMs with sparse, extreme outliers (a few channels
with values of 60+). Your blocks 9–10 have dense, moderate outliers (34–60% of
values exceed 6.0). At α=0.8–0.9 (needed to bring σ from 11.68 below 6.0), the
weight-side amplification may make weights unquantizable.

**Fix:** One-day feasibility check using existing Experiment 1 data:
1. Compute per-channel smoothing factors for blocks 9–10 fc1 at α ∈ {0.5, 0.7, 0.9}.
2. Apply smoothing: `W_smoothed = W * diag(s)`.
3. Quantize `W_smoothed` with per-channel INT8, measure MSE.
4. If weight MSE > 10× activation MSE reduction → SmoothQuant shifted the problem
   → fall back to Path A.

**Interaction with Warning 4:** Warning 4 must be resolved *before* this check,
because if the explosion is token-driven, SmoothQuant is the wrong remedy regardless.

---

### Warning 4: The Block 9–10 Explosion May Be Token-Driven

| Disruption | Strengthening | Verdict |
|:----------:|:-------------:|---------|
| **Low** | **High** | **Highest priority. Most novel contribution potential.** |

Darcet et al. (2023) showed ViTs produce high-norm "artifact tokens" in background
regions. If your block 9–10 explosion is caused by a few extreme tokens rather
than a channel-level property, per-channel smoothing (SmoothQuant) won't fix it.

**Why this is the most interesting:** It bridges two literatures that haven't been
connected — artifact tokens (Darcet) and quantization outliers (Dettmers). Nobody
has asked whether ViT quantization outliers are token-driven or channel-driven.
The answer is novel regardless of which way it goes.

**Fix:** Add per-token statistics to Experiment 1 hooks. For blocks 8–11 fc1, record:
- Per-position outlier density and mean L2 norm
- Top-k token mass fraction (what fraction of total activation mass is carried by
  the top 5, 10, 20 tokens?)

**Diagnostic:** If top 5 tokens carry >50% of activation mass at blocks 9–10 fc1 →
token-driven → SmoothQuant is wrong, per-token scaling is right. If mass is evenly
distributed → channel-driven → SmoothQuant is appropriate.

**Caveats:**
- The binary framing may be too simple. A 2D histogram (tokens × channels) would
  be more informative.
- Separate CLS token from patch tokens — they have fundamentally different roles.

---

### Warning 5: The LLM.int8() Outlier Topology May Be a Scale Effect

| Disruption | Strengthening | Verdict |
|:----------:|:-------------:|---------|
| **Zero** | **Medium** | **Writing task. Zero code impact.** |

You're comparing an 86M-parameter ViT to 6.7B–175B LLMs. A reviewer will notice.
Scope claims to "ViT-B/16 at 86M parameters" rather than "ViTs generally."

**Fix:** Update framing in thesis and advisor doc. No code changes. If time permits,
run Experiment 1 on ViT-L/16 (307M params) for a second data point.

---

### Warning 6: Calibration/Evaluation Overlap

| Disruption | Strengthening | Verdict |
|:----------:|:-------------:|---------|
| **Low** | **Medium** | **Do before thesis-print run. Not urgent for Week 1.** |

Using the same images for statistics (Exp 1) and accuracy evaluation (Exps 2–4)
is methodologically weak. Bondarenko et al. (2021) and Nagel et al. (2021)
recommend separate splits.

**Fix:** Add `validation_split` parameter to `create_imagenet_val_loader()`. Use
80% for statistics, 20% for accuracy evaluation. Deterministic split (fixed seed).

**Why it's not urgent:** The main finding (blocks 9–10 at 97–100%) is far too large
for data overlap to explain. This primarily affects borderline layers (the `attn.qkv`
cluster at 0.39–0.52%), which are the least interesting layers in the study.

---

## 3. Three Deeper Questions the Report Missed

### 3.1 Is the 6.0 threshold appropriate for ViTs?

The 6.0 threshold was calibrated to LLM activation distributions. At block 10,
σ = 11.68, so 6.0 is ~0.5σ — not an "outlier" in any statistical sense. The
routing fraction saturates at 97–100% partly because the threshold is miscalibrated
for ViT's activation scale.

**This doesn't mean change the threshold.** Your experiment is explicitly "what
would LLM.int8() do?" But acknowledge this calibration mismatch in the thesis
and discuss what threshold *would* be appropriate.

### 3.2 Is column-wise routing even the right approach for ViTs?

LLM.int8() works for LLMs because outliers are sparse (0.1% of columns). At
97–100% routing, you're doing FP16 with a tiny INT8 garnish. Alternatives:
- **FQ-ViT's PTF:** per-channel scaling *before* LayerNorm
- **Outlier Suppression** (Wei et al. 2022): learned gamma factor before LayerNorm
- **Full FP16 for blocks 9–10, INT8 everywhere else:** simpler, no column routing

If SmoothQuant fails, the right response may not be "fall back to Path A" — it
may be "question whether column-wise routing is the right framework at all."

### 3.3 What is the actual contribution if 47/49 layers work but the 2 most important ones don't?

Three possible narratives:
1. **"LLM.int8() transfers to ViTs with two exceptions"** — modest, safe.
2. **"SmoothQuant can fix the exceptions"** — ambitious, risky.
3. **"ViT outliers are fundamentally different from LLM outliers (dense vs. sparse)"** — negative result, harder to publish but scientifically honest.

The report doesn't explicitly choose a narrative. The fixes should serve the
narrative, not the other way around.

---

## 4. The Unified Narrative

### The story this project can tell (regardless of which path succeeds)

> LLM.int8() mixed-precision decomposition was designed for language models where
> activation outliers are sparse, extreme, and channel-persistent. We systematically
> test whether this framework transfers to vision transformers. We find that it
> works for 47 of 49 linear layers but fails on two late-middle MLP layers where
> outliers are dense rather than sparse. We characterize *why* these two layers
> are different — whether the explosion is token-driven (artifact tokens from
> background patches) or channel-driven (accumulated inter-channel scale variation
> from residual connections) — and evaluate whether activation-weight equalization
> (SmoothQuant) can recover INT8 quantizability. Our characterization of the
> token-vs-channel topology of ViT quantization outliers is, to our knowledge,
> the first to bridge the artifact-token literature (Darcet et al. 2023) with
> the quantization-outlier literature (Dettmers et al. 2022).

### Why this narrative unifies the warnings

- **Warning 4 (token vs. channel)** becomes the *central scientific contribution*,
  not a side diagnostic.
- **Warning 3 (SmoothQuant feasibility)** becomes the *remedy evaluation* that
  follows from the characterization.
- **Warning 1 (asymmetric quantization)** becomes a *methodological prerequisite*
  that ensures accuracy numbers are credible.
- **Warning 5 (scale effect)** becomes a *limitation* that scopes the claims honestly.
- **Warning 6 (calibration split)** becomes *methodological hygiene* applied before
  the final run.
- **Warning 2 (operation-specific quantizers)** is *deferred* because it doesn't
  apply to the current experimental design.

### What changes from the original plan

| Original plan element | How it changes |
|-----------------------|----------------|
| Experiment 1 is just outlier mapping | Experiment 1 gains a **per-token analysis pass** (Warning 4). This is the novel contribution. |
| Experiment 2 runs as-is | Experiment 2 gains **asymmetric quantization** for post-GELU layers (Warning 1). Configs A and B use asymmetric for `mlp.fc2`/`head`. |
| Experiment 3 runs as-is | Experiment 3 is **unchanged** (Warning 2 deferred). It only quantizes weights. |
| Path B starts in Week 3 | Path B is **gated by two checks** in Week 1: (a) token-vs-channel analysis (Warning 4), (b) SmoothQuant feasibility (Warning 3). |
| Experiment 4 runs in Week 3–4 | Experiment 4 runs with **whatever remedy the checks support**: SmoothQuant if channel-driven + feasible, per-token scaling if token-driven, or pure per-layer FP16/INT8 split if neither works. |
| Thesis framing: "ViTs vs. LLMs" | Thesis framing: **"ViT-B/16 at 86M parameters"** (Warning 5), with token-vs-channel topology as the novel finding. |

---

## 5. Phased Implementation Plan

### Phase 0: Prerequisites (Day 1, ~3 hours)

**Goal:** Set up the infrastructure changes that all subsequent phases depend on.

| Step | File(s) | What to do | Effort |
|------|---------|------------|--------|
| 0.1 | `src/quantization.py` | Add `quantize_per_tensor_asymmetric()` function | 30 min |
| 0.2 | `src/quantization.py` | Add `layer_type` parameter to `make_activation_quant_hook()`, dispatch to asymmetric for `FeedForward_fc2`/`Other` | 30 min |
| 0.3 | `tests/test_quantization.py` | Add `test_quantize_per_tensor_asymmetric` (all-positive tensor, verify zero preserved) | 30 min |
| 0.4 | `src/model_utils.py` | Verify `classify_linear_layer()` correctly identifies all `mlp.fc2` and `head` layers | 15 min |
| 0.5 | Run tests | `pytest -m "not slow"` — all existing tests must pass | 15 min |

**Deliverable:** Asymmetric quantization is available but off by default. All tests pass.

---

### Phase 1: The Token-vs-Channel Analysis (Day 2–3, ~6 hours)

**Goal:** Answer the question: is the block 9–10 fc1 explosion caused by a few
extreme tokens or by many tokens with elevated values across many channels?
**This is the novel scientific contribution.**

| Step | File(s) | What to do | Effort |
|------|---------|------------|--------|
| 1.1 | `src/hooks.py` | Add `PerTokenOutlierProfile` dataclass with fields: `layer_name`, `per_position_outlier_density: list[float]`, `per_position_mean_norm: list[float]`, `topk_token_mass_fraction: dict[int, float]` | 30 min |
| 1.2 | `src/hooks.py` | Add `PerTokenOutlierAccumulator` class that records per-token statistics during forward passes. Only active for layers matching a configurable list (default: blocks 8–11 fc1). | 2 hours |
| 1.3 | `src/hooks.py` | Add `PerTokenCollector` class (parallel to `MomentCollector`/`OutlierStatsCollector`) that owns one `PerTokenOutlierAccumulator` per layer of interest | 1 hour |
| 1.4 | `run_experiment1_mapping.py` | Add optional third pass (or piggyback on Pass 2) that collects per-token statistics. Controlled by `--per-token` flag. | 1 hour |
| 1.5 | `src/visualizer.py` | Add visualization: per-position outlier density bar chart (197 positions, CLS token highlighted), top-k mass fraction curve | 1 hour |
| 1.6 | Run on 4096 images | Collect per-token statistics. Analyze: (a) top-5 token mass fraction at blocks 9–10 fc1, (b) CLS vs. patch token behavior, (c) 2D histogram (tokens × channels) for block 10 fc1 | 30 min (compute) + analysis |

**Deliverable:** A clear answer to "token-driven or channel-driven?" with visualizations.
This becomes a figure in the thesis.

**Decision point:** If token-driven → Path B (SmoothQuant) is the wrong remedy.
Proceed to Phase 2a (per-token scaling evaluation). If channel-driven → proceed
to Phase 2b (SmoothQuant feasibility check).

---

### Phase 2a: Token-Driven Path (Day 4–5, ~8 hours)

**Only if Phase 1 finds the explosion is token-driven.**

| Step | File(s) | What to do | Effort |
|------|---------|------------|--------|
| 2a.1 | `run_exp2_granularity.py` | Run Experiment 2 with asymmetric quantization (Phase 0). Compare Config C (per-token activations) vs. Config A (per-tensor) specifically at blocks 9–10. | 2 hours (compute) |
| 2a.2 | Analysis | If per-token scaling recovers significant accuracy at blocks 9–10 relative to per-tensor, the token-driven hypothesis is confirmed. Quantify the recovery. | 1 hour |
| 2a.3 | `run_experiment3_sensitivity.py` | Run Experiment 3 (unchanged — weight-only quantization). | 3 hours (compute) |
| 2a.4 | Experiment 4 design | Design mixed-precision scheme: per-token activation quantization for blocks 9–10 fc1, per-tensor everywhere else. Implement and evaluate. | 2 hours |

**Narrative:** "The block 9–10 explosion is caused by artifact tokens (Darcet et al.
2023). Per-token scaling mitigates this. We propose a hybrid scheme: per-token
quantization for the two affected layers, standard per-tensor quantization elsewhere."

---

### Phase 2b: Channel-Driven Path (Day 4–6, ~10 hours)

**Only if Phase 1 finds the explosion is channel-driven.**

| Step | File(s) | What to do | Effort |
|------|---------|------------|--------|
| 2b.1 | `src/smoothing.py` | Implement `compute_smoothing_factors()`, `apply_smoothing_to_linear()`, `reverse_smoothing()`, `make_smoothing_activation_hook()` as specified in the course-correction report (lines 290–377) | 2 hours |
| 2b.2 | `tests/test_smoothing.py` | Add tests: smoothing preserves matmul output, factors are bounded, reverse restores weights | 1 hour |
| 2b.3 | Feasibility script | Write a standalone script that: loads model, loads Experiment 1 per-channel stats from `outputs/exp1_outlier_maps/outlier_stats.json`, computes smoothing factors for blocks 9–10 fc1 at α ∈ {0.5, 0.7, 0.9}, applies smoothing, quantizes weights, measures MSE | 2 hours |
| 2b.4 | Run feasibility check | Execute the script. If weight MSE > 10× activation MSE reduction at all α → SmoothQuant fails → fall back to Phase 2c. | 30 min (compute) |
| 2b.5 | `run_exp2_granularity.py` | Run Experiment 2 with asymmetric quantization (Phase 0). | 2 hours (compute) |
| 2b.6 | `run_experiment3_sensitivity.py` | Run Experiment 3 (unchanged). | 3 hours (compute) |
| 2b.7 | Experiment 4 with SmoothQuant | Apply SmoothQuant to blocks 9–10 fc1, run mixed-precision decomposition. Compare against baseline (no SmoothQuant). | 3 hours |

**Narrative:** "The block 9–10 explosion is channel-driven, caused by accumulated
inter-channel scale variation from residual connections. SmoothQuant activation-weight
equalization can migrate this variation to the weights, making activations
INT8-quantizable. We evaluate whether this transfer is viable given the dense-outlier
regime (unlike the sparse-outlier LLM regime SmoothQuant was designed for)."

---

### Phase 2c: Fallback Path (Day 4–5, ~6 hours)

**If SmoothQuant fails the feasibility check OR the explosion is token-driven but
per-token scaling doesn't help enough.**

| Step | File(s) | What to do | Effort |
|------|---------|------------|--------|
| 2c.1 | `run_exp2_granularity.py` | Run Experiment 2 with asymmetric quantization. | 2 hours (compute) |
| 2c.2 | `run_experiment3_sensitivity.py` | Run Experiment 3 (unchanged). | 3 hours (compute) |
| 2c.3 | Experiment 4 | Implement simple per-layer mixed-precision: blocks 9–10 fc1 in FP16, all other layers INT8. Evaluate accuracy. | 2 hours |

**Narrative:** "Neither SmoothQuant nor per-token scaling can recover INT8
quantizability for blocks 9–10 fc1. We propose a simple per-layer mixed-precision
scheme: FP16 for the two affected layers, INT8 for the remaining 47. This achieves
[accuracy]% with only [X]% of compute in FP16. The characterization of *why* these
two layers resist INT8 quantization — [token-driven / channel-driven with dense
outliers] — is the primary contribution."

---

### Phase 3: Methodological Cleanup and Final Runs (Day 7–8, ~8 hours + compute)

**Goal:** Apply calibration split, re-run experiments on proper splits, prepare
thesis-quality numbers.

| Step | File(s) | What to do | Effort |
|------|---------|------------|--------|
| 3.1 | `src/data_loader.py` | Add `validation_split` parameter to `create_imagenet_val_loader()` (Warning 6). Deterministic split with fixed seed. | 1 hour |
| 3.2 | `run_experiment1_mapping.py` | Add `--calibration-split` flag (default 0.8). Re-run Experiment 1 on 80% calibration split with per-token analysis (Phase 1). | 3 hours (compute) |
| 3.3 | Experiment scripts | Update Experiments 2–4 to use `validation_split=0.2` for accuracy evaluation. | 30 min |
| 3.4 | Final runs | Run Experiments 2, 3, and 4 on the proper splits with all Phase 0–2 fixes applied. Use 50,000 images for thesis-print numbers. | 4+ hours (compute) |
| 3.5 | `docs/` | Update advisor touchpoint doc with final narrative, results, and limitations (Warning 5 framing). | 2 hours |

---

### Phase 4: Writeup (Day 9–10, ongoing)

| Step | What to do |
|------|------------|
| 4.1 | Write thesis section on token-vs-channel characterization (Phase 1 results) |
| 4.2 | Write thesis section on remedy evaluation (Phase 2 results) |
| 4.3 | Write limitations section (Warning 5: scale confound, 6.0 threshold miscalibration) |
| 4.4 | Generate final figures from Phase 3 data |

---

## 6. What I Need From You (Dan)

### Decisions to make now

1. **Do you agree with the unified narrative?** The plan above centers the
   token-vs-channel characterization as the novel contribution. If you'd rather
   center SmoothQuant or the per-layer mixed-precision scheme, the plan changes.
   **This is the most important decision.**

2. **Do you agree with deferring Warning 2?** I'm confident the operation-specific
   quantizer concern doesn't apply to Experiment 3 (weight-only quantization).
   But if you plan to extend Experiment 3 to quantize activations, we should
   revisit this.

3. **Do you agree with the Phase 1 → Phase 2 gating?** The plan runs the
   token-vs-channel analysis *before* committing to SmoothQuant. This means
   SmoothQuant implementation may never happen. Is that acceptable?

4. **ViT-L/16 as a second data point?** Warning 5 suggests running Experiment 1
   on ViT-L/16 (307M params) to test the scale-effect hypothesis. This is a
   significant compute investment (~3× the runtime of ViT-B/16). Is it worth it
   for the thesis, or should it be "future work"?

### Topics to become more familiar with

1. **Darcet et al. (2023), "Vision Transformers Need Registers"** — Read the full
   paper, not just the abstract. Understand how they characterize artifact tokens
   (which layers? which token positions? what norms?). This directly informs the
   Phase 1 per-token analysis.

2. **Sun et al. (2024), "Massive Activations in Large Language Models"** — Read
   the ViT section specifically. The abstract mentions ViTs but the course-correction
   report notes the phenomenon described (fixed-position bias terms) may differ
   from your block 9–10 explosion. You need to know which before citing it.

3. **Xiao et al. (2023), "SmoothQuant"** — Read Section 3 (methodology) carefully.
   Understand the α parameter, the smoothing factor formula, and the conditions
   under which the paper reports SmoothQuant degrades. This informs the Phase 2b
   feasibility check design.

4. **Wei et al. (2022), "Outlier Suppression"** — Skim as a fallback if SmoothQuant
   fails. Understand how the gamma factor is derived and whether it could apply
   to ViTs.

### Data/Compute questions

1. **Do you have the full 50,000 ImageNet validation images downloaded?** The
   Phase 3 thesis-print runs need them. If not, start the download now (it runs
   in the background).

2. **What is your GPU availability?** The plan assumes an RTX 3070 or equivalent
   with ~8 GB VRAM. If you have less, batch sizes need adjustment. If you have
   more (e.g., an A100 via university cluster), we can run larger experiments.

3. **Do you have Experiment 1 output data?** The SmoothQuant feasibility check
   (Phase 2b) needs per-channel activation statistics from Experiment 1. If
   `outputs/exp1_outlier_maps/outlier_stats.json` exists, we can start Phase 2b
   immediately after Phase 1. If not, run Experiment 1 first.

---

## 7. Anti-Scope-Creep Rules

These are hard rules to prevent the project from exploding. If you're tempted to
violate one, re-read this document first.

1. **No new experiments beyond the four planned.** Experiment 1 (outlier mapping),
   Experiment 2 (granularity), Experiment 3 (sensitivity), Experiment 4
   (decomposition). The per-token analysis is an *extension* of Experiment 1, not
   a new experiment. The SmoothQuant feasibility check is a *gate* for Path B,
   not a new experiment.

2. **No new model architectures.** ViT-B/16 only. ViT-L/16 is a "if time permits"
   bonus, not a requirement.

3. **No new quantization bit-widths.** INT8 only. No INT4, no FP8, no mixed
   8/4-bit schemes.

4. **No new datasets.** ImageNet-1K validation only. No COCO, no CIFAR, no
   out-of-distribution evaluation.

5. **No hardware deployment.** This is a software simulation project. Jetson Orin
   Nano deployment is future work.

6. **No new papers to survey unless they directly address a Phase 1/2 finding.**
   The literature survey is complete for the current scope.

7. **If a phase takes more than 1.5× its estimated time, stop and re-evaluate.**
   The estimates above are generous but not infinite. If the per-token analysis
   (Phase 1, estimated 6 hours) takes more than 9 hours, something is wrong —
   either the approach is too complex or the infrastructure has a bug.

---

## 8. Appendix: File-Level Change Summary

### New files to create

| File | Phase | Purpose |
|------|-------|---------|
| `src/smoothing.py` | 2b | SmoothQuant transformation functions |
| `tests/test_smoothing.py` | 2b | Tests for smoothing module |
| `scripts/smoothquant_feasibility.py` | 2b | Standalone feasibility check script |

### Existing files to modify

| File | Phase | Changes |
|------|-------|---------|
| `src/quantization.py` | 0 | Add `quantize_per_tensor_asymmetric()`, add `layer_type` param to `make_activation_quant_hook()` |
| `src/hooks.py` | 1 | Add `PerTokenOutlierProfile`, `PerTokenOutlierAccumulator`, `PerTokenCollector` |
| `src/data_loader.py` | 3 | Add `validation_split` parameter to `create_imagenet_val_loader()` |
| `src/visualizer.py` | 1 | Add per-token visualization functions |
| `run_experiment1_mapping.py` | 1, 3 | Add `--per-token` flag, add `--calibration-split` flag |
| `run_exp2_granularity.py` | 2 | Pass `layer_type` info to activation quant hooks |
| `tests/test_quantization.py` | 0 | Add `test_quantize_per_tensor_asymmetric` |

### Files that do NOT change

| File | Reason |
|------|--------|
| `run_experiment3_sensitivity.py` | Warning 2 deferred. Experiment 3 only quantizes weights. |
| `src/model_utils.py` | `classify_linear_layer()` already correct. |
| `regenerate_plots.py` | May need minor updates for new visualizations, but logic is unchanged. |
| All test files except `test_quantization.py` | Existing tests should continue to pass. |

---

## Summary: The One-Paragraph Plan

We extend Experiment 1 to characterize whether the block 9–10 fc1 activation
explosion is token-driven (caused by artifact tokens in background patches, per
Darcet et al. 2023) or channel-driven (caused by accumulated inter-channel scale
variation from residual connections). This characterization — bridging the
artifact-token and quantization-outlier literatures for the first time — is the
novel scientific contribution. Based on the answer, we either (a) evaluate
per-token scaling as a remedy (if token-driven), (b) gate SmoothQuant with a
feasibility check and evaluate it if viable (if channel-driven), or (c) fall back
to a simple per-layer FP16/INT8 split and document why neither remedy works.
Throughout, we fix the post-GELU asymmetric quantization bug (correctness
prerequisite), apply a proper calibration/evaluation split (methodological hygiene),
and scope all claims to ViT-B/16 at 86M parameters (honest framing). The result
is a coherent story about *why* two specific layers resist INT8 quantization and
what can be done about it.