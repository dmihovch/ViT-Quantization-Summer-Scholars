# July Roadmap: ViT-B/16 PTQ for Edge Deployment

**Date:** June 30, 2026
**Deadline:** August 6, 2026
**Days remaining:** 37


## 1. Where we are on June 30

### What is done

Experiment 1 (the outlier map) is functionally complete. The pipeline runs
end-to-end: two-pass exact per-channel statistics over 49 linear layers of
ViT-B/16, producing per-layer routing fractions, outlier densities, channel
persistence, and max magnitudes. A minimal, well-tested codebase supports it
(44 tests pass, 43 fast plus 1 slow integration test). The routing-policy
table (2 FP16 / 3 LLM.int8 / 44 INT8) is directly readable from the measured
data and is backed by charts and a JSON artifact.

### What is not done

- Experiments 2 (granularity), 3 (per-layer sensitivity), and 4
  (decomposition) exist as written specs in `docs/vit-agent-grounding.md` but
  have no driver scripts, no `quantization.py` module, and no accuracy numbers
  of any kind.
- The calibration/evaluation data-split question is unresolved (see Section 4
  below).
- The 50,000-image thesis-print re-run of Experiment 1 has not been done (the
  current numbers come from a 5,000-image validation run, Section 3.1 below).
- Hardware (Jetson Orin Nano) has not been touched; no EDP numbers exist.

### What the advisor touchpoint doc captures

`docs/advisor-touchpoint-guide.md` is current as of today. It contains all
Experiment 1 findings (at 5,000 images), a full self-review (Section 9), a
list of items to flag for Nazim (Section 0.1), and a set of open questions
(Section 7) that require advisor input before July work can commit to a
direction.


## 2. The core strategic question for July

The project has two possible shapes, depending on one decision: **validate the
outlier map, or fix the problem it found.**

**Path A (validate the outlier map):** Run Experiments 3 and 4 to confirm
that the outlier topology predicts quantization accuracy drop. The deliverable
is a verified measurement + routing policy for one model. The risk is that
this reads as a diagnostic, not a novel method; the reward is that it is
concrete, completable, and unlikely to produce a negative result (Experiment 1
already says blocks 9-10 fc1 should dominate accuracy loss; Experiment 3 will
almost certainly confirm that). Time estimate: ~3 weeks for both experiments
plus the 50K re-run.

**Path B (fix the root cause):** Add a smoothing or equalization step that
reduces the fc1 scale explosion in blocks 9-10, aiming to bring those two
layers back into the INT8 regime (or at least into LLM.int8(), reducing the
FP16 penalty). The deliverable changes from a routing table to a technique
plus evidence that it works. The risk is that the smoothing may not work, or
may work but with an accuracy cost that undoes the gain, or may take longer to
implement than the remaining time allows. The reward is that a positive result
here is a stronger contribution than Path A.

**The decision does not need to be all-or-nothing on July 1.** The first two
weeks of July overlap between both paths: Experiment 3 (per-layer sensitivity)
needs to run regardless, because it is the cross-validation that closes the
causal loop and tells us whether the outlier map actually predicts accuracy.
Week 3 onward is where the paths diverge.

This roadmap is therefore written as a **two-phase plan**: Phase 1 (July 1-14)
is the convergence zone that both paths share, with a go/no-go checkpoint at
the end. Phase 2 (July 15-August 5) is the divergence, with the path chosen
based on the checkpoint and advisor input.


## 3. Phase 1: Shared convergence zone (July 1-14)

Goal: close the causal loop between the outlier map and accuracy, and resolve
all outstanding methodology decisions so that the second half of July can
proceed with a clear direction.

### 3.1 50,000-image Experiment 1 re-run (Day 1-2, parallel with other work)

The current numbers in `outputs/exp1_outlier_maps/outlier_stats.json` and all
charts come from a 5,000-image run. The qkv policy-bucket instability
documented in `docs/advisor-touchpoint-guide.md` Section 3 "Testing" shows
that the routing table is sample-size sensitive at the 0.5% boundary. The
quickest way to resolve that is to re-run at the full 50,000-image size:

```
python run_experiment1_mapping.py --num-images 50000 --batch-size 128
```

This takes a few hours on the RTX 3070 and can be launched first thing and
left to complete while other work proceeds. Output: a final
`outlier_stats.json` and regenerated charts that can replace the 5,000-image
numbers as the authoritative version, settling the qkv question about
which specific `attn.qkv` layers belong in the LLM.int8 bucket.

### 3.2 Build quantization.py (Day 1-4)

None of Experiments 2-4 can run without a simulated quantization module. The
spec (`docs/vit-agent-grounding.md`, Experiment 2) calls for:

- Naive fake quantization: quantize to INT8, dequantize back to FP32/16
- Three scaling strategies: per-tensor, per-channel (weights), per-token
  (activations)
- A clean API that can be slotted into forward hooks for Experiment 3
  (per-layer sensitivity) and into a threshold-based mask for Experiment 4
  (decomposition)

This module (`src/quantization.py`) gets built first because Experiment 3
needs it. It does not need the 50K re-run or any advisor decisions to
proceed. Write it against small unit tests (the `tests/test_hooks.py` pattern
can be followed: hand-built tensors with known quantize/dequantize outcomes).

### 3.3 Run Experiment 3: per-layer sensitivity (Day 4-10)

**This is the most important experiment of July.** Its result determines
whether the outlier map has predictive power:

- Iterate through all 49 linear layers (or a reasonable subset, prioritized:
  blocks 3-8-9-10-11 fc1, all qkv at 0.39%+, plus a sample of the near-zero
  layers as controls)
- For each layer, quantize it to naive INT8 while leaving the rest of the
  model in FP32, run inference on a labeled accuracy eval set (the validation
  split or a held-out subset, see 3.4), and record the Top-1 accuracy drop
- Output a sensitivity heatmap (bar chart per layer, accuracy drop on the y-axis)

If the two largest bars land on `blocks.9.mlp.fc1` and `blocks.10.mlp.fc1`,
the outlier map is validated. If they do not, the project has a bigger problem
than contribution framing and needs to understand why. A partial validation
(blocks 9-10 dominate but the qkv cluster also shows meaningful drops) is
still informative and expected.

An 8-bit accuracy simulation at 50,000 images takes significant time.
Prioritize a targeted layer list (maybe 15-20 of 49) over an exhaustive sweep
if wall-clock time is tight. The key layers are: all 12 fc1, blocks 3-6 and
8-11 qkv, all 12 proj as negativity controls, and the head.

### 3.4 Resolve the calibration/evaluation data-split question (Day 1-7, decision)

This issue (`docs/advisor-touchpoint-guide.md` Section 4.7) blocks Experiments
2-4: if we measure accuracy on the same 5,000/50,000 validation images used to
build the outlier map in Experiment 1, the whole pipeline is circular.

Before Experiment 3's accuracy numbers are trusted, one of these must happen:

| Option | What it means | Effort |
|---|---|---|
| A | Hold out ~20% of the validation set for accuracy eval, use the remaining ~80% for Experiment 1's statistics | Minimal: split the directory once |
| B | Download a separate calibration set (a few thousand images from the ImageNet training split) for Experiment 1, and reserve the full validation set for accuracy | Medium: download + re-run Exp 1 |
| C | Accept the overlap, document it, and argue the risk is bounded for the fixed-6.0 threshold (which is a constant from prior work, not fit to our data) | None |

Option A is the pragmatic middle ground: it avoids another download, separates
calibration and evaluation without a full new Experiment 1 run, and is
defensible (the calibration set still contains thousands of unseen images).

This decision needs advisor input or an executive call. If no decision is
reached by Day 7, default to Option A for Experiment 3 and flag the choice
explicitly in the writeup.

### 3.5 Phase 1 checkpoint (Day 14, approximately July 14)

By this point, the following should be true:

- [ ] Experiment 1 50K re-run is complete; final routing table is settled
- [ ] `src/quantization.py` is built and tested
- [ ] Experiment 3 has run on at least the priority layers, with accuracy
      numbers that show whether blocks 9-10 fc1 dominate the drop
- [ ] The calibration/eval split question is resolved
- [ ] Advisor input on contribution framing has been received (or a decision
      has been made unilaterally by the researcher)

At this checkpoint, the project commits to one of the two Phase 2 paths.


## 4. Phase 2 Path A: Validate and write up (July 15 - August 5)

### What this path delivers

A complete, verified story: the outlier map (Experiment 1) predicted which
layers are quantization-sensitive; the sensitivity experiment (3) confirmed
the prediction; the decomposition experiment (4) shows that LLM.int8 recovers
accuracy on the routable layers and that the FP16 layers are irrecoverable by
thresholding alone.

The contribution is: a principled method for assigning per-layer precision
regimes to ViT-B/16 based on activation statistics, validated against accuracy
measurements.

### Experiment 4: decomposition (Day 15-22)

Implement the fixed-6.0 threshold decomposition for every measured layer (or,
pragmatically, for blocks 8-9-10 fc1 plus a sample of controls). For each
layer:

- Quantize values below 6.0 to INT8; keep values above 6.0 in FP32
- Apply the 25% column-participation bar from Experiment 1 to decide
  whole-column routing (matching the routing policy logic)
- Measure Top-1 accuracy and the actual high-precision fraction

The expected results, from Experiment 1's data:
- Blocks 9-10 fc1: high-precision fraction ~97-100%, accuracy close to FP16
  baseline (no INT8 benefit, confirming the policy says FP16)
- Block 8 fc1: high-precision fraction ~3.8%, accuracy near FP32 (confirms
  LLM.int8() works on this layer)
- Controls (proj layers): high-precision fraction ~0%, accuracy near naive
  INT8

### Experiment 2: granularity (Day 22-27)

This is lower priority than 3 and 4 but fills out the picture: test whether
per-token activation scaling provides any relief for blocks 9-10 relative to
per-tensor scaling. The hypothesis is that per-token scaling helps somewhat
(adapting to token-by-token scale variation) but does not solve the structural
problem (too many outlier columns). This experiment can be scoped down to a
subset of layers to keep the timeline workable.

### Remaining time: writeup and the 50K ImageNet re-run (Day 27 - August 5)

If the 50K re-run was already completed in Phase 1, this buffer week is for
the thesis writeup: methods section, results, the routing table, discussion of
the "density, not scatter" finding, the borderline qkv cluster, limitations,
and comparison to prior work.


## 5. Phase 2 Path B: Add a smoothing remedy (July 15 - August 5)

### What this path delivers

A novel technique (activation-smoothing or weight-equalization, applied to
ViT-B/16) plus evidence that it reduces the fc1 scale explosion in blocks 9-10
and allows those layers to re-enter the LLM.int8 regime (or pure INT8). The
Experiment 1+3+4 pipeline still runs as validation, but the headline
contribution changes from "here is a per-layer policy" to "here is a method
that fixes the problem the per-layer policy found."

### What the smoothing needs to accomplish

Blocks 9-10 fc1 have σ = 6.42 and 11.68 (from the 5K run). The LLM.int8
fixed-6.0 threshold flags 97.7% and 99.7% of columns. If smoothing can bring
σ down to something closer to block 8's σ = 3.06 (where the routing fraction
is a manageable 3.78%), those two layers move from FP16 to LLM.int8. That is
the success criterion.

### Concrete approach: SmoothQuant-style activation smoothing

SmoothQuant (Xiao et al., 2023, "SmoothQuant: Accurate and Efficient
Post-Training Quantization for Large Language Models," arXiv:2211.10438)
introduces a per-channel scaling factor that migrates quantization difficulty
from activations to weights by applying a mathematically equivalent
transformation: scale activations down by factor `s`, scale weights up by the
same factor `s`. Because the factor cancels in the matmul output, the
transformation preserves the model's forward pass exactly before quantization
is applied.

Applied to ViT-B/16 blocks 9-10 fc1:
1. Compute a per-channel smoothing factor `s_c` for each of the 768 input
   channels, designed to reduce the activation scale in channels with the
   largest outliers
2. Apply `s_c` to fc1's input activations (scale down) and to fc1's weight
   matrix (scale up by `1/s_c`)
3. This shifts the activation scale explosion from the forward activation into
   the weight, where static quantization can absorb it more easily (weights
   are fixed after training)
4. Re-run Experiment 1's measurement to check whether the routing fraction
   drops below the LLM.int8 threshold

### Implementation plan (Day 15-25)

1. Implement per-channel smoothing in `src/quantization.py` (or a new
   `src/smoothing.py`). The smoothing factors can be derived from the
   per-channel activation statistics already computed in Experiment 1's Pass
   1, directly from `outputs/exp1_outlier_maps/outlier_stats.json`.

2. Apply smoothing to blocks 9-10 fc1 (start with those two, expand to others
   only if time permits).

3. Re-run the Experiment 1 measurement pipeline on the smoothed model to
   confirm the routing fraction dropped.

4. Run a focused Experiment 3-style sensitivity check on the two smoothed
   layers to confirm accuracy is preserved.

5. If smoothing works: integrate it into the Experiment 4 decomposition logic
   and run a combined accuracy sweep (smoothing + LLM.int8 at blocks 9-10) to
   get the headline number.

6. If smoothing does not work (routing fraction does not drop meaningfully, or
   accuracy degrades): document it as a negative result, default to Path A's
   Experiment 4, and the contribution becomes "we characterized the problem
   and tested a known remedy which was insufficient."

### Risk and fallback

The risk is that the smoothing factor that meaningfully reduces σ also
amplifies the weight's dynamic range enough that the weight itself becomes
unquantizable, shifting the problem rather than solving it. The fallback is
Path A: the validation story still stands, with the added note that a known
remedy was attempted and was insufficient. This is a defensible contribution
in its own right.


## 6. Path C (hybrid, lowest risk): partial smoothing + validate

If Path B is attempted but smoothing only partially succeeds (e.g., brings
block 9 down from FP16 to LLM.int8 but block 10 remains FP16), the
contribution becomes: "SmoothQuant-style activation smoothing recovers one of
the two catastrophic layers, demonstrating that the technique partially
transfers from LLMs to ViTs." This is a smaller but still positive result.

The implementation is identical to Path B up to the checkpoint on Day 20: try
smoothing on blocks 9-10, check whether σ dropped. If it worked partially,
write it up as a mixed result and still run Experiment 4 for completeness. If
it failed entirely, fall back to Path A.


## 7. What is explicitly out of scope for July

These items are listed to prevent scope creep. They were considered and
deferred:

- **Second architecture (ViT-L/16 or DeiT).** Extending to another model
  requires downloading new weights, re-running Experiment 1, and potentially
  debugging a different layer structure. There is not enough time to do this
  and also validate the causal loop on ViT-B/16, which is the prerequisite
  for any generalization claim. If Experiments 3 and 4 finish early (before
  Day 30), a quick ViT-L/16 Experiment 1 pass could be squeezed in, but it
  should not be counted on.

- **Hardware EDP measurement on Jetson Orin Nano.** This is the highest-impact
  item but requires hardware setup, cross-compilation, and benchmarking that
  cannot be parallelized with the software pipeline in the remaining time. The
  thesis should acknowledge the simulation-vs-hardware gap explicitly (already
  documented in `docs/vit-agent-grounding.md`, Experiment 4 scope note) and
  treat hardware EDP as future work.

- **Per-Q/K/V decomposition.** Splitting the fused `attn.qkv` projection into
  separate Q, K, V measurements is analytically interesting but does not
  change the routing policy in a way that matters for the thesis's core claim
  (the outlier qkv layers are ~0.4-0.8% routing fraction either way). Defer to
  the writeup's future-work section.

- **Bootstrap confidence intervals on routing fractions.** Acknowledge the
  absence in the writeup's limitations section but do not spend July compute
  budget on it.

- **Any new training or fine-tuning.** The project is described as post-training
  quantization. No gradient updates.

- **The 3-sigma threshold as a decomposition arm in Experiment 4.** This is
  recommended for demotion (see `docs/advisor-touchpoint-guide.md` Section
  4.5). If the advisor overrules this, it can be added back, but the
  self-normalizing behavior means it will produce a near-zero signal at blocks
  9-10 regardless, and the Experiment 1 charts already show this directly.
  Running the full decomposition sweep with it would only re-confirm what
  Section 3.5 already shows. Unless there is a specific reason to run it
  anyway (e.g., as a deliberate negative-result contrast in the final paper),
  keep it out of the July plan.


## 8. Week-by-week calendar

### Week 1 (July 1-7): Foundation

- Launch 50K Experiment 1 re-run (background)
- Build `src/quantization.py` (fake quantize, 3 scaling strategies)
- Unit-test quantization against hand-computed values
- Resolve calibration/eval split question (default to Option A if no advisor
  input by Day 7)
- Prepare the labeled accuracy-eval subset (hold-out split or the full val set
  with a split, depending on decision above)

### Week 2 (July 7-14): Causal validation

- Run Experiment 3 on the priority layer subset (~15-20 layers)
- If the 50K re-run finished, incorporate its numbers into the sensitivity
  experiment's reporting
- Checkpoint at end of week: do blocks 9-10 fc1 dominate the accuracy drop?

### Week 3 (July 15-21): Path decision and main experiment

| If Path A (validate) | If Path B (smoothing) |
|---|---|
| Run Experiment 4 decomposition at fixed-6.0 | Implement per-channel smoothing for blocks 9-10 |
| Verify accuracy recovery on routable layers | Re-measure σ and routing fraction on smoothed model |
| Verify no recovery on FP16 layers | Run focused accuracy check on smoothed layers |

### Week 4 (July 22-28): Completion

| If Path A continued | If Path B continued |
|---|---|
| Run Experiment 2 (granularity, scoped) | If smoothing worked: run Experiment 4 + smoothing |
| Begin thesis writeup | If smoothing partially worked: document hybrid result |
| | If smoothing failed: fall back to Path A Experiment 4 |

### Week 5 (July 29 - August 4): Writeup and buffer

- Complete thesis methods, results, discussion sections
- Generate final charts and tables
- Address limitations (hardware gap, n=1 model, calibration/eval overlap if
  unresolved, qkv bucket instability noted, channel-persistence caveats)
- Re-run any experiment that produced marginal or puzzling results

### August 5-6: Final polish

- Proofread, check cross-references, confirm all figures are current with the
  50K run
- Ensure `outputs/` contains the final versions of all JSON and chart files
- Confirm the full test suite still passes


## 9. What to do if Experiments 3-4 contradict the outlier map

This is the worst-case scenario for the project, but it is not the end of the
story. If Experiment 3 shows that block 8's fc1 causes more accuracy loss than
block 9's, or that an `attn.proj` layer shows unexpected sensitivity, the
outlier map is not a reliable predictor and the routing-policy contribution is
not credible. In that case:

1. Document the discrepancy explicitly: what the outlier map predicted vs.
   what happened, with quantitative numbers
2. Characterize why the discrepancy occurred: does the failure mode involve
   compounding (a layer that looks clean in isolation but amplifies noise from
   earlier blocks)? Is there a measurement artifact in the pre-hook pipeline?
3. Frame the contribution around the negative result: the outlier map is not
   sufficient as a standalone routing decision engine, and the community
   should know this before adopting per-layer routing policies for ViTs based
   solely on activation statistics

A well-documented, honest negative result is a legitimate thesis contribution.
It is better than an unvalidated positive claim.


## 10. Summary: what to walk away from this document with

1. **Experiment 3 is the key July experiment, period.** It closes the causal
   loop and tells us whether the whole project is on solid ground. Everything
   else sequences after it.

2. **The contribution framing needs a decision, but Experiment 3 buys time to
   make it.** By mid-July, the accuracy validation will be in hand, and the
   Path A/Path B/C decision can be made on real data rather than speculation.

3. **The quickest impact item is the 50K re-run**, which resolves the qkv
   boundary instability, costs a few hours of GPU time, and can run in the
   background on Day 1.

4. **Building `quantization.py` is the prerequisite for everything else**,
   and it does not depend on any advisor decision. Start there.

5. **Scope is cut deliberately.** No second architecture, no hardware, no 3σ
   decomposition arm, no per-Q/K/V analysis. These are good ideas that belong
   in the future-work section, not in the July sprint.

6. **The calibration/eval split is the one prerequisite decision that cannot
   be deferred past Week 1**, because it affects the integrity of every
   accuracy number collected from Week 2 onward. If the advisor has not
   weighed in by July 3, default to Option A (hold-out split) and document it.
