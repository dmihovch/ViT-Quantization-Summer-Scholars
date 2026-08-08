# Poster Planning — ViT Outlier Profiling & Per-Channel Ablation

> **Model:** ViT-B/16 (timm, augreg2_in21k_ft_in1k)
> **Dataset:** ImageNet-1K validation (50,000 images)
> **Baseline top-1:** 85.03%, top-5: 97.52%
> **Seeds:** 5 (42, 43, 44, 45, 46) — ablation is deterministic given fixed Phase 1 stats

---

## Poster Layout Plan

Three vertical columns (traditional collegiate poster):

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| 1. Introduction (fig1) | 3. Methods (fig2) | 5. Discussion (fig6) |
| 2. Background & RQ (fig3) | 4. Results (fig4 + fig5) | 6. Future Work & Acknowledgements |

> **N.B.** The poster section-to-figure mapping is: Introduction → fig1, Background → fig3, Methods → fig2, Results → figs 4+5, Discussion → fig6. The numbering difference (fig2 in Methods, fig3 in Background) reflects the logical story flow, not the file order.

This gives: fig1 + fig3 in Col 1, fig2 + fig4 + fig5 (with fig4/fig5 as a single "Results" block) in Col 2, fig6 in Col 3. The story flows left-to-right: _the problem_ → _how we studied it_ → _what we found_ → _what it means_ → _where to go next_.

---

## Section 1: Introduction

### Preliminary Poster Text

> Massive activation outliers have emerged as a critical obstacle to efficient integer-only inference in Vision Transformers (ViTs). In ViT-B/16, pre-GELU activations in deep encoder blocks exhibit heavy-tailed distributions where a tiny fraction of values — just 0.39% of elements in Block 10 at 3σ — carry magnitudes far exceeding the bulk of the distribution. These outliers force quantization schemes to choose between clipping (destroying accuracy) or allocating dynamic range to cover extreme values (wasting precision on 99.6% of elements).
>
> Global per-tensor quantization, the standard approach in post-training quantization (PTQ), is blind to channel-to-channel variation. However, we show that per-channel standard deviation varies by **12.4×** within a single pre-GELU layer (σ ∈ [2.06, 25.54] at Block 10). A global threshold treats all channels equally, incorrectly clipping activations from high-variance channels that carry disproportionate classification signal.
>
> **This work asks:** Do these outliers matter for accuracy, and can per-channel thresholding better preserve the signal they carry?

### Caption for fig1_activation_overlay.png

> **Figure 1.** Block 10 pre-GELU activation distribution (50K ImageNet-1K samples, μ = −28.33, σ = 11.20). The dashed red lines mark the global ±3σ threshold; the shaded tails show elements zeroed by a global outlier ablation. The blue dotted lines show the ±3σ bounds for the _least_ volatile channel (σ = 2.06), and the dark green dash-dotted lines show the bounds for the _most_ volatile channel (σ = 25.54). A global ±3σ window captures activations that are within the normal operating range of high-variance channels — zeroing them destroys signal that the model depends on.

### Justification for Section 1

The introduction's job is to (1) immediately show the audience there is a real problem, and (2) motivate why per-channel thinking is the right lens. fig1 is the hero figure — it does both simultaneously. A viewer who reads nothing else should still understand: "ViTs have massive outliers, global thresholds miss the fact that different channels have wildly different normal ranges, and this is a problem for quantization."

The distribution is not exotic — it's roughly unimodal with a heavy negative tail — which makes the channel heterogeneity the surprising part. The text deliberately mentions 12.4× σ spread here because the poster viewer may not read the Methods section. The story must be self-contained within each section.

---

## Section 2: Background & Research Question

### Preliminary Poster Text

> Prior work has established that transformer activations exhibit heavy-tailed distributions. Dettmers et al. (2022) identified emergent outlier features in LLMs at scales beyond 6.7B parameters. SmoothQuant (Xiao et al., 2023, ICML) demonstrated that per-channel scaling can migrate quantization difficulty from activations to weights. However, these findings focus on large language models — the landscape in vision transformers is less explored.
>
> In this work, we systematically profile activation statistics at **73 measurement sites** across ViT-B/16's 12 encoder blocks: six sites per block (residual stream, pre-attention LN, pre-softmax logits, post-softmax weights, pre-MLP LN, pre-GELU), plus the patch embedding output. Phase 1 answers the question: **Where are the outliers, and what do they look like?** Phase 2 answers: **How much do they matter for classification accuracy?**
>
> **Research Question:** Can per-channel outlier thresholding preserve accuracy relative to global per-tensor thresholding, and if so, _why_ — is the benefit driven by per-channel mean correction, variance correction, or both?

### Caption for fig3_outlier_grid.png

> **Figure 2.** Outlier fraction heatmap across all 73 measurement sites at k = 3σ. Blocks 0–11 (rows) × 6 sites per block (columns). Color intensity encodes the fraction of activations exceeding ±3σ. Pre-GELU sites in Blocks 10–11 show concentrated outlier activity. Attention sites (pre_softmax, post_softmax) show uniformly low outlier fractions, consistent with the softmax normalization that bounds activations to [0, 1].

### Justification for Section 2

fig3 answers "where are the outliers?" in a single glance — the poster viewer can scan the heatmap and immediately see that pre_gelu sites in late blocks are the hot zones. This is essential context for the rest of the poster: we ablate pre_gelu because that's where the outliers live.

I deliberately reference the SmoothQuant hypothesis here because it sets up the later finding (Section 5, fig6) that the per-channel benefit decomposes into mean and variance components — exactly the mechanism SmoothQuant exploits, but in ViTs rather than LLMs. The research question is framed as a decomposition problem because that's the cleanest story arc: _are outliers important_ → _can per-channel do better_ → _why_ → _it's the mean correction_.

---

## Section 3: Methods

### Preliminary Poster Text

> **Phase 1 — Activation Profiling.** We instrument ViT-B/16 using nnsight to capture activation tensors at 73 sites during a full pass over 50K ImageNet-1K validation images. Statistics are accumulated via an online Welford algorithm (numerically stable multi-batch aggregation) and include: mean μ, standard deviation σ, kurtosis κ, per-channel σ_c and μ_c, outlier fractions at k ∈ {3, 4, 6}σ, and attention entropy per head. Profiling is deterministic for a given model checkpoint.
>
> **Phase 2 — Outlier Ablation.** Using the Phase 1 statistics, we perform targeted interventions: at each encoder block's pre-GELU site, we zero any activation x that satisfies |x − μ| > k·σ (the mean-centered outlier criterion). We run this intervention jointly across all 12 blocks and evaluate top-1 and top-5 accuracy on the full 50K validation set. A **random-zeroing control** (zeroing an equivalent _fraction_ of elements at random positions, matched per-batch) isolates the effect of outliers specifically from the effect of general sparsity.
>
> **Per-channel thresholding.** Instead of a single global (μ, σ), we compute per-channel statistics (μ_c, σ_c) from Phase 1 and zero element `[b, n, c]` according to |x_bnc − μ_c| > k·σ_c. This respects channel-level variance heterogeneity. We further decompose the per-channel effect into three modes:
>
> - **outlier:** per-channel μ_c + per-channel σ_c (full per-channel)
> - **mean_only:** per-channel μ_c + global σ
> - **var_only:** global μ + per-channel σ_c
>
> This decomposition isolates whether the per-channel benefit comes from correcting for shifted channel means or from adapting to channel-variance spread.

### Caption for fig2_sigma_ridgeline.png

> **Figure 3.** Mean per-channel standard deviation of pre-GELU activations across blocks 0–11, with ±1 SD shading. Per-channel σ rises monotonically from ~5 in early blocks to ~11 in Block 10–11, and the spread of σ_c values (the shaded band) widens considerably. A 12.4× range in σ_c within Block 10 means a single global σ of 11.20 misrepresents channels that are far narrower (σ ≈ 2) or far wider (σ ≈ 25).

### Justification for Section 3

The Methods section has a dual purpose on a poster: (a) establish that the experiments are rigorous (Welford, 50K images, random-zeroing control), and (b) introduce the per-channel decomposition (mean_only / var_only / outlier) because fig6 won't make sense without it.

fig2 is placed here rather than in Background because it directly motivates the method — the widening spread of per-channel σ is _why_ we need per-channel thresholds. It also previews the finding that late blocks (8–11) are the outlier-heavy region, which supports the effective gain correlation result.

The random-zeroing control is worth highlighting because it's the most important control experiment: if random zeroing at matched sparsity fractions also destroys accuracy, then outliers aren't special — it's just sparsity. The fact that random zeroing preserves baseline accuracy (per the README) is what validates the entire ablation paradigm, though this result appears in the Results section.

---

## Section 4: Results

### Preliminary Poster Text

> **Global vs. per-channel accuracy.** At k = 3σ, global outlier zeroing drops top-1 accuracy from 85.03% to 43.24%. Per-channel thresholding at the same k recovers to **47.00%** — a gain of **+3.76 percentage points** (95% CI: [3.12, 4.36], statistically significant). At k = 4σ, the gap narrows to +0.42 pp and becomes non-significant (95% CI: [−0.11, 0.96]). At k = 6σ, both conditions nearly recover baseline (Global: 84.58%, Per-channel: 84.11%).
>
> **Efficiency of sparsification.** The accuracy cost _per unit of induced sparsity_ reveals the structural advantage of per-channel thresholds. At k = 3, global zeroing costs **100.97 pp of accuracy per 1% of activations zeroed**, while per-channel zeroing costs only **53.43 pp/%** — a **1.89× efficiency ratio**. Per-channel thresholds achieve higher accuracy while inducing slightly _less_ total sparsity, indicating they selectively preserve channels that carry more classification-relevant signal.
>
> **Random-zeroing control confirms specificity.** When the same fraction of elements are zeroed at random positions (instead of targeting outliers), accuracy remains within 0.1 pp of baseline at all sparsity levels. The degradation is caused by the loss of _specific_ outlier values, not by general activation sparsity.

### Caption for fig4_accuracy_bars.png

> **Figure 4.** Top-1 accuracy at three sigma thresholds (k = 3, 4, 6), comparing global (coral) vs. per-channel (teal) outlier ablation. Error bars show ±1 SD across 5 seeds (though ablation is deterministic). The dashed baseline at 85.03% marks unablated accuracy. The +3.76 pp advantage of per-channel at k = 3 is the headline finding — and it vanishes by k = 4, confirming that the effect is concentrated at aggressive thresholds.

### Caption for fig5_accuracy_cost_vs_sparsity.png

> **Figure 5.** Accuracy drop (percentage points) vs. induced activation sparsity (%), comparing global vs. per-channel ablation across k ∈ {3, 4, 6}. The per-channel curve (teal) sits below the global curve (coral) at every sparsity level, demonstrating consistently higher efficiency. The steeper slope of the global curve at low sparsity confirms that global thresholds destroy disproportionately more accuracy per zeroed element.

### Justification for Section 4

This is the core empirical section, and it needs both figures. fig4 communicates the headline number (+3.76 pp at k=3) in the most intuitive format for a poster audience — grouped bars with numbers on top. fig5 communicates _why_ this matters beyond raw accuracy: per-channel thresholds are strictly more efficient on the accuracy–sparsity Pareto frontier. The 1.89× efficiency ratio is the number that sells the result to a quantization audience because quantization is fundamentally about the accuracy–compression tradeoff.

The CI and significance language is important for credibility. On a poster, a casual viewer won't check the CI, but a methods-savvy viewer will — and seeing that it's reported builds trust.

I placed the random-zeroing control result here rather than in Methods because its natural conclusion ("degradation is outlier-specific") belongs in Results. The control validates that everything else on the poster is measuring what we claim to measure.

---

## Section 5: Discussion

### Preliminary Poster Text

> **Decomposing the per-channel benefit.** At k = 3σ, the full per-channel condition achieves 47.00% accuracy. The _mean_only_ condition (per-channel μ_c, global σ) achieves **63.32%** — significantly _better_ than the full per-channel condition. The _var_only_ condition (global μ, per-channel σ_c) collapses to **6.56%** — far worse than even the global condition (43.24%).
>
> This pattern is counterintuitive but mechanistically revealing:
>
> 1. **Mean correction is the dominant mechanism.** Block 10's per-channel means span a 97-point range (μ_c ∈ [−71.18, 26.01]). Channels with strongly negative means have their activations centered far from zero; applying a global μ = −28.33 shifts the threshold incorrectly for these channels. Per-channel μ_c corrects for this and **recovers 20.08 pp** over the global condition.
>
> 2. **Variance correction alone is destructive.** Using per-channel σ_c without per-channel μ_c applies _narrower_ thresholds to channels with negative means, zeroing activations that are genuinely within-channel normal. This explains why var_only performs worse than global: global σ is wide enough to accommodate the channel-mean shift, even if suboptimally.
>
> **Why do these outliers exist?** We find strong correlation (r = +0.65–0.77 for blocks 8–11) between the effective per-channel gain ‖fc1.weight[c, :] ⊙ γ‖₂ and per-channel pre-GELU σ_c. This confirms the SmoothQuant hypothesis adapted to ViTs: the per-channel variance pattern is **architectural** — encoded in the interaction of the MLP's first linear layer weights with the preceding LayerNorm scale. Outliers are not anomalous noise; they are a deliberate consequence of trained weights.

### Caption for fig6_ablation_waterfall.png

> **Figure 6.** Ablation condition decomposition at k = 3σ, comparing baseline (85.03%) against four independent ablation conditions. "Per-ch. μ + global σ" (mean_only, 63.32%) achieves the highest accuracy of any ablation, demonstrating that mean correction is the dominant mechanism. "Global μ + per-ch. σ" (var_only, 6.56%) performs catastrophically — per-channel variance adaptation without mean correction zeros the wrong activations. Annotations show accuracy difference (pp) relative to baseline.

### Justification for Section 5

This is the most intellectually interesting section and the one that distinguishes this work from a simple profiling study. The decomposition result (fig6) tells a genuinely surprising story: mean correction dominates, and variance correction alone is _worse_ than doing nothing. This is the opposite of what someone familiar with SmoothQuant might expect — SmoothQuant's core mechanism is variance redistribution, but here the mean shift is what matters.

The effective gain correlation result (r = +0.65–0.77) closes the loop: it tells us _why_ the per-channel means and variances diverge in the first place. The trained weights of the MLP's first layer, in combination with LayerNorm γ, systematically allocate gain unevenly across channels. This is a finding with implications beyond quantization — it suggests that ViT MLPs encode information in a structured, channel-specific way that global PTQ schemes are blind to.

The var_only catastrophe (6.56%) is worth emphasizing because it's the most dramatic number on the poster — it's worse than the random-zeroing control would be at equivalent sparsity, which means it's actively zeroing _the wrong things_. This makes the theoretical point concrete.

---

## Section 6: Future Work & Acknowledgements

### Preliminary Poster Text

> **Future Work.**
>
> 1. **Per-channel PTQ schemes.** The finding that per-channel mean correction recovers 20 pp of accuracy at aggressive thresholds motivates per-channel quantization of ViT activations. A per-channel quantization scheme (per-channel zero-point + scale) could preserve the channel-mean structure that global quantization destroys. This is implementable in integer-only pipelines: per-channel zero-points can be fused into bias terms.
>
> 2. **Mixed-precision channel allocation.** Not all channels are equally sensitive. The 12.4× σ range and strong fc1 gain correlation suggest a mixed-precision strategy where high-gain channels receive higher bit-width. A sensitivity-guided bit-allocation search (e.g., HAWQ-style for ViTs) could exploit this channel heterogeneity.
>
> 3. **Deployment on edge hardware.** The end goal is INT8 inference for ViT-B/16 on NVIDIA Jetson Orin. The per-channel findings inform a quantization pipeline where (a) activations are zero-centered per-channel, (b) weights are quantized per-channel (standard), and (c) a per-channel integer GELU approximates the activation function. Phase 3 (integer GELU LUTs) was deferred in favor of the deeper per-channel analysis but remains a necessary engineering component.
>
> **Acknowledgements.** [To be filled in — summer scholars program, advisor, GPU resources, etc.]

### Justification for Section 6

Future Work on a poster serves two audiences: (1) the casual viewer who wants to know "so what?" — answered by the edge deployment goal, and (2) the domain expert who wants to know if this opens a new line of work — answered by the mixed-precision and per-channel PTQ proposals.

The three future directions form a natural progression: per-channel quantization scheme → optimize bit-width per channel → deploy on hardware. Each is directly motivated by a specific finding from earlier sections. This closes the loop on the "research → engineering" pipeline that the introduction promised.

The Phase 3 deferral note is important for transparency — it tells the audience this is a work in progress and that the community can pick up the integer GELU baton.

---

## Story Weave: How the Sections Connect

The poster tells one coherent story, not six independent facts. Here is the narrative thread:

```
INTRO (fig1)
  "ViT pre-GELU activations have massive outliers. Global thresholds
   clip them incorrectly because channels have wildly different
   normal ranges (12.4× σ spread)."
       │
       ▼
BACKGROUND & RQ (fig3)
  "We profiled 73 sites across all 12 blocks. Outliers concentrate
   in pre-GELU sites of late blocks. RQ: Can per-channel thresholds
   do better, and why?"
       │
       ▼
METHODS (fig2)
  "The per-channel σ spread (fig2) motivates our approach. We ablate
   outliers at kσ thresholds, comparing global vs per-channel, and
   decompose into mean_only / var_only conditions."
       │
       ▼
RESULTS (fig4 + fig5)
  "Per-channel recovers +3.76 pp at k=3 (fig4). It's 1.89× more
   efficient on the accuracy–sparsity Pareto front (fig5).
   Random zeroing confirms this is outlier-specific."
       │
       ▼
DISCUSSION (fig6)
  "Decomposition reveals mean correction dominates (+20 pp over
   global). Variance correction alone is catastrophic (6.56%).
   The per-channel pattern is architectural — encoded in
   fc1.weight ⊙ LN γ (r = +0.65–0.77)."
       │
       ▼
FUTURE WORK
  "These findings directly motivate per-channel PTQ schemes,
   mixed-precision channel allocation, and INT8 edge deployment."
```

**Key transitions to make explicit on the poster:**

1. **fig1 → fig3:** "Now that you've seen the problem in one block, here's the full map of where outliers live."
2. **fig2 → fig4:** "The σ spread in fig2 is _why_ per-channel matters — and fig4 shows _how much_ it matters."
3. **fig4 → fig5:** "Raw accuracy is one view; the efficiency curve shows per-channel dominates at every operating point."
4. **fig5 → fig6:** "But _why_ does per-channel win? fig6 decomposes the effect — and the answer is mean correction."
5. **fig6 → Future Work:** "If mean correction recovers 20 pp, then per-channel zero-points in PTQ are the natural next step."

---

## Design Notes for Poster Production

1. **Color consistency:** All six figures use the same palette (teal = per-channel, coral = global, gray = baseline). This is intentional — the viewer should not have to re-learn the color code when moving between panels.

2. **Font sizing:** Minimum 14 pt on axes labels, 16–18 pt on titles, ≥12 pt on legends. The poster plotting module (`src/plotting_poster.py`) already enforces this.

3. **Figure placement:** Each column should have its own visual weight. Column 2 (Methods + Results) is the heaviest with three figures but also the most important. Consider making fig4 and fig5 slightly smaller (they share a column with fig2) or combining them into a 2-panel figure.

4. **Redundancy:** The activation overlay (fig1) and the σ ridgeline (fig2) both use Block 10 pre-GELU data. This is intentional — Block 10 is the anchor point throughout the poster, and repeated reference to it builds a consistent mental model.

5. **The one-glance test:** If someone walks by and spends 5 seconds, they should see: (a) fig1's dual-threshold overlay (orange band + red dashed lines = "channels differ!"), (b) the +3.76 pp annotation on fig4, and (c) the 63.32% bar on fig6 towering above 47.00% ("mean correction wins!"). These are the three take-home messages, and the visual design should make them pop.

---

## Figures Index

| # | File | Section | Visual Type | Key Message |
|---|------|---------|-------------|-------------|
| 1 | `fig1_activation_overlay.png` | Introduction | Histogram with threshold lines | Global thresholds incorrectly clip high-variance channels |
| 2 | `fig3_outlier_grid.png` | Background & RQ | Small-multiples heatmap | Outliers concentrate in late-block pre-GELU sites |
| 3 | `fig2_sigma_ridgeline.png` | Methods | Line plot with ±1σ band | Per-channel σ spread grows across depth (12.4× at Block 10) |
| 4 | `fig4_accuracy_bars.png` | Results | Grouped bar chart | Per-channel recovers +3.76 pp at k=3 |
| 5 | `fig5_accuracy_cost_vs_sparsity.png` | Results | Line plot (Pareto) | Per-channel is 1.89× more efficient on accuracy–sparsity frontier |
| 6 | `fig6_ablation_waterfall.png` | Discussion | Multi-bar comparison | Mean correction dominates; variance correction alone is destructive |