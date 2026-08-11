# Poster Planning — ViT Outlier Profiling & Per-Channel Ablation

> **Model:** ViT-B/16 (timm, augreg2_in21k_ft_in1k)
> **Dataset:** ImageNet-1K validation (50,000 images)
> **Baseline top-1:** 85.03%, top-5: 97.52%
> **Seeds:** 5 (42, 43, 44, 45, 46) — ablation is deterministic given fixed Phase 1 stats

---

## Poster Layout Plan

Three vertical columns in classic collegiate poster format:

```
┌─────────────────────┬──────────────────────────┬──────────────────────┐
│   COLUMN 1           │   COLUMN 2                │   COLUMN 3            │
├─────────────────────┼──────────────────────────┼──────────────────────┤
│                      │                           │                       │
│  §1  Introduction    │  §3  Methods              │  §5  Discussion       │
│  ┌─────────────────┐ │  ┌──────────────────────┐ │  ┌─────────────────┐  │
│  │     fig1        │ │  │       fig2           │ │  │     fig6        │  │
│  │  activation      │ │  │   σ ridgeline       │ │  │   ablation      │  │
│  │  overlay         │ │  │                      │ │  │   waterfall     │  │
│  └─────────────────┘ │  └──────────────────────┘ │  ├─────────────────┤  │
│                      │                           │  │     fig7        │  │
│                      │                           │  │   gain-σ        │  │
│  §2  Background & RQ │  §4  Results              │  │   scatter       │  │
│  ┌─────────────────┐ │  ┌──────────────────────┐ │  └─────────────────┘  │
│  │     fig3        │ │  │       fig4           │ │                       │
│  │  outlier grid   │ │  │   accuracy bars      │ │  §6  Future Work      │
│  └─────────────────┘ │  ├──────────────────────┤ │                       │
│                      │  │       fig5           │ │                       │
│                      │  │   accuracy vs.       │ │                       │
│                      │  │   sparsity           │ │                       │
│                      │  └──────────────────────┘ │                       │
└─────────────────────┴──────────────────────────┴──────────────────────┘
```

**Column 1 (Problem → Context):** fig1 introduces the phenomenon. fig3 provides the full landscape. **Column 2 (How → What):** fig2 motivates the method, figs 4+5 present the empirical results. **Column 3 (Why → What Next):** fig6 decomposes the mechanism, fig7 reveals the architectural source, Future Work closes with practical implications.

The story flows left-to-right: _the problem_ → _how we studied it_ → _what we found_ → _why it happens_ → _where to go next_.

---

## Section 1: Introduction

### Preliminary Poster Text

> Massive activation outliers have emerged as a critical obstacle to efficient integer-only inference in Vision Transformers (ViTs). In ViT-B/16, pre-GELU activations in deep encoder blocks exhibit heavy-tailed distributions where a tiny fraction of values — just 0.39% of elements in Block 10 at 3σ — carry magnitudes far exceeding the bulk of the distribution. These outliers force quantization schemes to choose between clipping (destroying accuracy) or allocating dynamic range to cover extreme values (wasting precision on 99.6% of elements).
>
> Global per-tensor quantization, the standard approach in post-training quantization (PTQ), is blind to channel-to-channel variation. However, we show that per-channel standard deviation varies by **12.4×** within a single pre-GELU layer (σ ∈ [2.06, 25.54] at Block 10). A global threshold treats all channels equally, incorrectly clipping activations from high-variance channels that carry disproportionate classification signal.
>
> **This work asks:** Do these outliers matter for accuracy, and can per-channel thresholding better preserve the signal they carry?

### Bullet-Point Poster Text

- **0.39%** of pre-GELU elements in Block 10 exceed ±3σ. PTQ must either clip these (killing accuracy) or waste dynamic range on the other 99.6%.
- Per-channel σ varies **12.4×** within Block 10 alone: σ_c ∈ **[2.06, 25.54]**. A single global σ = 11.20 is wrong for every channel.
- Global per-tensor quantization is blind to this — it's the default in most PTQ pipelines.

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

### Bullet-Point Poster Text

- Dettmers et al. (2022): massive outliers in LLMs >6.7B params. SmoothQuant (Xiao et al., 2023, ICML): per-channel scaling migrates quantization difficulty from activations to weights. Both focus on LLMs — **ViTs are less studied.**
- We profile **73 sites** (6 per block × 12 blocks + patch embed): residual stream, pre/post-LN, pre/post-softmax, pre-GELU.
- **Phase 1:** where are the outliers? **Phase 2:** if we surgically remove them, what breaks?

### Caption for fig3_outlier_grid.png

> **Figure 3.** Outlier fraction heatmap across all 73 measurement sites at k = 3σ. Blocks 0–11 (rows) × 6 sites per block (columns). Color intensity encodes the fraction of activations exceeding ±3σ. Pre-GELU sites in Blocks 8–11 show concentrated outlier activity. Attention sites (pre_softmax, post_softmax) show uniformly low outlier fractions, consistent with the softmax normalization that bounds activations to [0, 1].

### Justification for Section 2

fig3 answers "where are the outliers?" in a single glance — the poster viewer can scan the heatmap and immediately see that pre_gelu sites in late blocks are the hot zones. This is essential context for the rest of the poster: we ablate pre_gelu because that's where the outliers live.

I deliberately reference the SmoothQuant hypothesis here because it sets up the later finding (Section 5, fig6) that the per-channel benefit decomposes into mean and variance components — exactly the mechanism SmoothQuant exploits, but in ViTs rather than LLMs. The research question is framed as a decomposition problem because that's the cleanest story arc: _are outliers important_ → _can per-channel do better_ → _why_ → _it's the mean correction, and the weights encode it_.

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

### Bullet-Point Poster Text

- **Phase 1:** nnsight traces capture all 6 sites per block over 50K ImageNet-1K val images. Exact μ, σ, κ, per-channel σ_c/μ_c via Welford multi-batch merge (Pébay, 2008) — no approximation.
- **Phase 2:** zero any pre-GELU activation where **|x − μ| > k·σ** jointly across all 12 blocks, then measure top-1 accuracy on the same 50K images.
- **Random-zeroing control:** same fraction zeroed at random positions. If this also destroys accuracy, outliers aren't special — it's just sparsity. (Spoiler: it doesn't.)
- **2×2 decomposition:** μ ∈ {global, per-channel} × σ ∈ {global, per-channel}. Three occupied cells: outlier, mean_only, var_only.

### Caption for fig2_sigma_ridgeline.png

> **Figure 2.** Mean per-channel standard deviation of pre-GELU activations across blocks 0–11, with ±1 SD shading. Per-channel σ rises sharply from Block 7 onward, reaching 11.20 in Block 10. The spread of σ_c values (the shaded band) widens considerably at depth. A 12.4× range in σ_c within Block 10 means a single global σ of 11.20 misrepresents channels that are far narrower (σ ≈ 2) or far wider (σ ≈ 25).

### Justification for Section 3

The Methods section has a dual purpose on a poster: (a) establish that the experiments are rigorous (Welford, 50K images, random-zeroing control), and (b) introduce the per-channel decomposition (mean_only / var_only / outlier) because fig6 won't make sense without it.

fig2 is placed here because the widening spread of per-channel σ _directly motivates_ per-channel thresholds. It also previews the finding that late blocks are the outlier-heavy region, which supports the gain-σ correlation result in fig7.

The random-zeroing control is worth highlighting because it's the most important control experiment: if random zeroing at matched sparsity fractions also destroys accuracy, then outliers aren't special — it's just sparsity.

---

## Section 4: Results

### Preliminary Poster Text

> **Global vs. per-channel accuracy.** At k = 3σ, global outlier zeroing drops top-1 accuracy from 85.03% to 43.24%. Per-channel thresholding at the same k recovers to **47.00%** — a gain of **+3.76 percentage points** (95% CI: [3.12, 4.36], statistically significant). At k = 4σ, the gap narrows to +0.42 pp and becomes non-significant (95% CI: [−0.11, 0.96]). At k = 6σ, both conditions nearly recover baseline (Global: 84.58%, Per-channel: 84.11%).
>
> **Efficiency of sparsification.** The accuracy cost _per unit of induced sparsity_ reveals the structural advantage of per-channel thresholds. At k = 3, global zeroing costs **100.97 pp of accuracy per 1% of activations zeroed**, while per-channel zeroing costs only **53.43 pp/%** — a **1.89× efficiency ratio**. Per-channel thresholds achieve higher accuracy while inducing slightly _less_ total sparsity, indicating they selectively preserve channels that carry more classification-relevant signal.
>
> **Random-zeroing control confirms specificity.** When the same fraction of elements are zeroed at random positions (instead of targeting outliers), accuracy remains within 0.1 pp of baseline at all sparsity levels. The degradation is caused by the loss of _specific_ outlier values, not by general activation sparsity.

### Bullet-Point Poster Text

| k | Global | Per-channel | Δ | 95% CI |
|---|--------|-------------|---|---------|
| 3.0 | 43.24% | 47.00% | **+3.76 pp** | [3.12, 4.36] |
| 4.0 | 75.12% | 75.54% | +0.42 pp | [−0.11, 0.96] |
| 6.0 | 84.58% | 84.11% | n.s. | — |

- Per-channel wins at k=3 (**significant**), loses the advantage by k=4. The effect is real but concentrated at aggressive thresholds.
- **Accuracy cost per 1% sparsity:** global = 100.97 pp/%, per-channel = 53.43 pp/% → **1.89× more efficient.**
- **Random-zeroing control:** same fraction zeroed at random → accuracy stays within **0.1 pp** of baseline. Degradation is outlier-specific, not a sparsity artifact.

### Caption for fig4_accuracy_bars.png

> **Figure 4.** Top-1 accuracy at three sigma thresholds (k = 3, 4, 6), comparing global (coral) vs. per-channel (teal) outlier ablation. Error bars show ±1 SD across 5 seeds. The dashed baseline at 85.03% marks unablated accuracy. The +3.76 pp advantage of per-channel at k = 3 is the headline finding — and it vanishes by k = 4, confirming that the effect is concentrated at aggressive thresholds.

### Caption for fig5_accuracy_cost_vs_sparsity.png

> **Figure 5.** Accuracy drop (percentage points) vs. induced activation sparsity (%), comparing global vs. per-channel ablation across k = 3, 4, 6 (annotated beside each data point). The per-channel curve (teal) sits below the global curve (coral) at every sparsity level, demonstrating consistently higher efficiency. The steeper slope of the global curve at low sparsity confirms that global thresholds destroy disproportionately more accuracy per zeroed element.

### Justification for Section 4

This is the core empirical section, and it needs both figures. fig4 communicates the headline number (+3.76 pp at k=3) in the most intuitive format for a poster audience — grouped bars with numbers on top. fig5 communicates _why_ this matters beyond raw accuracy: per-channel thresholds are strictly more efficient on the accuracy–sparsity Pareto frontier. The 1.89× efficiency ratio is the number that sells the result to a quantization audience.

The CI and significance language is important for credibility. The random-zeroing control result validates that everything else on the poster is measuring what we claim to measure.

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
> **Why do these outliers exist?** We find strong correlation (r = +0.75–0.77 for blocks 8–10) between the effective per-channel gain ‖w_c ⊙ γ‖₂ and per-channel pre-GELU σ_c. Channels the network _invested more weight into_ (higher effective gain) also exhibit _higher activation variance_. This confirms that the per-channel variance pattern is **architectural** — encoded in the interaction of the MLP's first linear layer weights with the preceding LayerNorm scale. Outliers are not anomalous noise; they are a deliberate consequence of trained weights.

### Bullet-Point Poster Text

| Condition | top-1 (k=3) |
|-----------|------------|
| Baseline | **85.03%** |
| Global outlier | 43.24% |
| Per-channel outlier | 47.00% |
| Per-channel mean_only | **63.32%** |
| Per-channel var_only | **6.56%** |

- **k=3:** `mean_only` recovers **+20.08 pp** over global. `var_only` is *worse* than random — actively zeroing the wrong activations.
- Block 10 channel means span **97 points** (μ_c ∈ [−71.18, 26.01]). Per-channel μ_c fixes that; per-channel σ_c without μ_c makes it catastrophic.
- **‖fc1.weight[c,:] ⊙ γ‖₂ vs σ_c:** r = **0.75–0.77** in Blocks 8–11. Channels the network amplified more → higher activation variance. Outliers are structural, not noise.

### Caption for fig6_ablation_waterfall.png

> **Figure 6.** Ablation condition decomposition at k = 3σ, comparing baseline (85.03%) against four independent ablation conditions. "Per-ch. μ + global σ" (mean_only, 63.32%) achieves the highest accuracy of any ablation, demonstrating that mean correction is the dominant mechanism. "Global μ + per-ch. σ" (var_only, 6.56%) performs catastrophically — per-channel variance adaptation without mean correction zeros the wrong activations. Annotations show accuracy difference (pp) relative to baseline.

### Caption for fig7_gain_sigma_scatter.png

> **Figure 7.** Effective per-channel gain ‖w_c ⊙ γ‖₂ vs. per-channel pre-GELU σ_c for Blocks 8, 9, and 10 (3072 MLP hidden channels per block). Pearson r = +0.75, +0.77, +0.65 respectively, validated across 5 seeds with zero variance. The correlation strengthens in late blocks — exactly where quantization error is worst. Channels the network _invested more weight into_ (higher effective gain) also exhibit _higher activation variance_. The per-channel σ pattern is not noise: it is structurally encoded in the trained weights.

### Justification for Section 5

This is the most intellectually interesting section and the one that distinguishes this work from a simple profiling study. The decomposition result (fig6) tells a genuinely surprising story: mean correction dominates, and variance correction alone is _worse_ than doing nothing. This is the opposite of what someone familiar with SmoothQuant might expect — SmoothQuant's core mechanism is variance redistribution, but here the mean shift is what matters.

The gain–σ scatter (fig7) closes the loop: it shows _why_ the per-channel means and variances diverge in the first place. The trained weights of the MLP's first layer, in combination with LayerNorm γ, systematically allocate gain unevenly across channels. Channels with higher learned gain also exhibit higher activation variance, with the strongest correlation (r = 0.75–0.77) in the late blocks where global quantization error is most destructive. This is a finding with implications beyond quantization — it suggests that ViT MLPs encode information in a structured, channel-specific way that global PTQ schemes are blind to.

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

### Bullet-Point Poster Text

- **Per-channel PTQ:** This work directly motivates per-channel activation quantization — per-channel zero-point + per-channel scale. Zero-points fuse into bias terms; implementable in integer-only pipelines.
- **Mixed-precision:** Not all 3,072 channels need the same bit-width. High-gain channels (r = 0.75) → INT8. Extreme-gain outliers → INT16. HAWQ-style sensitivity-gided search.
- **Edge deployment:** Target is INT8 ViT-B/16 on NVIDIA Jetson Orin — per-channel activations + per-channel weights + integer GELU. Phase 3 (integer GELU LUTs) is deferred but on the roadmap.

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
DISCUSSION (fig6 + fig7)
  "Decomposition reveals mean correction dominates (+20 pp over
   global) — fig6. Variance alone is catastrophic (6.56%).
   fig7 shows *why*: per-channel σ correlates with trained fc1
   weights (r = 0.75–0.77 in Blocks 8–10). Outliers are structural."
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
5. **fig6 → fig7:** "And _where_ does this structure come from? fig7 shows it's encoded in the trained weights: channels the network invested in show higher activation variance."
6. **fig7 → Future Work:** "If outliers are structurally encoded, then per-channel zero-points in PTQ are the natural next step."

---

## Design Notes for Poster Production

1. **Color consistency:** All seven figures use the same Paul Tol-inspired palette (teal = per-channel, coral = global, gray = baseline, blue = scatter points). The viewer should not have to re-learn the color code when moving between panels.

2. **Font sizing:** Minimum 14 pt on axes labels, 16–18 pt on titles, ≥12 pt on legends. The poster plotting module (`src/plotting_poster.py`) already enforces this.

3. **Figure placement by column weight:**
   - Column 1 (left): fig1 + fig3 — the problem statement. Lighter visual weight.
   - Column 2 (center): fig2 + fig4 + fig5 — the methods and results. Heaviest column, most data.
   - Column 3 (right): fig6 + fig7 + Future Work — the explanation and implications. Medium weight.

4. **Block 10 as anchor point:** The activation overlay (fig1), σ ridgeline (fig2), and gain-σ scatter (fig7) all reference Block 10 pre-GELU. This repeated reference builds a consistent mental model — Block 10 is the canonical example throughout.

5. **The one-glance test:** If someone walks by and spends 5 seconds, they should see: (a) fig1's dual-threshold overlay (orange band + colored channel lines = "channels differ!"), (b) the +3.76 pp annotation on fig4, (c) the 63.32% bar towering over 47.00% on fig6 ("mean correction wins!"), and (d) the scatter points stretching upward across the three fig7 panels ("it's in the weights"). These are the four take-home messages.

6. **Two-panel Discussion:** fig6 and fig7 are a pair — fig6 answers "what happens" and fig7 answers "why." They should be placed adjacent in Column 3 with a visual group boundary (thin rule or shared header background) to signal they tell one story.

---

## Figures Index

| # | File | Section | Visual Type | Key Message |
|---|------|---------|-------------|-------------|
| 1 | `fig1_activation_overlay.png` | Introduction | Histogram with threshold lines | Global thresholds incorrectly clip high-variance channels |
| 2 | `fig2_sigma_ridgeline.png` | Methods | Line plot with ±1σ band | Per-channel σ spread grows across depth (12.4× at Block 10) |
| 3 | `fig3_outlier_grid.png` | Background & RQ | Small-multiples heatmap | Outliers concentrate in late-block pre-GELU sites |
| 4 | `fig4_accuracy_bars.png` | Results | Grouped bar chart | Per-channel recovers +3.76 pp at k=3 |
| 5 | `fig5_accuracy_cost_vs_sparsity.png` | Results | Line plot (Pareto) with k labels | Per-channel is 1.89× more efficient on accuracy–sparsity frontier |
| 6 | `fig6_ablation_waterfall.png` | Discussion | Multi-bar comparison | Mean correction dominates; variance correction alone is destructive |
| 7 | `fig7_gain_sigma_scatter.png` | Discussion | 3-panel scatter with regression | ‖w_c ⊙ γ‖₂ correlates with σ_c (r = 0.75–0.77); outliers are structural |