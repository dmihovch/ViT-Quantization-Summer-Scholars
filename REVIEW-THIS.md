# Research Homework: Papers and Resources to Review

Each entry explains *why* you need it and *what you should learn from it*. This is not a reading list for general knowledge — every item is tied to a specific weakness identified in `PROBLEMS.md`. The problem IDs (P1–P17) cross-reference that document.

Work through the sections roughly in order. Sections 1–3 are load-bearing for your core claims. Sections 4–6 are required before you run experiments. Section 7 is background you need to sound credible in conversation.

---

## Section 1: Foundational Quantization Math
*Addresses: P5, P6, P7 (MSE objective, dyadic error, LUT error bounds)*

**Why you need this first:** You cannot defend your mathematical pipeline if you do not understand where the numbers come from. These two papers are the bedrock of everything in METHODOLOGIES.md §Phase B and §Phase C.

---

- **Jacob et al. (2018) — Quantization and Training of Neural Networks for Inference at Fixed-Point**
  https://arxiv.org/abs/1712.05877
  > **Why:** This is the paper your Phase B is built on. The dyadic decomposition $S \approx M_0 \cdot 2^{-p}$ comes directly from here.
  > **What to learn:** Read Section 2 in full. Understand how the zero-point $z$ and scale $S$ are derived from a floating-point range. Understand the full quantization round-trip: $q = \text{clip}(\lfloor x/S \rceil + z, \text{qmin}, \text{qmax})$ and the dequantization $\hat{x} = S(q - z)$. This is what $\hat{X}$ in your MSE objective actually means — you need to be able to write it out from memory.

---

- **Krishnamoorthi (2018) — Whitepaper: Quantizing Deep Convolutional Networks for Efficient Inference**
  https://arxiv.org/abs/1806.08342
  > **Why:** Google's practical companion to Jacob et al. Fills in the engineering gaps the paper leaves out.
  > **What to learn:** The difference between symmetric and asymmetric quantization, and why symmetric is preferred for activations after ReLU/GELU (which are non-negative or near-zero). Understand per-channel vs. per-tensor scale factors and why per-channel matters for weights but is harder for activations. This directly informs whether your MSE sweep in Phase C should be per-tensor or per-channel.

---

- **Gholami et al. (2021) — A Survey of Quantization Methods for Efficient Neural Network Inference**
  https://arxiv.org/abs/2103.13630
  > **Why:** This is the best single survey in the field. Read it before you read anything else if you want context for every other paper on this list.
  > **What to learn:** Read Sections 2 and 3 carefully. Section 2 gives you the vocabulary for quantization error (granular distortion, overload distortion — these are the terms you need to replace "zero approximation error" with something accurate). Section 3 covers PTQ pipelines end to end. After reading this, you will be able to situate your work in the landscape of prior methods.

---

- **Gray & Neuhoff (1998) — Quantization (IEEE Transactions on Information Theory)**
  https://ieeexplore.ieee.org/document/720541
  > **Why:** This is the foundational mathematical reference for quantization error theory. It is dense, but Sections II and III give you the tools to make precise claims about your LUT output error (P7).
  > **What to learn:** The granular distortion (rounding error within the quantization range) vs. overload distortion (clipping error past the range) distinction. For your LUT, the output quantization error is granular distortion bounded by $S_{out}/2$ per activation. After reading this you will be able to replace the false "zero error" claim with a technically correct bound.

---

## Section 2: MSE-Optimized Clipping
*Addresses: P5, P8 (underspecified MSE objective, Gaussian assumption)*

**Why you need this:** Your Phase C is a rediscovery of ACIQ. You need to know what that paper already proved — both to avoid reinventing the wheel and to position your work correctly ("we apply ACIQ-style calibration to ViT FFN activations, which exhibit non-Gaussian distributions, and compare it to zero-strip suppression").

---

- **Banner et al. (2019) — Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment (ACIQ)**
  https://arxiv.org/abs/1810.05723
  > **Why:** Derives the closed-form optimal clipping threshold for Gaussian and Laplacian distributions. Your MSE sweep is a numerical version of what ACIQ does analytically.
  > **What to learn:** How the optimal $T^*$ is derived as a function of bit-width and distribution shape. Read Section 3. Understand why the optimal threshold is different for Gaussian vs. Laplacian distributions — this is directly relevant because ViT activations may be neither, and your kurtosis measurements from Phase 1 will tell you which distribution family is closer. After reading this, you can state whether your MSE sweep is doing more or less than ACIQ, and why.

---

- **Nagel et al. (2021) — A White Paper on Neural Network Quantization**
  https://arxiv.org/abs/2106.08295
  > **Why:** The most complete single reference for PTQ pipelines. Ties together everything in Sections 1 and 2.
  > **What to learn:** Read Section 3 (PTQ pipeline) and Section 4.1 (range estimation). Understand the difference between min-max calibration, percentile calibration, and MSE calibration — these are the three choices you are implicitly making in your Phase C, and a reviewer will ask you why you chose MSE over the alternatives. Also read the discussion of cross-layer dependency, which foreshadows why sequential independent optimization (what you are doing) may be suboptimal.

---

- **Nagel et al. (2020) — Up or Down? Adaptive Rounding for Post-Training Quantization (AdaRound)**
  https://arxiv.org/abs/2004.10568
  > **Why:** Shows that naive rounding to nearest is suboptimal and that the rounding decision should be learned. You are using nearest-rounding in your LUT construction. This paper gives you the vocabulary to acknowledge that limitation.
  > **What to learn:** The core insight: rounding one weight up while rounding another down can reduce the overall task loss even if it increases per-weight quantization error. After reading this, you will understand why your LUT construction (which always rounds to nearest) may not be optimal, and how to frame that as a limitation.

---

## Section 3: Integer-Only Transformers — The State of the Art
*Addresses: P1, P4, P11 (100% integer-only overclaim, no novelty claim, no external baseline)*

**Why you need this:** Before you claim anything as novel or call your pipeline "100% integer-only," you need to know exactly what prior work already achieved. These papers are the direct predecessors of your work. A reviewer will ask you how you differ from I-BERT on the first question they ask.

---

- **Kim et al. (2021) — I-BERT: Integer-only BERT Quantization**
  https://arxiv.org/abs/2101.01321
  > **Why:** The primary reference for integer-only transformer inference. This is the work you need to position against most explicitly.
  > **What to learn:** How they handle GELU (polynomial approximation), Softmax (integer-only exp approximation via bit-shifting), and LayerNorm (integer-only Newton-Raphson for reciprocal square root). Compare their GELU solution (polynomial) against yours (LUT) and write down three concrete tradeoffs. This is what your novelty section needs to articulate. Also note the accuracy results — these are your external baseline for P11.

---

- **Lin et al. (2022) — FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer**
  https://arxiv.org/abs/2111.13824
  > **Why:** Directly addresses ViT quantization (not BERT), which is your exact architecture. This is the closest prior work in the literature to what you are doing.
  > **What to learn:** How they quantize LayerNorm (Power-of-Two Factor) and Softmax (Log-Int Softmax). Read their accuracy results on ViT-B/16 specifically — those numbers are your external baseline (P11). After reading this paper, you should be able to write one paragraph explaining what you do differently and why that difference might matter.

---

- **Xiao et al. (2022) — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**
  https://arxiv.org/abs/2211.10438
  > **Why:** SmoothQuant is the state-of-the-art approach to the outlier problem you are directly attacking. Instead of clipping outliers, it *migrates* the quantization difficulty from activations to weights via a mathematically equivalent per-channel scaling.
  > **What to learn:** The core idea: if activations have large outliers in specific channels, you can absorb those outliers into the weight matrix by dividing the activation channel by a factor $s_j$ and multiplying the corresponding weight row by $s_j$. This is equivalent mathematically but moves the hard-to-quantize values to the weights (which are easier to quantize per-channel). After reading this, you need to understand why your approach (clipping) might be less accurate than SmoothQuant (migration), and whether that tradeoff is acceptable given your edge deployment context.

---

- **Yuan et al. (2022) — PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization**
  https://arxiv.org/abs/2111.12293
  > **Why:** Another direct prior work on ViT PTQ. Uses a "twin uniform quantizer" to handle the bimodal activation distributions in ViT attention layers.
  > **What to learn:** Read the activation distribution analysis in Section 3. This is empirical evidence for the distribution shapes you will observe in Phase 1-2. Compare their characterization of ViT distributions against your Phase 1 metrics (max, mean, variance, kurtosis) — you will need to reference papers like this when claiming that ViT FFN activations are heavy-tailed.

---

## Section 4: ViT-Specific Activation Distributions and Outliers
*Addresses: P8, P12 (Gaussian assumption, hook coverage verification)*

**Why you need this:** Your entire research program assumes that ViT pre-GELU activations are heavy-tailed and that outliers matter. You are not the first to look at these distributions. Read what others found before you write your own profiling code — it will tell you what to look for and save you from misinterpreting your own results.

---

- **Dettmers et al. (2022) — LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**
  https://arxiv.org/abs/2208.07339
  > **Why:** The paper that empirically demonstrated activation outliers in transformers are real, systematic, and tied to model scale. This is the foundational empirical grounding for your Phase 1 hypothesis.
  > **What to learn:** Section 3 shows that outlier features emerge suddenly at a certain model scale, appear in specific channels, and are consistent across inputs. Compare this against ViT-B/16's scale (~86M parameters) and ask: does the scale-dependent outlier emergence apply here? After reading this, you will know what "dominant channels" actually look like and whether your Phase 2 channel-wise variance mapping will find them.

---

- **Bondarenko et al. (2023) — Understanding and Overcoming the Challenges of Efficient Transformer Quantization**
  https://arxiv.org/abs/2109.12948
  > **Why:** Specifically studies ViT (not LLM) activation distributions and identifies the exact quantization failure modes you are building a solution for.
  > **What to learn:** Read the activation distribution analysis. Pay attention to the inter-channel variance findings — some channels have variance orders of magnitude higher than others. This tells you whether per-tensor or per-channel calibration is necessary and informs how you interpret kurtosis measurements from Phase 1. After reading this, you will have a concrete empirical prior for what your Phase 1 histograms should look like, and you will know which layers tend to be worst.

---

## Section 5: Outlier Suppression Strategies and Ablation Design
*Addresses: P2, P3, P9 (OVERVIEW vs. METHODOLOGIES contradiction, unfalsifiable hypothesis, calibration set size)*

**Why you need this:** Before you can resolve the contradiction between OVERVIEW and METHODOLOGIES on zero-strip vs. clamping, you need to understand what the literature says about each approach. These papers give you the theoretical and empirical grounding to make an informed, defensible design choice.

---

- **Wei et al. (2022) — Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Model Quantization**
  https://arxiv.org/abs/2209.13325
  > **Why:** Directly studies outlier suppression strategies in transformer quantization. This is the most relevant paper for resolving the zero-strip vs. clamp debate.
  > **What to learn:** What suppression strategies (zeroing vs. shifting vs. scaling) do to activation distributions and downstream accuracy. After reading this, you will have evidence-based grounds for choosing between Dr. Yang's zero-strip method and MSE clamping — or for running both as a fair ablation, which is what the current OVERVIEW calls for.

---

- **Hubara et al. (2021) — Accurate Post Training Quantization With Small Calibration Sets**
  https://arxiv.org/abs/2102.13630
  > **Why:** Studies how calibration set size affects PTQ quality. This directly answers why your 1,024-image calibration set may or may not be sufficient (P9).
  > **What to learn:** What minimum calibration set size their experiments suggest for stable scale factor estimation. Whether the answer scales with model depth (your 12-block ViT runs 24 separate calibration decisions). After reading this, you can either justify the 1,024 figure with a citation or add a small ablation over calibration set sizes (128, 512, 1024, 4096).

---

## Section 6: Statistical Rigor and Reproducibility
*Addresses: P3, P10, P15 (unfalsifiable hypothesis, single-seed variance, non-determinism)*

**Why you need this:** You are a novice researcher. The single most common mistake novice ML researchers make is treating a single evaluation number as a fact. These resources give you the minimum statistical toolkit to run defensible experiments.

---

- **Bouthillier et al. (2021) — Accounting for Variance in Machine Learning Benchmarks**
  https://arxiv.org/abs/2103.03098
  > **Why:** Empirically shows that reported benchmark results vary substantially across seeds, hardware, and library versions — even for deterministic-seeming evaluations like PTQ on a fixed dataset.
  > **What to learn:** What sources of variance exist even in "deterministic" evaluations. How to report results with confidence intervals. After reading this, you will understand why running your calibration with multiple seeds and reporting mean ± std is not optional — it is what separates a result from an anecdote.

---

- **Dodge et al. (2019) — Show Your Work: Improved Reporting of Experimental Results**
  https://arxiv.org/abs/1909.03004
  > **Why:** A short, accessible paper arguing that the ML community systematically under-reports variance and cherry-picks results. Written specifically for researchers new to rigorous evaluation.
  > **What to learn:** The concept of reporting a result distribution rather than a single number. How to state a hypothesis with a pre-specified effect size so the outcome can actually falsify something. This directly addresses P3 in one short read.

---

- **PyTorch Reproducibility Documentation**
  https://pytorch.org/docs/stable/notes/randomness.html
  > **Why:** The official reference for `torch.use_deterministic_algorithms`, `cudnn.deterministic`, and `cudnn.benchmark`. These are not optional for reproducible research on GPU.
  > **What to learn:** The difference between seeding (controls random number generation) and determinism (controls whether the same seed always produces the same result). Read the full page — it is short. After reading it, add `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic = True` to your `seed_everything` implementation.

---

## Section 7: ViT Architecture Internals
*Addresses: P1, P12 (LayerNorm/Softmax gap, hook coverage)*

**Why you need this:** You cannot claim integer-only execution and you cannot write correct hooks without understanding what operations are actually inside the model you are targeting.

---

- **Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)**
  https://arxiv.org/abs/2010.11929
  > **Why:** The original ViT paper. If you are quantizing this architecture, you must have read the paper that defines it.
  > **What to learn:** The exact sequence of operations in each transformer block: LayerNorm → Multi-Head Self-Attention → residual → LayerNorm → FFN (Linear → GELU → Linear) → residual. Write out every operation in one encoder block on paper. This is the map you will use to decide which operations are currently outside your integer pipeline (P1) and to verify your hooks cover all GELU instances (P12).

---

- **timm ViT source code — `vision_transformer.py`**
  https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
  > **Why:** The actual code your experiment runs on. Reading the paper is not enough — the implementation may differ from the paper (e.g., in how GELU is called, whether attention uses fused kernels, how LayerNorm is applied).
  > **What to learn:** Find the `Mlp` class and confirm that GELU is applied as `nn.GELU` (a module, not `F.gelu`). Count the number of GELU modules in a 12-block model — it should be 24 (2 per block). Find the `Attention` class and locate the Softmax. Find every LayerNorm. These are all the operations you need to account for in your integer pipeline claim.
