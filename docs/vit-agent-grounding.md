# Master Research & Architecture Specification: ViT-B/16 PTQ for Edge Deployment

## 1. AGENT SYSTEM DIRECTIVES (STRICT COMPLIANCE REQUIRED)
You are an expert Python systems and machine learning engineer assisting in the development of a senior thesis codebase. Your primary objective is to generate code that is mathematically rigorous, highly optimized for VRAM constraints, and strictly typed. 

When generating code based on this document, you **MUST** adhere to the following rules:
1. **STRICT TYPE HINTING:** Every function signature, variable, and class attribute must use Python 3.10+ type hints.
2. **NO NAKED DICTIONARIES:** Never use dynamic dictionaries for state tracking or data passing. Use `@dataclass` or `pydantic` models exclusively.
3. **EXPLICIT RETURNS:** Functions must explicitly define what they return.
4. **MEMORY SAFETY (OOM PREVENTION):** The target hardware for this summer phase is an RTX 3070 with 8GB of VRAM. When writing PyTorch forward hooks, you **MUST** calculate statistics on-the-fly and immediately `detach()` or delete the raw activation tensors. Never append raw high-dimensional tensors to a global list.
5. **MODULARITY:** Do not duplicate code. Build reusable mathematical functions in a `src/` directory and keep execution scripts (`run_expX.py`) lightweight.

---

## 2. PROJECT CONTEXT & THEORETICAL GROUNDING

### The Core Goal
Investigate the hardware efficiency (Energy-Delay Product or EDP) and accuracy tradeoffs of Post-Training Quantization (PTQ) on a Vision Transformer (`ViT-B/16`). Ultimately, this will be deployed on a power-constrained NVIDIA Jetson Orin Nano, but the current summer phase is strictly software-based simulation and mapping on an x86 GPU.

### The Problem (The Friction Point)
Standard INT8 execution (via cuBLAS) is highly efficient but ruins accuracy due to activation outliers. The `LLM.int8()` mixed-precision decomposition solves this by routing outliers into FP16 and computing the rest in INT8.
* **The Flaw:** `LLM.int8()` assumes outliers are **sparse** and **channel-persistent** (a few specific feature columns are always large), as seen in Large Language Models.
* **Our Hypothesis:** Preliminary data shows `ViT-B/16` has **dense** outliers (up to 63% in late blocks), specifically in feedforward layers. If outlier density is too high, or if outliers scatter randomly instead of persisting in specific channels, the `LLM.int8()` routing premise breaks down. It will result in the majority of operations running unoptimized in FP16, destroying the EDP benefit of edge quantization.

### The Solution: Selective Routing (Local Transfer)
We are moving away from applying `LLM.int8()` globally. Instead, we aim to design a **heterogeneous (selective) routing policy**. We will use our experiments to identify which specific layers have the topological traits (sparse, persistent outliers) suited for mixed-precision routing, and which layers should either remain entirely in FP16 or be forced into pure INT8.

---

## 3. DIRECTORY ARCHITECTURE

The codebase must scale cleanly across four experiments. Enforce this structure:

vit_thesis_workspace/
│
├── data/                       # Git-ignored. ImageNet-1K validation split (4,096 imgs).
├── outputs/                    # Git-ignored. Dumps for JSONs, CSVs, and plots.
│   ├── exp1_outlier_maps/      
│   ├── exp2_granularity/       
│   ├── exp3_sensitivity/       
│   └── exp4_decomposition/     
│
├── src/                        # THE SHARED TOOLCHAIN
│   ├── __init__.py
│   ├── data_loader.py          # Strict-typed ImageNet loading & batching
│   ├── model_utils.py          # ViT loading, layer tagging (Attention vs MLP)
│   ├── hooks.py                # Hook logic & Dataclasses for on-the-fly math
│   ├── quantization.py         # Simulated INT8 math (per-tensor, per-channel)
│   └── visualizer.py           # Matplotlib/Seaborn heatmap and chart generators
│
├── run_exp1_mapping.py         # Driver script for Experiment 1
├── run_exp2_granularity.py     # Driver script for Experiment 2
├── run_exp3_sensitivity.py     # Driver script for Experiment 3
├── run_exp4_decomposition.py   # Driver script for Experiment 4

---

## 4. EXPERIMENT SPECIFICATIONS

### Experiment 1: Full Per-Layer Outlier Characterization (The Decision Engine)
**Goal:** Map the distribution, density, and persistence of activation outliers across every linear projection layer in `ViT-B/16` using 4,096 ImageNet validation images.

**Model implementation (IMPORTANT):** Use the `timm` implementation of ViT-B/16 (`vit_base_patch16_224`), **not** torchvision's. torchvision routes attention through a fused `nn.MultiheadAttention` kernel that bypasses the internal `out_proj` submodule and fuses Q/K/V into one opaque weight, so forward hooks can only observe the whole attention block's output (collapsing four projections into one measurement point, 37 hookable modules total). `timm` instead exposes every projection as a plain, independently hookable `nn.Linear`:
  * `blocks.N.attn.qkv` - fused Query/Key/Value projection,
  * `blocks.N.attn.proj` - attention output projection,
  * `blocks.N.mlp.fc1` / `blocks.N.mlp.fc2` - the two MLP linears,
  * `head` - the final classifier.

  This yields **49 separately hookable linear layers** (12 blocks x 4 in-block linears + 1 head), matching the goal of characterizing each projection independently. Q, K and V still share one fused `qkv` weight and cannot be split at the module level; if per-Q/K/V analysis is needed, chunk the `qkv` output tensor into three along its feature axis inside the hook.

**Implementation Details for Agent:**
* **Measure matmul INPUTS, not outputs (forward PRE-hooks).** Attach a forward *pre*-hook to every `nn.Linear` and characterize the activation **entering** each projection (`inputs[0]`) - the post-LayerNorm hidden state `X` that the GEMM consumes. For `Y = X @ W.T`, `LLM.int8()` inspects `X` and decides, per input feature column, whether to route to INT8 or FP16. Measuring `X` therefore captures the outliers at the **exact point where the routing decision is made**; measuring the output would characterize a different tensor the quantizer never routes on. (For ViT-B/16 the input width is 768 for `attn.qkv`/`attn.proj`/`mlp.fc1` and the `head`, and 3072 for `mlp.fc2`.)
* **Use a rigorous TWO-PASS algorithm (exactness over speed).** The fixed `|x| > 6.0` threshold needs no data statistics, but the statistical `3-sigma` threshold is defined relative to each feature channel's mean and standard deviation, which are only known after seeing every image. So run two passes over the (deterministic, unshuffled) data: **Pass 1** computes each layer's *exact* per-channel mean and std using the numerically-stable Chan/Welford parallel merge in float64 applied independently to each input feature dimension; **Pass 2** freezes those per-channel statistics and counts outliers/routing fractions for both thresholds. Reading the data twice is the deliberate price of an exact `3-sigma` cutoff rather than a per-batch approximation. Surface the exact per-channel `channel_means`/`channel_stds` (and derived `global_mean`/`global_std` aggregates) in the output so every threshold is auditable.
* **Statistical threshold is PER-CHANNEL, not global.** Different input feature channels have different activation distributions — a tight-variance channel and a wide-variance channel must not share the same 3-sigma bar. Pass 1 computes a separate mean and standard deviation for each of the 768 (or 3072 for `mlp.fc2`) feature channels. Pass 2 then flags a value in channel `c` as a statistical outlier when `|x - mean[c]| > 3 * std[c]`. This is the natural complement to `LLM.int8()`'s per-column routing decision and captures per-channel quantization sensitivity that a global threshold would obscure.
* **Layer Tagging:** Differentiate and tag layers based on topology: `Attention_QKV` (both `attn.qkv` and `attn.proj`) vs. `FeedForward_MLP` (both `mlp.fc1` and `mlp.fc2`).
* **Metrics to compute on-the-fly (inside the pre-hook), over the INPUT activation:**
    1.  **Maximum Magnitude:** The absolute maximum input value observed.
    2.  **Routing Fraction (PRIMARY, per-column - the true LLM.int8 cost):** `LLM.int8()` is a *structured* scheme - cuBLAS INT8 GEMMs force it to route **entire input feature columns** to FP16, never arbitrary scattered scalars. So the routing cost is the fraction of input feature **columns** that are *outlier columns*: columns exceeding the threshold in at least a minimum fraction of tokens. The participation bar differs by threshold:
        * **Fixed threshold (`|x| > 6.0`): 25% participation bar** — the faithful LLM.int8() criterion. This answers "what would LLM.int8() actually route?" and must match the paper exactly.
        * **Statistical threshold (`> 3 per-channel std`): 5% participation bar** — calibrated to ViT-B/16's ~1% per-channel outlier density. A 25% bar would flag zero columns on every layer (a true but uninformative result — ViT outliers are moderate and evenly distributed, unlike LLM outliers which are extreme and persistent). A 5% bar still requires persistence (~492K out of 9.85M tokens on a full validation run — far above a stray spike) but produces a meaningful per-layer signal.
    This per-column fraction is the actual share of the matmul's contraction dimension pushed to FP16.
    3.  **Per-Value Outlier Density (unstructured baseline):** Also record the fraction of *individual* input values exceeding each threshold (`magnitude > 6.0`, and the exact `> 3 per-channel standard deviations`) - the cost an idealized per-scalar scheme would pay. The **gap** between this and the matching per-column routing fraction quantifies the penalty of `LLM.int8()`'s whole-column constraint: a small gap means outliers are neatly column-aligned (routing works); a large gap means they are scattered and `LLM.int8()` over-routes. Reporting **both thresholds** for both the routing fraction and the per-value density makes the fixed-6.0-vs-3-sigma comparison direct.
    4.  **Channel Persistence (CRITICAL):** Calculate the variance of outlier locations across the **input feature dimension** (the columns `LLM.int8()` routes over). We must prove if outliers are localized to specific feature channels across tokens/images, or if they are scattered.

### Experiment 2: Accuracy by Quantization Granularity
**Goal:** Measure the Top-1 ImageNet accuracy of simulated INT8 implementations.

**Implementation Details for Agent:**
* Implement standard, naive simulated quantization (Fake Quantization: Quantize to INT8 -> Dequantize to FP32/16).
* Test and compare three scaling strategies:
    1.  **Per-tensor:** One scale factor for the entire weight/activation tensor.
    2.  **Per-channel (Weights):** One scale factor per output channel of the weight matrix.
    3.  **Per-token (Activations):** One scale factor per token in the activation sequence.

### Experiment 3: Per-Layer Sensitivity
**Goal:** Isolate the effect of quantizing specific layers to correlate accuracy drops with the outlier maps from Experiment 1.

**Implementation Details for Agent:**
* Iterate through the model sequentially.
* Apply simulated INT8 quantization to **one layer at a time** while leaving all other layers in full precision (FP32/16).
* Record the Top-1 accuracy drop.
* Output the data in a format easily readable by `visualizer.py` to plot a sensitivity heatmap.

**Limitation (residual error propagation):** This experiment measures *isolated* per-layer sensitivity — one quantized layer in an otherwise full-precision model. It does not capture the compound accumulation of quantization noise through residual connections across the 12 sequential blocks. A layer at block 10 may appear resilient in isolation but fail when the residual stream entering it carries quantization noise from blocks 1–9. Results should be interpreted as a layer's *intrinsic* sensitivity, not its sensitivity in a fully-quantized stack. The compound effect is measured in Experiments 2 and 4.

### Experiment 4: Simulated Outlier-Protecting Decomposition
**Goal:** Simulate the `LLM.int8()` mixed-precision logic strictly in software to see how much accuracy is recovered and measure the exact routing overhead.

**Implementation Details for Agent:**
* Implement a threshold-based mask. Values above the threshold are kept in exact higher precision; values below are quantized to INT8 and dequantized.
* Test using two distinct thresholds:
    1.  The standard `LLM.int8()` threshold (`magnitude = 6.0`).
    2.  A calculated threshold (`magnitude > 3 per-channel standard deviations from the channel's mean`).
* **Key Output:** For every layer, record the recovered Top-1 accuracy relative to naive INT8, and report the **High-Precision Fraction** (the exact percentage of values that bypassed quantization).

**Scope note (software simulation only):** This experiment is a software accuracy simulation. It does not model the memory bandwidth penalties of gathering scattered outlier columns on shared LPDDR5 memory — a critical bottleneck for the Jetson Orin Nano deployment target. Gathering non-contiguous outlier columns out of a sequential tensor for a separate FP16 kernel introduces severe memory access penalties that a pure arithmetic simulation cannot capture. The gap between simulated accuracy recovery and actual hardware Energy-Delay Product (EDP) will need to be measured in a future hardware-in-the-loop phase on the Jetson.

---

## 5. NEXT STEPS (For the Agent)
When instructed to begin coding, start by building the core `src/` modules. Focus first on `src/model_utils.py` to establish the layer tagging and `src/hooks.py` to define the strict `@dataclass` structures for Experiment 1's statistics tracking.