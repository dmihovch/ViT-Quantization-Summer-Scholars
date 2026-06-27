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
**Goal:** Map the distribution, density, and persistence of activation outliers across all 48 linear projection layers in `ViT-B/16` using 4,096 ImageNet validation images.

**Implementation Details for Agent:**
* Use forward hooks attached to `nn.Linear` layers.
* **Layer Tagging:** Differentiate and tag layers based on topology: `Attention_QKV` vs. `FeedForward_MLP`.
* **Metrics to compute on-the-fly (inside the hook):**
    1.  **Maximum Magnitude:** The absolute maximum value observed.
    2.  **Outlier Density (Routing Fraction):** The percentage of values exceeding specific thresholds (e.g., `magnitude > 6.0`, and `> 3 standard deviations`). This represents the fraction of math that would be routed to FP16.
    3.  **Channel Persistence (CRITICAL):** Calculate the variance of outlier locations across the channel dimension. We must prove if outliers are localized to specific feature channels across tokens/images, or if they are scattered.

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

### Experiment 4: Simulated Outlier-Protecting Decomposition
**Goal:** Simulate the `LLM.int8()` mixed-precision logic strictly in software to see how much accuracy is recovered and measure the exact routing overhead.

**Implementation Details for Agent:**
* Implement a threshold-based mask. Values above the threshold are kept in exact higher precision; values below are quantized to INT8 and dequantized.
* Test using two distinct thresholds:
    1.  The standard `LLM.int8()` threshold (`magnitude = 6.0`).
    2.  A calculated threshold (`magnitude > 3 standard deviations from the layer's mean`).
* **Key Output:** For every layer, record the recovered Top-1 accuracy relative to naive INT8, and report the **High-Precision Fraction** (the exact percentage of values that bypassed quantization).

---

## 5. NEXT STEPS (For the Agent)
When instructed to begin coding, start by building the core `src/` modules. Focus first on `src/model_utils.py` to establish the layer tagging and `src/hooks.py` to define the strict `@dataclass` structures for Experiment 1's statistics tracking.
