# ViT Quantization & Outlier Profiling: Experimental Framework

## 1. Project Objective
Profile intermediate activations in an encoder-only Vision Transformer (ViT) to quantify the impact of massive outliers, evaluate outlier ablation, and establish a pathway for integer-only GELU non-linearities. The experiments are scoped pragmatically to prioritize measurable, executable outcomes suitable for edge deployment profiling on NVIDIA Jetson hardware.

## 2. Literature Review: Targeted Reading List
Focus exclusively on literature that supports the defined boundaries. Avoid overly theoretical frameworks that do not directly translate to hardware-friendly quantization.

### Core Outlier & Sparsity Mechanics
*   **The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers (arXiv:2210.06313)**
    *   *Focus:* Understanding the baseline for activation sparsity and how a minority of neurons dictate magnitude.
*   **Massive Activations in Large Language Models (arXiv:2402.17762)**
    *   *Focus:* The structural role of extreme outliers. Extract the methodology for measuring their impact without adopting LLM-scale assumptions for a ViT.

### Integer Non-linearities & ViT Quantization
*   **I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference**
    *   *Focus:* The `ShiftGELU` mechanism. This is critical for Dr. Yang’s requirement to execute GELU on integers, demonstrating how to approximate the floating-point operation using integer bit-shifting or Lookup Tables (LUTs).

### Special Attentions
*   *Search Query Parameters:* "Vision Transformer attention outlier mitigation", "Sparse attention activation distribution".
*   *Focus:* Briefly survey if alternative attention mechanisms (e.g., Swin windowed attention) naturally suppress massive activations compared to standard global Softmax attention. Keep this scoped to vision models.

## 3. Experimental Design

**Target Model:** `google/vit-base-patch16-224` (Encoder-only)

### Phase 1: Baseline Pre-GELU Profiling
Establish the unquantized ground truth of the network's activation distributions.
*   **Mechanism:** Register forward hooks immediately before all `nn.GELU` layers.
*   **Data Collection:** Extract flattened pre-GELU intermediate tensors for a subset of the validation dataset.
*   **Metrics:** Record the Max, Min, and Standard Deviation ($\sigma$) per layer.
*   **Deliverable:** Log-scale histograms visualizing the heavy-tailed distribution of pre-GELU activations to pinpoint exactly where the outliers sit relative to the bulk data.

### Phase 2: Outlier Ablation (Zeroing)
Test the structural reliance on massive activations by forcing them to zero.
*   **Mechanism:** Intercept pre-GELU activations and apply a hard mask: $x = 0$ if $|x| > 	au$.
*   **Thresholding:** Define $	au$ dynamically using a scalar multiple of the layer's standard deviation (e.g., $	au = 3\sigma$, $	au = 4\sigma$). 
*   **Metrics:** 
    *   Calculate the percentage of zeroed parameters per layer.
    *   Measure top-1 validation accuracy at each threshold step.
*   **Deliverable:** A degradation curve showing model accuracy vs. outlier ablation percentage to definitively map the breaking point of the attention distribution.

### Phase 3: Integer GELU Exploration
Reconcile the bounded distribution (post-ablation) into an integer space.
*   **Mechanism:** Utilize the profiling metrics from Phase 1 & 2 to define a symmetric or asymmetric dynamic range for INT8 or INT4 quantization.
*   **Implementation:** Construct a piece-wise linear approximation or a Lookup Table (LUT) that accepts the integer bounds and performs the GELU transformation without dequantizing to FP32/FP16.
*   **Metrics:** Compare the output distribution of the Integer GELU against the FP32 GELU outputs. 

## 4. Hardware Profiling Considerations
*   Ensure the custom profiling hooks and masking operations do not introduce artifact overhead that skews subsequent energy-delay product (EDP) measurements when transitioning the optimized model to the target hardware constraints.
