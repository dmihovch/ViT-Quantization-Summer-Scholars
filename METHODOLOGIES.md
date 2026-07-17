# PROJECT-DETAILS: Integer-Only Vision Transformer Quantization Pipeline

## 1. Architectural Objective and Bottleneck Analysis
Deploying Vision Transformers (ViTs) in resource-constrained edge environments is bottlenecked by the non-linear activation functions, specifically GELU. Standard Post-Training Quantization (PTQ) converts linear layer matrix multiplications to 8-bit integers (INT8). However, computing the non-linear GELU curve typically forces a de-quantization back to 16-bit or 32-bit floating-point (FP16/FP32). 

This fallback destroys the integer execution pipeline, causing severe hardware scheduling bottlenecks, increasing the Energy-Delay Product (EDP), and doubling memory bandwidth requirements for intermediate feature maps. Sequential application of independent compression techniques yields suboptimal results; therefore, a joint optimization approach is required. This pipeline establishes a mathematically rigorous, 100% integer-only inference architecture by resolving the activation scale-dependency trap.

---

## 2. Mathematical Pipeline Specification

### Phase A: Matrix Multiplication and the INT32 Trap
In a quantized transformer block, the input to the Feed-Forward Network (FFN) undergoes a linear projection. The hardware multiplies an INT8 weight matrix by an INT8 activation matrix. To prevent overflow, the hardware Arithmetic Logic Units (ALUs) accumulate these dot products into a 32-bit integer register.

Before these values can be passed into an 8-bit activation function, the INT32 tensor must be requantized down to INT8.

### Phase B: Dyadic Arithmetic Requantization
Standard requantization multiplies the INT32 values by a floating-point scale factor $S$. To maintain strict integer execution, this pipeline explicitly forbids floating-point multipliers at runtime. Instead, the scale factor $S$ must be decomposed offline into dyadic arithmetic: an integer multiplier $M_0$ and a right bit-shift $p$.

The floating-point scale is approximated as:
$$S \approx M_0 \cdot 2^{-p}$$

At runtime, the coding agent must implement the requantization from INT32 to INT8 using purely integer ALUs:
$$Y_{INT8} = \text{clip}\left( \lfloor (Y_{INT32} \cdot M_0) \gg p \rceil, -128, 127 \right)$$

This guarantees the tensor is mathematically scaled correctly and bounded to signed 8-bit space without triggering floating-point units.

### Phase C: MSE-Optimized Activation Clamping
ViT FFNs exhibit heavy-tailed activation distributions. Extreme outliers act as critical semantic routing signals. Destructively zeroing out these outliers eliminates network knowledge.

Instead, the pipeline mandates layer-wise saturation (clamping) optimized via Mean Squared Error (MSE). During offline calibration, the agent must sweep potential clipping thresholds $T$ to find the optimal boundary that restricts the dynamic range stretch while preserving information.

The optimization objective for the agent is to minimize the MSE between the unquantized FP32 tensor $X$ and the de-quantized clamped tensor $\hat{X}$:
$$\min_{T} \| X - \hat{X} \|_{2}^{2}$$

Once optimal $T$ is found, the scale factor $S$ for that specific layer is fixed as:
$$S = \frac{T}{127}$$

### Phase D: Scale-Aware GELU Lookup Tables (LUTs)
GELU is not scale-invariant: $GELU(S \cdot x) \neq S \cdot GELU(x)$. Implementing an integer polynomial approximation of GELU dynamically at runtime introduces compounding quantization noise and wastes compute cycles on complex fixed-point arithmetic.

Because the input to the activation layer has been strictly requantized to signed INT8, the tensor contains exactly 256 possible discrete states. The coding agent must pre-compute the exact floating-point GELU response for all 256 scaled inputs offline, quantize the outputs, and map them into a 1D Lookup Table (LUT).

The LUT generation formula to be implemented offline is:
$$LUT[x_q] = \left\lfloor \frac{GELU(x_q \cdot S_{in})}{S_{out}} \right\rceil$$

Where:
*   $x_q \in [-128, 127]$
*   $S_{in}$ is the specific scale factor of the tensor entering the GELU layer.
*   $S_{out}$ is the target scale factor for the subsequent linear projection layer.

At runtime, the non-linear activation layer executes zero arithmetic. It utilizes the incoming $Y_{INT8}$ value as an array index to fetch the pre-computed output:
$$Y_{activated} = LUT[Y_{INT8}]$$
This guarantees an $O(1)$ memory fetch with zero online approximation error.

---

## 3. Implementation and Profiling Directives

Coding agents must adhere to the following procedural constraints during implementation:

1.  **Strict Integer Emulation:** All runtime forward passes in the custom model class must utilize `torch.bitwise_right_shift` or equivalent integer operations. `torch.mul` with floating-point scalars is strictly prohibited during the forward pass of the quantized model.
2.  **Layer-Wise Granularity:** The LUTs and scale factors ($M_0$ and $p$) are not globally shared. They must be calculated and stored independently for every single transformer block, as activation distributions shift significantly depending on network depth.
3.  **Perturbation Profiling:** Agents must output a per-layer sensitivity profile. If aggressive clamping at the early embedding layers or final classification head results in catastrophic accuracy collapse during the calibration sweep, those specific layers must be flagged for alternative thresholding strategies while the robust middle layers utilize strict MSE clamping.
