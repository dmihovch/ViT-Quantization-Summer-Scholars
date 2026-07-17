# Experimental Protocol: Integer-Only Edge Deployment for Vision Transformers via Pre-GELU Activation Clipping

**Target Architecture:** Google Vision Transformer (`vit_base_patch16_224` via `timm`)
**Dataset:** ImageNet-1K (Validation Split for profiling/ablation)
**Overarching Objective:** Solve the non-linear activation bottleneck for ultra-low-bit Edge deployment by characterizing and suppressing heavy-tailed pre-GELU activation outliers, enabling the use of integer-only polynomial approximations without exacerbating quantization error.

## Phase 1: Instrumentation and Profiling

**Objective:** Intercept and record the exact tensor distributions immediately preceding the GELU operations across all transformer encoder blocks.
**Theoretical Context:** Current research demonstrates that activation distributions significantly impact efficiency[cite: 1]. Despite this, most methods focus heavily on weight and token redundancy, leaving activation quantization underexplored. We must establish a baseline of the raw feature maps before any compression is applied.

**Execution Steps:**
1. **Model Instantiation:** Load a pre-trained `vit_base_patch16_224` model in evaluation mode to disable dropout.
2. **Hook Registration:** Implement PyTorch forward hooks specifically targeting every `nn.GELU` module within the transformer's Feed-Forward Network (FFN). 
3. **Data Ingestion:** Pass a representative calibration subset of ImageNet-1K (e.g., 1024 images, stratified across classes) through the network.
4. **Metric Aggregation:** For every layer, calculate and store the following statistics for the intercepted tensors:
    * Absolute Maximum and Minimum
    * Mean
    * Variance and Standard Deviation
    * Kurtosis (to quantify the heaviness of the distribution tails)

## Phase 2: Outlier Characterization

**Objective:** Map the geometry of the activation bottlenecks to determine if the extreme values are systemic, layer-dependent, or isolated to specific dominant channels.
**Theoretical Context:** Extreme compression scenarios—sub-4-bit quantization, 90%+ sparsity, or their combination—receive limited attention[cite: 1]. To push into these ultra-low precision regimes, we must know exactly what data we are throwing away.

**Execution Steps:**
1. **Distribution Visualization:** Generate logarithmic-scale histograms of the aggregated pre-GELU tensors. Group these plots by early (layers 0-3), middle (layers 4-7), and late (layers 8-11) transformer blocks.
2. **Channel-Wise Variance Mapping:** Calculate the standard deviation strictly along the channel dimension. 
3. **Threshold Identification:** Identify the specific threshold at which the density of the distribution drops off (e.g., 3σ, 4σ, or an absolute magnitude scalar). Output a list of "dominant channels" that consistently produce values outside these thresholds.

## Phase 3: The Clipping Ablation Study

**Objective:** Measure the destructive impact of forcibly stripping activation outliers prior to the non-linear transformation. 
**Theoretical Context:** Pruning can exacerbate quantization error by removing redundancy that buffers against precision loss[cite: 1]. We must measure if Dr. Yang's hypothesis ("strip out outliers to 0") destroys the semantic signal or if the network is robust to the structural loss.

**Execution Steps:**
1. **Threshold Implementation:** Modify the forward pass to inject a hard clipping function immediately before the GELU activation.
2. **Ablation Sweep:** Run the full ImageNet-1K validation set through the model using four distinct clipping strategies:
    * **Baseline:** No clipping.
    * **3σ Clip:** Clamp all values outside 3σ to the 3σ boundary.
    * **2σ Clip:** Clamp all values outside 2σ to the 2σ boundary.
    * **Zero-Strip (Dr. Yang's Method):** Set any value outside the 3σ threshold strictly to 0.
3. **Performance Logging:** Record the Top-1 and Top-5 accuracy for each sweep. Log the exact percentage drop compared to the unclipped baseline. 

## Phase 4: Integer GELU Integration (Co-optimization)

**Objective:** Replace the floating-point transcendental GELU function with an integer-only bit-shift polynomial approximation, measuring the synergy between outlier suppression and integer execution.
**Theoretical Context:** Previous works, such as HeatViT, employed 8-bit fixed-point quantization with polynomial approximations for nonlinear functions[cite: 1]. However, sequential application of independent techniques yields suboptimal results compared to joint optimization frameworks[cite: 1]. We are testing the hypothesis that clipping (Phase 3) acts as a necessary prerequisite for stable integer approximation.

**Execution Steps:**
1. **Function Replacement:** Swap the PyTorch `nn.GELU` modules with a custom integer approximation layer. Use a piecewise quadratic or bit-shift polynomial that restricts all math to INT8 or INT16 arithmetic.
2. **Unclipped Integer Test:** Run the ImageNet-1K validation set using the integer GELU *without* clipping the outliers. Record the Top-1 accuracy. (This will likely crash the accuracy due to dynamic range overflow at the tails).
3. **Co-optimized Test:** Run the validation set using *both* the optimal clipping threshold discovered in Phase 3 and the integer GELU. 
4. **Final Comparison:** Calculate the accuracy delta between the Unclipped Integer Test and the Co-optimized Test. A higher accuracy in the Co-optimized Test empirically validates the core hypothesis of this research.