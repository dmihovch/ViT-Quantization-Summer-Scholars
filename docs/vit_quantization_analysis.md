# Research Evaluation: ViT-B/16 Post-Training Quantization

## 1. Codebase Architecture and Quality Evaluation
The repository establishes a standard, modular structure separating data ingestion (`src/data_loader.py`), instrumentation (`src/hooks.py`), model operations (`src/model_utils.py`), and visualization (`src/visualizer.py`). The inclusion of a dedicated `tests/` directory utilizing `pytest` demonstrates baseline software engineering competency, ensuring reproducibility of data pipelines and hook attachments.

**Weaknesses in Code Architecture:**
The current implementation is purely observational. `run_experiment1_mapping.py` maps activation distributions but lacks the computational graph modifications necessary to simulate quantization. There are no custom autograd functions or module wrappers to inject fake-quantization operations (e.g., `torch.fake_quantize_per_tensor_affine`). The codebase is currently a profiling tool, not a quantization framework.

## 2. Methodological and Mathematical Rigor
Experiment 1 successfully targets activation outliers, which are the primary driver of quantization error in Vision Transformers. The metrics generated (channel persistence, max magnitude, routing fractions, outlier density) are necessary empirical baselines. 

**Deficiencies in Rigor:**
The research currently stops at statistical characterization. The fundamental mathematical objective of Post-Training Quantization (PTQ) is minimizing the quantization error, typically defined by the $L_2$ distance or cosine similarity between the full-precision tensor $X$ and the dequantized tensor $\hat{X}$:

$$\min_{s, z} || X - \hat{X} ||_2^2 \quad 	ext{where} \quad \hat{X} = s \cdot (	ext{clip}(	ext{round}(X / s) + z, 0, 2^b - 1) - z)$$

The project lacks this error modeling. You are observing the outliers but not mathematically quantifying their destructive impact on the attention mechanism's information entropy when compressed to INT8 or INT4.

## 3. Literature Alignment and Deep Dive
The focus on outlier mapping aligns with foundational PTQ literature, but the interpretation must expand to address specific ViT pathologies.

* **LayerNorm Asymmetry:** Literature (e.g., PTQ4ViT, FQ-ViT) dictates that outliers in ViTs are heavily concentrated in specific channels post-LayerNorm. Standard uniform quantization fails here because the dynamic range is stretched by a few high-magnitude channels. Your channel persistence maps address this, but the subsequent step must be mathematical compensation—such as shifting the outlier channels or applying channel-wise scaling factors—before applying the quantization grid.
* **Softmax Power-Law Distribution:** The self-attention mechanism relies on Softmax, which produces a highly skewed, power-law distribution where most values are near zero and a few are near one. Uniformly distributing quantization bins across $[0, 1]$ wastes resolution. Literature suggests applying a $\log_2$ quantizer or a power-of-two non-uniform quantizer for post-Softmax activations to preserve the exact values of the dominant attention weights. Your research currently treats all tensors as candidates for standard evaluation; it must be bifurcated by operation type (Linear vs. Softmax).

## 4. Operational Inconsistencies
There is a fundamental disconnect between the stated objective (optimizing the Energy-Delay Product on edge hardware) and the current methodology (PyTorch FP32 hooks on desktop hardware).

$$EDP = 	ext{Energy} 	imes 	ext{Delay}$$

Statistical maps in PyTorch do not correlate directly to $EDP$. A low-bit quantization scheme that suppresses outliers might introduce complex asymmetric scaling operations that actually increase latency on edge hardware lacking native support for those specific arithmetic instructions.

## 5. Strategic Roadmap and Directives
To move this research from observation to actionable findings, the following steps are required:

1.  **Implement Fake-Quantization Nodes:** Develop module wrappers that simulate INT8/INT4 precision during the forward pass. This allows you to measure top-1 accuracy degradation on the ImageNet validation set.
2.  **Apply Operation-Specific Quantizers:** Implement uniform symmetric quantization for weights, uniform asymmetric quantization for post-LayerNorm activations, and non-uniform (e.g., $\log_2$) quantization for post-Softmax activations.
3.  **Hardware Compilation Pipeline:** PyTorch FP32 execution time is irrelevant for edge EDP. Transition the quantized models through a compiler toolchain (e.g., ONNX to TensorRT or TFLite) to profile actual latency and compute utilization on target architectures (e.g., ARM Cortex, Nvidia Jetson).
4.  **Mixed-Precision Allocation:** Utilize the nuclear norm or Hessian trace to determine the sensitivity of individual ViT blocks to quantization. Allocate bit-widths dynamically (e.g., INT8 for early layers, INT4 for deeper layers) rather than applying a blanket policy.
