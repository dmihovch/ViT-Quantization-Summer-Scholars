## TL;DR

> **Citations:** All literature references in this document are catalogued with
> full bibliographic details in [`docs/CITATIONS.md`](CITATIONS.md).

Researchers mostly compute entropy on per-image attention maps or per-head activations and then aggregate (mean/sum) across batches or datasets depending on purpose. Entropy is typically computed on softmax-normalized attention distributions or via continuous/differential estimators, and the CLS token is usually treated explicitly (often analyzed via its attention distribution) rather than blindly pooled with patches.

----

## Entropy computation scope

Researchers compute entropy both at the single-sample level and as aggregated summaries depending on the use case. For analysis or pruning they compute per-sample or per-head entropies and then aggregate (e.g., average or sum) across images/heads to rank components; for training regularizers the per-sample entropy is incorporated into the batch loss and thus averaged during optimization.

- **Per-sample measures** Many works compute entropy per image (e.g., per attention map or per-token map) as the basic quantity before any aggregation for pruning or loss computation [1] [2].  
- **Aggregation for ranking or reporting** Entropies computed per-sample or per-head are commonly averaged or summed across the dataset or a validation set to decide which heads/tokens to compress or prune [3] [1].  
- **Batch-averaged loss usage** When entropy is used as a training regularizer it is evaluated per image and then averaged in the minibatch loss (i.e., included in the training objective) [2] [4].

----

## Entropy estimation methods

Different papers use different entropy estimators depending on whether inputs are attention weights, discrete bins, or continuous activations. The dominant pattern is to treat attention weights as probability distributions (softmax outputs) and apply Shannon entropy; for continuous hidden representations authors use differential-entropy estimators or explicit probabilistic models; some works adopt alternative entropy definitions (Rényi).

- **Attention as a probability distribution** Researchers compute Shannon entropy on attention maps by treating softmax-normalized attention weights across tokens as a probability vector and computing H = −∑ p log p per head or per query (for example to quantify head informativeness) [3] [5].  
- **Entropy across attention dimensions** Practical implementations sometimes compute entropy along a specific axis (e.g., key dimension) and then sum or average that per-map score to get a single scalar for the attention map [1].  
- **Spatial entropy regularizer** Some approaches formulate a spatial entropy on attention/feature maps and minimize that quantity as a self-supervised loss; this is computed per-image from the attention/spatial map and then added to the training loss (batch-averaged) [2].  
- **Differential entropy for continuous activations** When working with continuous pre-classification feature vectors, papers apply a differential-entropy regularizer (i.e., continuous-entropy formulation on activations) as part of metric or representation learning objectives [4].  
- **Probabilistic density modeling** Instead of directly computing H on raw vectors, some methods fit a probabilistic model over patch embeddings (e.g., variational/Bayesian models) and derive entropy or information measures from the learned distribution [6].  
- **Alternative entropy variants** Token-pruning work has explored Rényi-type entropies as token-importance metrics rather than classical Shannon entropy [7].  
- **On raw activations without normalization** There is no clear, general evidence in these papers that researchers simply apply Shannon entropy directly to raw, unnormalized hidden activations as a standard practice; instead they either normalize (attention softmax), use continuous/differential entropy formulations, or estimate densities/histograms via probabilistic models [3] [1] [2] [4] [6] [7].

----

## CLS token handling

The CLS token is frequently treated as a special signal rather than being indiscriminately grouped with patches; many works compute entropy of the CLS→patch attention distribution or analyze the CLS embedding separately.

- **Entropy of CLS attention distributions** Several works compute entropy of the attention distribution from the CLS token to patches (i.e., treat the CLS query’s softmax weights over patches as a probability vector) and use that entropy as a confidence or token-importance measure [5] [8].  
- **Using CLS for pruning or selection** Token-dropping/pruning methods commonly rely on CLS-attention maps (or CLS-derived scores) to guide which patch tokens to drop or keep, so CLS-attention entropy is computed and used directly in the decision rule [9] [8].  
- **CLS analyzed separately as a representation** Some defenses and distillation methods use the CLS embedding itself as a separate detector or distillation target (i.e., they do not fold CLS into patch-level entropy but inspect its own representation) [10] [11].  
- **Patch-focused probabilistic models typically exclude CLS** Methods that model patch embedding distributions or produce patch-level conceptual explanations focus on patch embeddings rather than the CLS token; those approaches treat patch distributions separately from the global CLS representation [6].

## References

[1]“Attention Map Guided Transformer Pruning for Edge Device,” Apr. 2023, doi: 10.48550/arxiv.2304.01452.

[2]E. Peruzzo et al., “Spatial Entropy as an Inductive Bias for Vision Transformers,” June 2022.

[3]L. Maisonnave, K. Haroun, and T. Pegeot, “Exploiting Information Redundancy in Attention Maps for Extreme Quantization of Vision Transformers,” arXiv.org, vol. abs/2508.16311, Aug. 2025, doi: 10.48550/arxiv.2508.16311.

[4]A. El-Nouby, N. Neverova, I. Laptev, and H. Jégou, “Training Vision Transformers for Image Retrieval”, doi: 10.48550/arxiv.2102.05644.

[5]Y. Mali, “AttenDence: Maximizing Attention Confidence for Test Time Adaptation,” arXiv.org, vol. abs/2511.18925, Nov. 2025, doi: 10.48550/arxiv.2511.18925.

[6]H. Wang, S. Tan, and H. Wang, “Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations   for Vision Foundation Models,” June 2024, doi: 10.48550/arxiv.2406.12649.

[7]W. Su, R. Zhang, and Z. Zhang, “R\’enyi Entropy: A New Token Pruning Metric for Vision Transformers”, [Online]. Available: https://arxiv.org/abs/2603.27900

[8]G. Lee and H. Kim, “AE-Guide: Attention and Entropy Guided Visual Token Dropping for Accelerating High-Resolution Vision-Language Models,” pp. 1–2, Oct. 2025, doi: 10.1109/isocc66390.2025.11329950.

[9]A. Yadav and P. Das, “GateAttn-ViT: Entropy-gated, attention-guided token pruning for resource-efficient vision transformer acceleration on FPGAs”, [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1383762126001542

[10]S. Sun, K. Nwodo, S. Sugrim, A. Stavrou, and H. Wang, “ViTGuard: Attention-aware Detection against Adversarial Examples for   Vision Transformer,” Sept. 2024, doi: 10.48550/arxiv.2409.13828.

[11]M. Kang, S. Son, and D. Kim, “Adaptive class token knowledge distillation for efficient vision transformer,” Knowledge Based Systems, Sept. 2024, doi: 10.1016/j.knosys.2024.112531.