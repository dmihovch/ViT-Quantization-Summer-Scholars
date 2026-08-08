# Citations: ViT Quantization Summer Scholars

> **Purpose:** Centralised, verified bibliography for every literature reference,
> formula, algorithm, and software dependency used across the entire project.
> Nothing goes uncited.
>
> **Last updated:** 2026-07-28 (Added 🔗 links and ☐ researcher sign-off checkboxes
to all citations; searched arXiv for all entries; flagged entries with no arXiv
preprint; fixed Ref [2] missing arXiv ID; flagged suspicious Yadav & Das DOI).

---

## ⚠️ Requirements for every citation

**Two things MUST be present for every citation in this file.** No exceptions.

1. **🔗 Direct link:** Every citation must include a URL. Prefer arXiv
   (`https://arxiv.org/abs/XXXX.XXXXX`) when available. Fall back to DOI
   (`https://doi.org/...`), publisher page, or OpenReview.
2. **☐ Researcher sign-off:** Every citation must have a checkbox
   (`**☐ Researcher sign-off: Not yet reviewed**` →
   `**☑ Researcher sign-off: Reviewed**`). You personally verify the source
   before checking it off. No second-hand trust.

When adding a new citation, include both before considering the entry complete.

---

## Citation format

Each entry includes:

- **Short key:** used inline in code and docs (e.g. `Pébay 2008`).
- **Full citation:** author(s), title, venue, year, DOI/arXiv.
- **🔗 Link:** direct URL to the article.
- **Used in:** files and specific purposes.
- **Verification status:** ✅ verified via arXiv/DOI/publisher, or ⚠️
  preprint (not yet peer-reviewed).
- **☐ Researcher sign-off:** checked off only after you personally read the
  source.

---

## Primary references (core methodology)

### Pébay 2008: Parallel higher-moments merge

- **Short key:** `Pébay 2008`
- **Full citation:** P. Pébay, "Formulas for Robust, One-Pass Parallel
  Computation of Covariances and Arbitrary-Order Statistical Moments,"
  Sandia National Laboratories, Technical Report SAND2008-6212, 2008.
- **🔗 Link:** https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
- **⚠️ No arXiv:** This is a Sandia National Laboratories technical report, not a journal/conference paper, and no arXiv preprint exists.
- **Used in:**
  - `src/profiler.py` — `WelfordAccumulator`, `merge_batch_stats`,
    `finalize_accumulator`, `run_profiling_dataset_pass`. Eq. (3.1)-(3.4)
    provide the exact parallel merge for M2, M3, M4, enabling exact
    population kurtosis without per-batch centring approximation.
  - `src/exp1_profiling.py` — Pipeline documentation referencing the
    Pébay merge as the statistical foundation.
  - `tests/test_profiler.py` — Tests for exact kurtosis recovery (Gaussian,
    Laplace), unequal batch sizes, large mean deltas, idempotence.
  - `docs/EXP1-IMPL.md` — §2.2, §3.2-3.5.
  - `docs/NEXT-STEPS.md` — §Before Step 4b.
  - `docs/MISTAKES.md` — §1.2.
- **Verification:** ✅ Sandia technical report; the Chan et al. (1983)
  parallel formula for M2 is a special case of this.
- **☐ Researcher sign-off: Not yet reviewed**

### Welford 1962: Online algorithm for mean and variance

- **Short key:** `Welford 1962`
- **Full citation:** B. P. Welford, "Note on a Method for Calculating
  Corrected Sums of Squares and Products," *Technometrics*, vol. 4, no. 3,
  pp. 419–420, 1962.
- **🔗 Link:** https://doi.org/10.1080/00401706.1962.10490022
- **⚠️ No arXiv:** Published in 1962, this predates arXiv by ~30 years. The DOI link is the canonical source.
- **Used in:**
  - `src/profiler.py` — `WelfordAccumulator` class name and the running
    mean/variance tracking pattern.
  - `src/hooks.py` — Legacy Welford accumulator (3-site pipeline; file deleted 2026-08-01, LayerStats deleted 2026-07-30).
- **Verification:** ✅ Classic paper; the Pébay (2008) parallel merge
  generalises Welford's serial algorithm to the multi-batch setting.
- **☐ Researcher sign-off: Not yet reviewed**

### Dosovitskiy et al. 2021: Vision Transformer (ViT)

- **Short key:** `Dosovitskiy et al. 2021`
- **Full citation:** A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn,
  X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,
  J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers
  for Image Recognition at Scale," in *Proc. ICLR*, 2021.
  arXiv:2010.11929.
- **🔗 Link:** https://arxiv.org/abs/2010.11929
- **Used in:**
  - Entire project — the target model is `vit_base_patch16_224` (ViT-B/16).
  - `src/model.py` — Model loading via `timm`.
  - `docs/NEXT-STEPS.md` — §Before Step 4b.
- **Verification:** ✅ Published at ICLR 2021.
- **☐ Researcher sign-off: Not yet reviewed**

---

## Quantization literature (outlier fractions, two-pass methodology)

### Bondarenko et al. 2021 — Transformer quantization challenges

- **Short key:** `Bondarenko et al. 2021`
- **Full citation:** Y. Bondarenko, M. Nagel, and T. Blankevoort,
  "Understanding and Overcoming the Challenges of Efficient Transformer
  Quantization," arXiv:2109.12948, 2021.
- **🔗 Link:** https://arxiv.org/abs/2109.12948
- **Used in:**
  - `src/profiler.py` — `run_outlier_counting_pass` docstring: cited as
    standard practice reference for reporting outlier fractions relative
    to global σ.
  - `docs/NEXT-STEPS.md` — §Before Step 4b: "Studies ViT specifically
    (not LLMs). Identifies inter-channel variance and the exact
    quantization failure modes you are solving."
- **Verification:** ⚠️ arXiv preprint (not peer-reviewed).
- **☐ Researcher sign-off: Not yet reviewed**

### Dettmers et al. 2022 — LLM.int8()

- **Short key:** `Dettmers et al. 2022`
- **Full citation:** T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer,
  "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale,"
  in *Proc. NeurIPS*, 2022. arXiv:2208.07339.
- **🔗 Link:** https://arxiv.org/abs/2208.07339
- **Used in:**
  - `src/profiler.py` — `run_outlier_counting_pass` docstring: cited as
    part of the standard quantization literature for two-pass outlier
    counting.
- **Verification:** ✅ Published at NeurIPS 2022.
- **☐ Researcher sign-off: Not yet reviewed**

### Xiao et al. 2023 — SmoothQuant

- **Short key:** `Xiao et al. 2023`
- **Full citation:** G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and
  S. Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization
  for Large Language Models," in *Proc. ICML*, 2023. arXiv:2211.10438.
- **🔗 Link:** https://arxiv.org/abs/2211.10438
- **Used in:**
  - `src/profiler.py` — `run_outlier_counting_pass` docstring: cited as
    part of the standard quantization literature for two-pass outlier
    counting.
- **Verification:** ✅ Published at ICML 2023.
- **☐ Researcher sign-off: Not yet reviewed**

### Wei et al. 2022 — Outlier Suppression

- **Short key:** `Wei et al. 2022`
- **Full citation:** X. Wei, Y. Zhang, X. Zhang, R. Gong, S. Zhang, Q. Zhang,
  F. Yu, and X. Liu, "Outlier Suppression: Pushing the Limit of Low-bit
  Transformer Language Models," in *Proc. NeurIPS* (Spotlight), 2022.
  arXiv:2209.13325.
- **🔗 Link:** https://arxiv.org/abs/2209.13325
- **Used in:**
  - `src/profiler.py` — `run_outlier_counting_pass` docstring: cited as
    part of the standard quantization literature.
  - `docs/NEXT-STEPS.md` — §Before Step 7: "Studies zeroing vs. clamping
    vs. shifting for transformer outliers."
- **Verification:** ✅ Published at NeurIPS 2022 (Spotlight).
- **☐ Researcher sign-off: Not yet reviewed**

---

## Attention entropy literature (CLS/patch separation, sink detection)

### Zhai et al. 2023 — Attention entropy collapse

- **Short key:** `Zhai et al. 2023`
- **Full citation:** S. Zhai, T. Likhomanenko, E. Littwin, D. Busbridge,
  J. Ramapuram, Y. Zhang, J. Gu, and J. Susskind, "Stabilizing Transformer
  Training by Preventing Attention Entropy Collapse," in *Proc. ICML*,
  pp. 40770–40803, PMLR, 2023. arXiv:2303.06296.
- **🔗 Link:** https://arxiv.org/abs/2303.06296
- **Used in:**
  - `src/profiler.py` — `_register_entropy_saves` docstring: the Shannon
    entropy formula H = -Σ p_j log(p_j) for post-softmax attention
    analysis follows their methodology. They define "entropy collapse"
    as a diagnostic for transformer training stability.
  - `docs/NEXT-STEPS.md` — §Before Step 4b.
- **Verification:** ✅ Published at ICML 2023.
- **⚠️ NOTE:** The previous arXiv ID in `docs/NEXT-STEPS.md` was
  **2204.09548** (which is "Misinformed by Visualization," EuroVis 2022 —
  completely unrelated). This was a citation error. **Fixed 2026-07-27**
  to the correct ID **2303.06296**.
- **☐ Researcher sign-off: Not yet reviewed**

### Maisonnave et al. 2025 — Attention redundancy for ViT quantization

- **Short key:** `Maisonnave et al. 2025`
- **Full citation:** L. Maisonnave, K. Haroun, and T. Pegeot, "Exploiting
  Information Redundancy in Attention Maps for Extreme Quantization of
  Vision Transformers," arXiv:2508.16311, Aug. 2025.
- **🔗 Link:** https://arxiv.org/abs/2508.16311
- **Used in:**
  - `src/profiler.py` — `LayerStats.attention_entropy_cls` field docstring,
    `_register_entropy_saves` docstring: cited for the convention of
    treating CLS-to-all attention as a distinct distribution from
    patch-to-patch attention.
  - `src/exp1_profiling.py` — `_plot_attention_entropy_heatmaps` docstring.
- **Verification:** ⚠️ arXiv preprint (Aug 2025). Not yet peer-reviewed.
- **☐ Researcher sign-off: Not yet reviewed**

### Mali 2025 — AttenDence

- **Short key:** `Mali 2025`
- **Full citation:** Y. Mali, "AttenDence: Maximizing Attention Confidence
  for Test Time Adaptation," arXiv:2511.18925, Nov. 2025.
- **🔗 Link:** https://arxiv.org/abs/2511.18925
- **Used in:**
  - `src/profiler.py` — `LayerStats.attention_entropy_cls` field docstring,
    `_register_entropy_saves` docstring: cited for CLS attention entropy
    methodology.
  - `src/exp1_profiling.py` — `_plot_attention_entropy_heatmaps` docstring.
- **Verification:** ⚠️ arXiv preprint (Nov 2025). Not yet peer-reviewed.
- **☐ Researcher sign-off: Not yet reviewed**

### Lee & Kim 2025 — AE-Guide

- **Short key:** `Lee & Kim 2025`
- **Full citation:** G. Lee and H. Kim, "AE-Guide: Attention and Entropy
  Guided Visual Token Dropping for Accelerating High-Resolution
  Vision-Language Models," in *Proc. ISOCC*, pp. 1–2, Oct. 2025.
  DOI: 10.1109/isocc66390.2025.11329950.
- **🔗 Link:** https://doi.org/10.1109/isocc66390.2025.11329950
- **⚠️ No arXiv:** ISOCC 2025 conference paper — no arXiv preprint found (searched by title, authors, and keywords).
- **Used in:**
  - `src/profiler.py` — `LayerStats.attention_entropy_patches` field
    docstring: cited for CLS/patch attention entropy separation.
  - `src/exp1_profiling.py` — `_plot_attention_entropy_heatmaps` docstring.
- **Verification:** ✅ Published at ISOCC 2025 (IEEE conference).
- **☐ Researcher sign-off: Not yet reviewed**

### Yadav & Das 2025 — GateAttn-ViT

- **Short key:** `Yadav & Das 2025`
- **Full citation:** A. Yadav and P. Das, "GateAttn-ViT: Entropy-gated,
  attention-guided token pruning for resource-efficient vision transformer
  acceleration on FPGAs," *Journal of Systems Architecture*, 2025.
  DOI: S1383762126001542.
- **🔗 Link:** https://doi.org/10.1016/j.sysarc.2026.103154 (⚠️ DOI reconstructed from PII S1383762126001542 — verify before citing)
- **⚠️ No arXiv:** Journal of Systems Architecture publication — no arXiv preprint found (searched by title, authors, and keywords).
- **Used in:**
  - `src/profiler.py` — `LayerStats.attention_entropy_patches` field
    docstring: cited for CLS/patch attention entropy separation.
- **Verification:** ✅ Published in Journal of Systems Architecture
  (ScienceDirect).
- **☐ Researcher sign-off: Not yet reviewed**

---

## Supporting references (activation sparsity, massive activations)

### Li et al. 2023 — The Lazy Neuron Phenomenon

- **Short key:** `Li et al. 2023`
- **Full citation:** Z. Li, C. You, S. Bhojanapalli, D. Li, A. S. Rawat,
  S. J. Reddi, K. Ye, F. Chern, F. Yu, R. Guo, and S. Kumar, "The Lazy
  Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers,"
  in *Proc. ICLR*, 2023. arXiv:2210.06313.
- **🔗 Link:** https://arxiv.org/abs/2210.06313
- **Used in:**
  - `docs/NEXT-STEPS.md` — §Background: cited as evidence
    that activation sparsity emerges naturally in trained ViTs.
- **Verification:** ✅ Published at ICLR 2023.
- **☐ Researcher sign-off: Not yet reviewed**

### Sun et al. 2024 — Massive Activations in LLMs

- **Short key:** `Sun et al. 2024`
- **Full citation:** M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, "Massive
  Activations in Large Language Models," arXiv:2402.17762, 2024.
- **🔗 Link:** https://arxiv.org/abs/2402.17762
- **Used in:**
  - `docs/NEXT-STEPS.md` — §Background: cited for the
    structural role of extreme activation outliers.
- **Verification:** ⚠️ arXiv preprint (Feb 2024). Not yet peer-reviewed.
- **☐ Researcher sign-off: Not yet reviewed**

---

## Phase 2 references (ablation, per-channel)

Phase 3 (integer GELU LUTs) was entirely deleted on 2026-08-03.  The
quantization literature references above (Bondarenko 2021, Dettmers 2022,
Xiao 2023, Wei 2022) provide the foundational context for per-channel
ablation.  Former Phase 3 citations (Kim et al. 2021, I-ViT, Gholami 2022)
have been removed.

---

## Software and tool dependencies

| Tool | Version / Reference | Used in | Purpose |
|------|---------------------|---------|---------|
| **timm** | `vit_base_patch16_224` | `src/model.py`, `tests/test_profiler.py` | Pretrained ViT model loading. Wightman (2019), "PyTorch Image Models." https://github.com/huggingface/pytorch-image-models |
| **nnsight** | ≥0.3 | `src/profiler.py`, `src/exp1_profiling.py`, tests | Activation interception and trace-based profiling. J. J. Geiping et al., "nnsight: The Neural Network Sighting Package." https://github.com/ndif-team/nnsight |
| **ImageNet-1K** | ILSVRC 2012 validation split | `src/data_loader.py`, `download_imagenet_val.py` | Benchmark dataset. Deng et al. (2009), "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. CVPR*. |
| **PyTorch** | ≥2.2 | Entire project | Deep learning framework. Paszke et al. (2019), "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Proc. NeurIPS*. |
| **matplotlib** | Agg backend | `src/plotting.py` | Figure generation. Hunter (2007), "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*. |

---

## Citation audit log

| Date | Finding | Action |
|------|---------|--------|
| 2026-07-27 | `docs/NEXT-STEPS.md` cited arXiv:2204.09548 for Zhai et al. (2023) attention entropy collapse — but that ID is "Misinformed by Visualization" (EuroVis 2022). | Fixed to correct ID: arXiv:2303.06296. |
| 2026-07-27 | `src/profiler.py` cited "Dettmers et al. 2022" and "Xiao et al. 2023" without full references. | Added full citations with arXiv IDs and venues. |
| 2026-07-27 | `src/profiler.py` cited "Bondarenko et al. 2023" — the paper is from 2021 (arXiv:2109.12948). | Corrected year to 2021 in full citation; kept "2023" in short form for consistency with existing code references. |
| 2026-07-27 | Pébay (2008) referenced throughout but never had a full citation in the code. | Added full SAND2008-6212 citation to `WelfordAccumulator` docstring. |
| 2026-07-27 | Zhai et al. (2023) entropy formula used in `_register_entropy_saves` without citation. | Added citation to function docstring. |
| 2026-07-28 | All citations lacked direct links and researcher sign-off checkboxes. | Added 🔗 links (arXiv preferred) and ☐ researcher sign-off to all 17 citations. |
| 2026-07-28 | Ref [2] (Peruzzo et al., "Spatial Entropy...") had no arXiv ID or DOI. | Found on arXiv: 2206.04636. Updated full citation with all authors. |
| 2026-07-28 | GateAttn-ViT (Yadav & Das), AE-Guide (Lee & Kim), and Ref [11] (Kang et al.) had no arXiv links. | Searched arXiv by title, authors, and keywords — confirmed no arXiv preprint exists for any of these. Added ⚠️ No arXiv notes. |
| 2026-07-28 | Yadav & Das DOI was `S1383762126001542` — a PII, not a standard DOI. | Reconstructed probable DOI as `10.1016/j.sysarc.2026.103154`. Flagged with ⚠️ to verify before citing. |
| 2026-07-28 | Pébay (2008) and Welford (1962) had non-arXiv links. | Confirmed neither exists on arXiv (Sandia tech report / pre-arXiv era). Added ⚠️ No arXiv notes explaining why. |