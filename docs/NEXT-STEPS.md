# Next Steps: Implementation Roadmap

## Current State

Every module in `src/` has typed stubs with complete docstrings. Every test in
`tests/` has a concrete assertion waiting for real code. The conda environment
is installed and verified. **Nothing is implemented yet — 16 tests fail on
`NotImplementedError`, 6 pass (exception hierarchy + frozen config checks).**

The runner scripts (`run_phase1_profiling.py`, etc.) are wired and ready — they
parse args, build frozen configs, and call `run()`. They will work the moment
the stubs are filled in.

---

## Implementation Order

Work through these in sequence. Each step unlocks the next. Estimated effort is
for a novice researcher working evenings — adjust as you go.

### Step 1: `src/utils.py` — ~15 minutes

Three tiny functions with no ML logic. Do this first because every other module
calls these.

- `seed_everything(seed)` — set `random.seed`, `np.random.seed`,
  `torch.manual_seed`, `torch.cuda.manual_seed_all`, plus
  `torch.backends.cudnn.deterministic = True` and
  `torch.backends.cudnn.benchmark = False`.
- `get_device()` — `torch.device("cuda")` if `torch.cuda.is_available()` else
  `torch.device("cpu")`.
- `ensure_dir(path)` — `path.mkdir(parents=True, exist_ok=True)`.

**Tests that will go green:** `test_utils.py` (3 tests)

---

### Step 2: `src/model.py` — ~20 minutes

Two functions. This is the only module that touches `timm`.

- `load_vit(device)` — `timm.create_model("vit_base_patch16_224", pretrained=True)`,
  set eval mode, move to device, fetch transform via
  `timm.data.resolve_data_config` + `timm.data.create_transform`.
- `evaluate_accuracy(model, loader, device)` — standard top-1/top-5 loop inside
  `torch.no_grad()`. Use `outputs.topk(5)` and compare against labels.

**Tests that will go green:** none yet (model tests are `@pytest.mark.slow` and
not written — the existing `test_model.py` was deleted in the scaffold reset).

---

### Step 3: `src/data_loader.py` — ~15 minutes

One function. Straightforward `torchvision.datasets.ImageFolder` wrapper.

- `build_val_loader(data_dir, transform, batch_size, num_images, device)` —
  `ImageFolder(data_dir, transform=transform)`, optionally wrap in `Subset`,
  return `DataLoader` with `shuffle=False`, `pin_memory=(device.type == "cuda")`,
  `num_workers=4`. Raise `DataDirectoryError` if dir missing or empty.

**Tests that will go green:** `test_data_loader.py` (2 tests)

---

### Step 4: `src/hooks.py` — ~90 minutes

The core profiling machinery. This is the most important module for Phase 1.
You are now registering hooks at **five measurement sites**, not just pre-GELU.

#### Hook registration targets

| Site key | Module type to hook | Hook type | Tensor of interest |
|----------|--------------------|-----------|-----------------------|
| `pre_gelu` | `nn.GELU` | `register_forward_pre_hook` | `input[0]` — pre-GELU hidden states, shape `[B, N, D_mlp]` |
| `pre_softmax` | `nn.MultiheadAttention` | `register_forward_pre_hook` | The raw QKᵀ/√d logit tensor — accessible in PyTorch ≥ 2.0 via `need_weights=True` or by patching the `scaled_dot_product_attention` call |
| `post_softmax` | `nn.MultiheadAttention` | `register_forward_hook` | `output[1]` — the attention weight tensor returned when `need_weights=True`, shape `[B, N, N]` (averaged over heads unless `average_attn_weights=False`) |
| `post_layernorm` | `nn.LayerNorm` | `register_forward_hook` | `output` — the normalized tensor, shape `[B, N, D]` |
| `residual_stream` | The `nn.LayerNorm` that closes each encoder block (i.e., the second LN in the block) | `register_forward_pre_hook` | `input[0]` — the accumulated residual *before* the final LN, shape `[B, N, D]` |

> **Identifying the residual-stream hook target:** In `timm`'s ViT, each
> encoder block has `norm1` (pre-attention LN) and `norm2` (pre-MLP LN). The
> post-block residual is passed to the *next* block's `norm1`, or to the final
> head `norm`. Hook the input to the *next* block's `norm1` (or the head norm
> for the last block) to capture the accumulated residual stream after both
> sub-blocks.

#### Per-site statistics to compute inside the hook

Compute all of the following **per batch** and accumulate with Welford's
online algorithm (or simple running sum/sum-of-squares) to produce a single
aggregate value over the full dataset pass:

- **Scalar stats (all sites):** `max`, `min`, `mean`, `std`.
- **Kurtosis (all sites):** $\kappa = \mathbb{E}[(x-\mu)^4]/\sigma^4$. Use the
  fourth central moment formula; accumulate the first four moments online.
- **Outlier fractions (all sites):** percentage of elements where
  $|x| > k\sigma$ for $k \in \{3, 4, 6\}$. Store as a dict
  `{"3": float, "4": float, "6": float}`.
- **Per-channel σ vector (`post_layernorm` and `pre_gelu` sites only):**
  Compute `tensor.std(dim=(0, 1))` — standard deviation over batch and token
  dimensions, keeping the channel dimension. Shape: `[D]` or `[D_mlp]`. Store
  as a list of floats in `LayerStats`.
- **Attention entropy (`post_softmax` site only):** For each head,
  $H = -\sum_j p_j \log_2 p_j$ averaged over batch and query tokens. Store as
  a list of per-head mean entropies (length = number of heads).

#### `LayerStats` dataclass — extend it to hold the new fields

```python
@dataclass
class LayerStats:
    site: str                        # one of the five site keys above
    layer_name: str
    max: float
    min: float
    mean: float
    std: float
    kurtosis: float
    outlier_frac: dict[str, float]   # {"3": ..., "4": ..., "6": ...}
    per_channel_std: list[float] | None   # None for sites where not applicable
    attn_entropy: list[float] | None      # None for non-attention sites
    n_samples: int                   # total number of elements seen (for averaging)
```

#### Function signatures

- `register_profiling_hooks(model)` — iterate `model.named_modules()`,
  match each module type to its site key, register the appropriate hook type.
  Return `HookHandle(handles=..., stats=...)` where `stats` is
  `dict[str, LayerStats]` keyed by `"{layer_name}/{site}"`.
  Raise `HookRegistrationError` if zero hooks are registered.
- `remove_hooks(handle)` — `for h in handle.handles: h.remove()`.
- `save_stats(stats, path)` — `json.dump` with `dataclasses.asdict`.
- `load_stats(path)` — `json.load` and reconstruct `LayerStats` from dicts.

**Tests that will go green:** `test_hooks.py` (1 test, plus add new assertions
for the extra fields once you expand the test)

---

### Step 5: `src/exp1_profiling.py` — ~45 minutes

Wire everything together for Phase 1.

- `run(config)` —
  1. `load_vit(config.device)` → model, transform.
  2. `build_val_loader(config.data_dir, transform, config.batch_size, config.num_images, config.device)`.
  3. `register_profiling_hooks(model)` → hook_handle.
  4. Loop over loader: `model(images.to(device))` inside `torch.no_grad()`.
  5. `remove_hooks(hook_handle)`.
  6. `ensure_dir(config.output_dir)`.
  7. `save_stats(hook_handle.stats, config.output_dir / "layer_stats.json")`.
  8. For histograms, run a **second pass** with a sampling hook that stores
     every Nth element in a buffer (start with `N=10` to bound memory), then
     call `plot_activation_histogram(...)` per site.
  9. Call `plot_per_channel_std_heatmap(...)` for all `post_layernorm` and
     `pre_gelu` stats entries that have a non-`None` `per_channel_std`.
  10. Call `plot_attention_entropy_heatmap(...)` for all `post_softmax` stats
      entries.

> **Design note:** The hooks accumulate all final statistics in-hook (no raw
> tensors stored). Histograms require a second dedicated pass with a lightweight
> sampling hook that appends every Nth scalar to a `list[float]` buffer and
> clears it after saving. Keep the two passes separate so the stats pass remains
> deterministic and cheap.

**Tests that will go green:** none directly, but you can now run
`python run_phase1_profiling.py --num-images 128` and see real output.

---

### Step 6: `src/plotting.py` — ~45 minutes

Six figure functions. Do this after Phase 1 runs so you have real data to test
with.

- `plot_activation_histogram(activations, layer_name, site, output_path, log_scale)` —
  `plt.hist(activations.flatten(), bins=100, log=log_scale)`, add vertical
  lines at ±3σ and ±6σ, annotate the outlier fraction, save, close.
- `plot_per_channel_std_heatmap(per_channel_stds, layer_names, output_path)` —
  2-D heatmap with layers on the y-axis and channel index on the x-axis;
  cell value is per-channel σ. Use a diverging colormap so outlier channels
  (high σ) are visually distinct. Applies to `post_layernorm` and `pre_gelu`
  sites.
- `plot_attention_entropy_heatmap(entropies, layer_names, output_path)` —
  2-D heatmap with layers on the y-axis and head index on the x-axis; cell
  value is mean entropy in bits. Low-entropy cells flag attention sink heads.
  Applies to the `post_softmax` site.
- `plot_accuracy_vs_threshold(results, output_path)` — group by site and
  sigma, plot one line per site.
- `plot_pct_zeroed_per_layer(results, sigma_k, site, output_path)` — bar chart
  showing zeroed fraction per layer for a given site and threshold.
- `plot_lut_vs_fp32(lut, output_path)` — overlay FP32 GELU curve and LUT steps.

**Tests that will go green:** `test_plotting.py` (2 tests; add new smoke tests
for the heatmap functions)

---

### Step 7: `src/ablation.py` — ~60 minutes

Phase 2 — outlier zeroing across multiple sites.

- `compute_pct_zeroed(tensor, threshold)` — `(tensor.abs() > threshold).float().mean() * 100`.
- `build_zeroing_hook(layer_name, site, threshold, stats)` — return a closure
  that takes `(module, args)`, zeros elements of `args[0]` where
  `|x| > threshold * stats[f"{layer_name}/{site}"].std`, returns modified
  tuple. Do NOT mutate in-place. The `site` argument determines which
  `LayerStats` entry provides the reference σ.
- `patch_model_for_ablation(model, sigma_k, site, layer_stats)` — iterate
  modules matching the given site (GELU for `pre_gelu`, MHA for `pre_softmax`,
  second-block LN input for `residual_stream`), register the zeroing pre-hook
  for each, return handles.
- `compute_entropy_delta(stats_before, stats_after)` — given two
  `dict[str, LayerStats]` collected from a `post_softmax` hook (one before
  ablation, one after), return a per-head dict of entropy changes. This
  quantifies whether zeroing pre-softmax outliers disrupts attention routing.
- `save_ablation_results(results, path)` — write CSV with `csv.DictWriter`;
  include a `site` column so results from all three ablation sweeps live in
  one file.

**Tests that will go green:** `test_ablation.py` (4 tests; extend with tests
for the `site` parameter and `compute_entropy_delta`)

---

### Step 8: `src/exp2_ablation.py` — ~45 minutes

Wire Phase 2. Run three independent ablation sweeps — one per site.

- `run(config)` —
  1. Load model + transform.
  2. `load_stats(config.layer_stats_path)`.
  3. For each `site` in `("pre_gelu", "pre_softmax", "residual_stream")`:
     a. For each `sigma_k` in `config.sigma_thresholds`:
        - `patch_model_for_ablation(model, sigma_k, site, stats)`.
        - Collect a `post_softmax` stats snapshot (temporary hook, one pass)
          if `site == "pre_softmax"` so you can compute entropy delta.
        - `evaluate_accuracy(model, loader, device)`.
        - For each layer: `compute_pct_zeroed(...)` via a temporary hook
          during the eval pass.
        - Unpatch model.
        - Record `AblationResult(site=site, ...)` per layer.
     b. If `site == "pre_softmax"`: call `compute_entropy_delta(baseline_post_softmax_stats, ablated_post_softmax_stats)` and log it.
  4. `save_ablation_results(...)` (single CSV, `site` column distinguishes sweeps).
  5. Generate per-site accuracy-vs-threshold plots and per-layer zeroed-fraction bar charts.

---

### Step 9: `src/integer_gelu.py` — ~30 minutes

Phase 3 — LUT construction.

- `build_lut(scale_in, scale_out)` — loop `i` from 0 to 255, compute
  `x = (i - 128) * scale_in`, `y = GELU(x)` (use `torch.nn.functional.gelu`),
  `q = round(y / scale_out)`, clip to `[-128, 127]`.
- `apply_lut(tensor, lut)` — `tensor + 128` to shift indices, then index into
  `lut.lut` (convert to a tensor first).
- `compare_lut_vs_fp32(lut, scale_in)` — compute FP32 GELU for all 256 inputs,
  compare against LUT outputs, return `max_abs_error`, `mean_abs_error`, `rmse`.

**Tests that will go green:** `test_integer_gelu.py` (4 tests)

---

### Step 10: `src/exp3_integer_gelu.py` — ~20 minutes

Wire Phase 3.

- `run(config)` —
  1. `load_stats(config.layer_stats_path)`.
  2. For each layer: compute `scale_in = max_abs / 127`, `scale_out` (from
     GELU output range), `build_lut(...)`, `compare_lut_vs_fp32(...)`.
  3. Save comparison metrics JSON and LUT-vs-FP32 plots.

---

## What to Read (and When)

Read these **as you implement**, not all upfront. Each paper is tied to a
specific step so you learn the theory right before you need it.

### Before Step 4 (hooks + profiling)

- **Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words (ViT)**
  https://arxiv.org/abs/2010.11929
  > You are now hooking into five sites across this architecture, not just
  > GELU. Read Section 3.1 (the encoder block diagram) carefully: note both
  > MLP sub-blocks (where pre-GELU and hidden-state hooks go) and the
  > attention sub-block (where pre/post-softmax hooks go). Residual stream
  > hooks target the skip-connection accumulation points between sub-blocks.

- **PyTorch Forward Hooks Tutorial**
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
  > The official docs for `register_forward_hook` and
  > `register_forward_pre_hook`. Pay attention to the signature difference:
  > pre-hooks receive `(module, args)` while post-hooks receive
  > `(module, input, output)`. Several of your sites use pre-hooks (pre-GELU,
  > pre-softmax, residual stream).

- **Sun et al. (2023) — Massive Activations in Large Language Models**
  https://arxiv.org/abs/2402.17762
  > Describes "massive activation" outliers in attention and residual streams.
  > Although focused on LLMs, the measurement methodology applies directly:
  > they identify outliers per hidden-state dimension and track how they
  > propagate through residual streams. Use their per-channel magnitude
  > analysis as a template for your hidden-state dimension and residual-stream
  > measurement sites.

- **Zhai et al. (2023) — Stabilizing Transformer Training by Preventing
  Attention Entropy Collapse**
  https://arxiv.org/abs/2204.09548
  > Provides a definition and measurement methodology for attention entropy and
  > shows how near-zero entropy ("attention collapse") emerges during training.
  > Read the entropy measurement formula (Section 2) before implementing the
  > `post_softmax` hook — it is the exact quantity you will be computing and
  > plotting in your per-head entropy heatmap.

### Before Step 7 (ablation)

- **Wei et al. (2022) — Outlier Suppression: Pushing the Limit of Low-bit
  Transformer Quantization**
  https://arxiv.org/abs/2209.13325
  > Directly studies what happens when you zero transformer activation outliers.
  > Read Section 3 (method) to understand the difference between zeroing,
  > clamping, and shifting. This gives you the vocabulary to discuss your
  > Phase 2 results.

- **Bondarenko et al. (2023) — Understanding and Overcoming the Challenges of
  Efficient Transformer Quantization**
  https://arxiv.org/abs/2109.12948
  > Specifically studies **ViT** activation distributions (not LLMs). Identifies
  > the exact quantization failure modes you are building a solution for —
  > inter-channel variance, which layers are worst, and why standard PTQ fails
  > on vision transformers. Read the activation distribution analysis before
  > you interpret your Phase 1 histograms so you know what patterns to look for.
  > After reading, you will have a concrete empirical prior for what your
  > kurtosis measurements and channel-wise variance maps should reveal.

### Before Step 9 (integer GELU)

- **Kim et al. (2021) — I-BERT: Integer-only BERT Quantization**
  https://arxiv.org/abs/2101.01321
  > The primary reference for integer-only GELU. They use a polynomial
  > approximation; you are using a LUT. Read Section 3.2 (GELU approximation)
  > to understand the tradeoffs. After reading, you should be able to explain
  > why a LUT might be better for edge hardware (no polynomial compute at
  > runtime) and worse (256 bytes of storage per layer).

- **I-ViT: Integer-only Quantization for Efficient Vision Transformer
  Inference** (search on arXiv — the paper is referenced in your framework doc)
  > Introduces `ShiftGELU`, a bit-shift-based integer GELU approximation.
  > Compare this against your LUT approach. If `ShiftGELU` is simpler and
  > equally accurate, it may be a better fit for Jetson hardware.

### Background (read anytime)

- **Gholami et al. (2021) — A Survey of Quantization Methods for Efficient
  Neural Network Inference**
  https://arxiv.org/abs/2103.13630
  > The best single survey in the field. Read Sections 2–3 to get the
  > vocabulary right: symmetric vs. asymmetric quantization, per-tensor vs.
  > per-channel, granular vs. overload distortion. This is background — read it
  > when you have a free evening, not as a blocker.

---

## Quick Wins (Do These Now)

These take under 5 minutes each and will make the rest smoother:

1. **Run `conda init zsh`** in your terminal so `conda activate vitquant` works
   without the full path.
2. **Verify the env:** `conda activate vitquant && python -c "import torch; import timm; print('ok')"`.
3. **Run the 6 passing tests:** `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_exceptions.py tests/test_config.py -v`.
   Seeing green tests before you write any code is motivating.
4. **Download a tiny dataset:** `python download_imagenet_val.py --num-images 128`.
   You'll need real images to test Phase 1, and 128 images downloads in seconds.