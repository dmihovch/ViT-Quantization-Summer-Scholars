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

### Step 4: `src/hooks.py` — ~45 minutes

The core profiling machinery. This is the most important module for Phase 1.

- `register_profiling_hooks(model)` — iterate `model.named_modules()`, find
  `nn.GELU` instances, register a forward hook on each. The hook must:
  1. Accept `(module, input, output)` — you want `input[0]`, the pre-GELU tensor.
  2. Compute `max`, `min`, `std`, `mean` on the **flattened** tensor.
  3. Update a shared `dict[str, LayerStats]` — if the layer already has stats,
     update them with a running aggregate (Welford's algorithm or simple
     per-batch accumulation). If not, insert a new `LayerStats`.
  4. Return `HookHandle(handles=..., stats=...)`.
  5. Raise `HookRegistrationError` if zero GELU modules found.
- `remove_hooks(handle)` — `for h in handle.handles: h.remove()`.
- `save_stats(stats, path)` — `json.dump` with `dataclasses.asdict`.
- `load_stats(path)` — `json.load` and reconstruct `LayerStats` from dicts.

**Tests that will go green:** `test_hooks.py` (1 test)

---

### Step 5: `src/exp1_profiling.py` — ~30 minutes

Wire everything together for Phase 1.

- `run(config)` —
  1. `load_vit(config.device)` → model, transform.
  2. `build_val_loader(config.data_dir, transform, config.batch_size, config.num_images, config.device)`.
  3. `register_profiling_hooks(model)` → hook_handle.
  4. Loop over loader: `model(images.to(device))` inside `torch.no_grad()`.
  5. `remove_hooks(hook_handle)`.
  6. `ensure_dir(config.output_dir)`.
  7. `save_stats(hook_handle.stats, config.output_dir / "layer_stats.json")`.
  8. For each layer in stats: collect activations (you'll need to modify the
     hook to also store raw values, or do a second pass — see note below),
     call `plot_activation_histogram(...)`.

> **Design decision needed:** The current hook spec says "reduce to scalars
> immediately, no raw tensor storage." That's correct for stats, but histograms
> need raw values. Options:
> - **A:** Run two passes — one for stats, one for histogram sampling (simpler,
>   slower).
> - **B:** Add an optional `sample_every_n: int` parameter to the hook that
>   stores every Nth activation value in a buffer (more complex, faster).
>
> Start with option A. Optimize later if needed.

**Tests that will go green:** none directly, but you can now run
`python run_phase1_profiling.py --num-images 128` and see real output.

---

### Step 6: `src/plotting.py` — ~30 minutes

Four figure functions. Do this after Phase 1 runs so you have real data to test
with.

- `plot_activation_histogram(activations, layer_name, output_path, log_scale)` —
  `plt.hist(activations.flatten(), bins=100, log=log_scale)`, add vertical
  lines at ±3σ, save, close.
- `plot_accuracy_vs_threshold(results, output_path)` — group by sigma, plot
  line chart.
- `plot_pct_zeroed_per_layer(results, sigma_k, output_path)` — bar chart.
- `plot_lut_vs_fp32(lut, output_path)` — overlay FP32 GELU curve and LUT steps.

**Tests that will go green:** `test_plotting.py` (2 tests)

---

### Step 7: `src/ablation.py` — ~45 minutes

Phase 2 — outlier zeroing.

- `compute_pct_zeroed(tensor, threshold)` — `(tensor.abs() > threshold).float().mean() * 100`.
- `build_zeroing_hook(layer_name, threshold, stats)` — return a closure that
  takes `(module, args)`, zeros elements of `args[0]` where
  `|x| > threshold * stats.std`, returns modified tuple. Do NOT mutate in-place.
- `patch_model_for_ablation(model, sigma_k, layer_stats)` — iterate GELU
  modules, register pre-hooks, return handles.
- `save_ablation_results(results, path)` — write CSV with `csv.DictWriter`.

**Tests that will go green:** `test_ablation.py` (4 tests)

---

### Step 8: `src/exp2_ablation.py` — ~30 minutes

Wire Phase 2.

- `run(config)` —
  1. Load model + transform.
  2. `load_stats(config.layer_stats_path)`.
  3. For each `sigma_k` in `config.sigma_thresholds`:
     - `patch_model_for_ablation(model, sigma_k, stats)`.
     - `evaluate_accuracy(model, loader, device)`.
     - For each layer: `compute_pct_zeroed(...)` (you'll need to collect
       pre-GELU tensors during the eval pass — add a temporary hook).
     - Unpatch model.
     - Record `AblationResult` per layer.
  4. `save_ablation_results(...)`.
  5. Generate plots.

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
  > You are hooking into this architecture. Read Section 3.1 (the encoder block
  > diagram) so you know exactly where GELU sits: after the first Linear in the
  > MLP, before the second Linear. There are 2 GELU modules per block × 12
  > blocks = 24 hooks to register.

- **PyTorch Forward Hooks Tutorial**
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
  > The official docs for `register_forward_hook`. Pay attention to the
  > signature: `hook(module, input, output)` — `input` is a tuple, you want
  > `input[0]`. Also read the warning about modifying inputs in-place (don't).

### Before Step 7 (ablation)

- **Wei et al. (2022) — Outlier Suppression: Pushing the Limit of Low-bit
  Transformer Quantization**
  https://arxiv.org/abs/2209.13325
  > Directly studies what happens when you zero transformer activation outliers.
  > Read Section 3 (method) to understand the difference between zeroing,
  > clamping, and shifting. This gives you the vocabulary to discuss your
  > Phase 2 results.

- **Dettmers et al. (2022) — LLM.int8(): 8-bit Matrix Multiplication for
  Transformers at Scale**
  https://arxiv.org/abs/2208.07339
  > Section 3 shows that outlier features are systematic (same channels, across
  > inputs) and tied to model scale. After reading this, you will know what
  > "dominant channels" look like and can check whether your Phase 1 data shows
  > the same pattern.

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