# Experiment 2: Accuracy by Quantization Granularity — Full Plan

**Status:** Planning complete — implementation in progress  
**Last updated:** 2026-07-05

---

## 1. Objective

Measure the Top-1 ImageNet accuracy of a **fully INT8-quantized ViT-B/16** under
four different weight × activation scaling granularities. This establishes the
"naive INT8" baseline that Experiment 4's mixed-precision decomposition must
beat. It also records **per-layer quantization error (MSE)** so we can correlate
quantization damage with the outlier maps from Experiment 1 and the sensitivity
results from Experiment 3.

---

## 2. The Four Configurations

| # | Config Name | Weight Quant | Activation Quant | Hypothesis |
|---|------------|-------------|-----------------|------------|
| **A** | `per_tensor` | per-tensor | per-tensor | Simplest possible INT8. The accuracy floor. |
| **B** | `per_channel_weights` | per-channel | per-tensor | Finer weight representation recovers some accuracy, but coarse activations bottleneck it. |
| **C** | `per_token_activations` | per-tensor | per-token | Per-token scaling adapts to token-by-token scale variation (helps blocks 9–10), but outlier columns still destroy weight precision. |
| **D** | `per_channel_per_token` | per-channel | per-token | Best naive INT8. The ceiling for non-mixed-precision quantization. If this is still bad, mixed-precision is mandatory. |

Config D was added per discussion on 2026-07-05 as the natural fourth cell in
the 2×2 grid.

### Dynamic (per-batch) activation scaling

We compute activation quantization scales from the **current batch** at runtime
rather than using a static calibration set. This is the most optimistic case —
if accuracy is bad even with per-batch adaptation, it will be worse with static
scales. This gives us a *lower bound* on accuracy loss, which is the right
baseline to beat.

---

## 3. Per-Layer Quantization Error (MSE)

For every config, we record the **mean squared error** between the FP32
activation entering each `nn.Linear` and its quantized counterpart:

```
MSE_layer = mean((X_fp32 - X_int8)²)
```

This is accumulated across all batches (weighted by batch size for a correct
global mean) and saved per-layer. The per-layer MSE lets us:

1. **Correlate with Experiment 1:** Do layers with high outlier density also
   show high quantization MSE?
2. **Correlate with Experiment 3:** Do layers that cause large accuracy drops
   when quantized in isolation also show high MSE?
3. **Compare granularities:** Does per-token scaling reduce MSE in blocks 9–10
   relative to per-tensor scaling?

### Implementation

A lightweight `MSEAccumulator` class tracks per-layer `(sum_sq_error, count)`
and computes the weighted mean at the end. The activation quant hook records
MSE on every forward pass. Both tensors are `detach()`ed before recording to
avoid graph entanglement (the entire evaluation runs under `torch.no_grad()`
regardless).

---

## 4. Three-Pass Evaluation

Each config is evaluated **3 times** and results are reported as **mean ± std**.
This serves two purposes:

1. **Measurement stability:** Confirms that the evaluation is deterministic
   (std ≈ 0 with fixed data ordering and eval mode).
2. **Future-proofing:** If we later add stochastic elements (random subset
   sampling, calibration-set randomization), the infrastructure is already in
   place.

With the current deterministic data loader and `model.eval()` mode, all three
passes are expected to produce identical results (std = 0). This is not a bug —
it confirms the measurement is noise-free.

---

## 5. Implementation Plan

### 5.1 `src/quantization.py` — New functions

```python
def quantize_all_weights(
    model: nn.Module,
    strategy: str,
    linear_layers: dict[str, nn.Linear] | None = None,
) -> dict[str, torch.Tensor]:
    """Quantize every nn.Linear weight in-place. Returns {name: original_weight}."""

def restore_weights(
    model: nn.Module,
    originals: dict[str, torch.Tensor],
) -> None:
    """Restore original weights from the dict returned by quantize_all_weights."""

def make_activation_quant_hook(
    strategy: str,
    layer_name: str,
    mse_tracker: MSEAccumulator | None = None,
) -> Callable:
    """Returns a forward pre-hook that quantizes the input activation.

    For per-token strategy on 2D inputs (e.g., the classifier head after
    pooling), falls back to per-tensor quantization gracefully.
    """

class MSEAccumulator:
    """Tracks per-layer MSE between original and quantized activations."""
    def record(self, name: str, original: Tensor, quantized: Tensor) -> None: ...
    def get_all(self) -> dict[str, float]: ...
```

### 5.2 `run_exp2_granularity.py` — Driver script

Follows the lightweight pattern established by `run_experiment3_sensitivity.py`:

```
1. Parse CLI (--num-images, --batch-size, --data-dir, --output-dir, --num-runs)
2. Load ViT-B/16 model + ImageNet validation loader
3. Evaluate baseline FP32 Top-1 accuracy
4. For each of the 4 configs:
   a. For each of N runs (default 3):
      i.   quantize_all_weights(model, weight_strategy)
      ii.  Register activation quant hooks on all nn.Linear layers
           (with MSEAccumulator attached)
      iii. Evaluate Top-1 accuracy
      iv.  Collect per-layer MSE from accumulator
      v.   Remove hooks, restore_weights(model, originals)
   b. Compute mean ± std across runs
5. Save results:
   - outputs/exp2_granularity/accuracy_results.csv (per-run)
   - outputs/exp2_granularity/accuracy_summary.json (mean ± std)
   - outputs/exp2_granularity/per_layer_mse.csv (per-config, per-layer)
```

### 5.3 Tests

| Test | File | Type | What it verifies |
|------|------|------|-----------------|
| `test_quantize_all_weights_and_restore` | `tests/test_quantization.py` | unit | Weight quantize → restore roundtrip preserves original values |
| `test_activation_quant_hook_per_tensor` | `tests/test_quantization.py` | unit | Hook quantizes a 3D tensor with per-tensor strategy |
| `test_activation_quant_hook_per_token` | `tests/test_quantization.py` | unit | Hook quantizes a 3D tensor with per-token strategy |
| `test_activation_quant_hook_2d_fallback` | `tests/test_quantization.py` | unit | Per-token on 2D input falls back to per-tensor (no crash) |
| `test_mse_accumulator_correctness` | `tests/test_quantization.py` | unit | MSEAccumulator computes correct weighted mean |
| `test_exp2_full_pipeline_runs` | `tests/test_integration.py` | integration (slow) | Full quantize → hook → forward → restore cycle on real ViT |
| `test_exp2_accuracy_drops` | `tests/test_integration.py` | integration (slow) | Quantized accuracy < FP32 accuracy on a small subset |

---

## 6. Output Format

### `accuracy_results.csv` (per-run raw data)

```csv
config_name,weight_strategy,activation_strategy,run,top1_accuracy,accuracy_drop
per_tensor,per_tensor,per_tensor,1,72.34,8.91
per_tensor,per_tensor,per_tensor,2,72.34,8.91
per_tensor,per_tensor,per_tensor,3,72.34,8.91
per_channel_weights,per_channel,per_tensor,1,74.12,7.13
...
```

### `accuracy_summary.json` (mean ± std)

```json
{
  "baseline_fp32_accuracy": 81.25,
  "num_images": 4096,
  "num_runs": 3,
  "configs": [
    {
      "name": "per_tensor",
      "weight_strategy": "per_tensor",
      "activation_strategy": "per_tensor",
      "mean_accuracy": 72.34,
      "std_accuracy": 0.0,
      "mean_drop": 8.91,
      "std_drop": 0.0
    }
  ]
}
```

### `per_layer_mse.csv` (one row per config × layer)

```csv
config_name,layer_name,mean_mse
per_tensor,blocks.0.attn.qkv,0.0234
per_tensor,blocks.0.attn.proj,0.0189
per_tensor,blocks.0.mlp.fc1,0.0312
...
```

---

## 7. Timeline & Dependencies

| Item | Depends on | Est. effort |
|------|-----------|-------------|
| `quantization.py` additions | nothing | ~1 hour |
| Unit tests | `quantization.py` | ~1 hour |
| `run_exp2_granularity.py` | `quantization.py` | ~2 hours |
| Integration tests | driver script | ~1 hour |
| Run on 4,096 images | all of the above | ~2–4 hours GPU |
| Run on 50K images | 4,096 run validated | ~12–24 hours GPU |

Per the [July roadmap](./july-roadmap.md), Experiment 2 is scheduled for
**Day 22–27** (late July), after Experiments 3 and 4. It is lower priority
than 3 and 4 but fills out the picture. The roadmap notes it "can be scoped
down to a subset of layers to keep the timeline workable" — we are running the
full model by choice (per 2026-07-05 discussion).

---

## 8. Open Questions (resolved)

- ✅ **Fourth config (per-channel + per-token)?** Added as Config D.
- ✅ **Full model or subset?** Full model (all 49 linear layers).
- ✅ **Multiple runs?** 3 runs, report mean ± std.
- ✅ **Per-layer MSE?** Yes, recorded via `MSEAccumulator`.

## 9. Open Questions (unresolved)

- **Calibration vs. evaluation split?** Currently we use the same data for both
  (dynamic per-batch scaling). If we later switch to static calibration, we'll
  need a separate calibration set. This is the same open question noted in the
  [July roadmap](./july-roadmap.md) Section 3.4.