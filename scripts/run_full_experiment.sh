#!/usr/bin/env bash
# =============================================================================
# Full experiment orchestration script — Phase 1 + Phase 2
#
# Full run:
#   bash scripts/run_full_experiment.sh
#
# Smoke test (tiny dataset, 1 seed, ~2 min):
#   bash scripts/run_full_experiment.sh --smoke
#
# Named output directory:
#   bash scripts/run_full_experiment.sh --name run-2026-08-04
#
# Custom configuration:
#   bash scripts/run_full_experiment.sh --num-seeds 5 --batch-size 128
#
# Show all options:
#   bash scripts/run_full_experiment.sh --help
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SMOKE=false
RUN_NAME=""
DATA_DIR="data"
BATCH_SIZE=64
NUM_SEEDS=3
BASE_SEED=42

# Derived defaults (set after parsing)
OUTPUT_ROOT=""
PHASE1_OUTPUT=""
PHASE2_GLOBAL_OUTPUT=""
PHASE2_PER_CHANNEL_OUTPUT=""

# Sigma thresholds: Phase 1 convention {3, 4, 6}.
DEFAULT_SIGMAS="3.0 4.0 6.0"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

print_help() {
    cat <<EOF
Usage: bash scripts/run_full_experiment.sh [OPTIONS]

Runs the full ViT quantization experiment pipeline: Phase 1 profiling,
Phase 2 global ablation, Phase 2 per-channel ablation, and per-channel
ablation modes (mean_only, var_only).

Options:
  --smoke                     Run a smoke test (128 images, 1 seed).
  --name <name>               Run name (default: "final" or "smoke").
  --data-dir <path>           ImageNet val directory (default: data).
  --output-root <path>        Output root directory (default: outputs/<name>).
  --batch-size <int>          Batch size (default: 64 full, 32 smoke).
  --num-seeds <int>           Number of seeds (default: 3 full, 1 smoke).
  --base-seed <int>           First seed value (default: 42).
  --phase1-output <path>      Phase 1 output dir (default: <output-root>/phase1-profiling).
  --phase2-global-output <path>  Phase 2 global output dir (default: <output-root>/phase2-global).
  --phase2-per-channel-output <path>  Phase 2 per-channel output dir (default: <output-root>/phase2-per-channel).
  -h, --help                  Show this help message and exit.

Examples:
  bash scripts/run_full_experiment.sh
  bash scripts/run_full_experiment.sh --smoke
  bash scripts/run_full_experiment.sh --name my-run --num-seeds 5
  bash scripts/run_full_experiment.sh --data-dir /path/to/imagenet --batch-size 128
EOF
}

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=true
            shift
            ;;
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --base-seed)
            BASE_SEED="$2"
            shift 2
            ;;
        --phase1-output)
            PHASE1_OUTPUT="$2"
            shift 2
            ;;
        --phase2-global-output)
            PHASE2_GLOBAL_OUTPUT="$2"
            shift 2
            ;;
        --phase2-per-channel-output)
            PHASE2_PER_CHANNEL_OUTPUT="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/run_full_experiment.sh [--smoke] [--name <name>] [--help]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Apply smoke defaults and derive remaining paths
# ---------------------------------------------------------------------------

if $SMOKE; then
    RUN_NAME="${RUN_NAME:-smoke}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${RUN_NAME}}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    NUM_SEEDS="${NUM_SEEDS:-1}"
    NUM_IMAGES=128
    PHASE1_FLAGS="--num-images $NUM_IMAGES"
    PHASE2_FLAGS="--num-images $NUM_IMAGES"
    echo "=== SMOKE TEST — $NUM_IMAGES images, $NUM_SEEDS seed(s) ==="
else
    RUN_NAME="${RUN_NAME:-final}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${RUN_NAME}}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    NUM_SEEDS="${NUM_SEEDS:-3}"
    NUM_IMAGES=50000
    PHASE1_FLAGS="--all"
    PHASE2_FLAGS="--num-images $NUM_IMAGES"
fi

PHASE1_OUTPUT="${PHASE1_OUTPUT:-${OUTPUT_ROOT}/phase1-profiling}"
PHASE2_GLOBAL_OUTPUT="${PHASE2_GLOBAL_OUTPUT:-${OUTPUT_ROOT}/phase2-global}"
PHASE2_PER_CHANNEL_OUTPUT="${PHASE2_PER_CHANNEL_OUTPUT:-${OUTPUT_ROOT}/phase2-per-channel}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

_start_timer() {
    echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
}

_end_timer() {
    local label="${1:-Step}"
    echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------

_print_header "Validating prerequisites"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory '$DATA_DIR' not found."
    echo "  Run: python download_imagenet_val.py --num-images 50000"
    exit 1
fi

IMAGE_COUNT=$(find "$DATA_DIR" -name "*.JPEG" -o -name "*.jpeg" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "  Data: $IMAGE_COUNT images in $DATA_DIR"
if [[ "$IMAGE_COUNT" -lt "$NUM_IMAGES" ]]; then
    echo "WARNING: Only $IMAGE_COUNT images available, but run expects $NUM_IMAGES."
fi

if ! python -c "import timm; import nnsight; print('OK')" 2>/dev/null; then
    echo "ERROR: timm or nnsight not importable. Is the conda env active?"
    exit 1
fi
echo "  Python deps: OK"

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found'; print('OK')" 2>/dev/null; then
    echo "ERROR: CUDA not available. This experiment requires a GPU."
    exit 1
fi
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}')")
echo "  GPU: $GPU_NAME ($GPU_MEM GB)"
echo "  Batch size: $BATCH_SIZE"
echo "  Output root: $OUTPUT_ROOT"

echo "  Seeds: $BASE_SEED..$((BASE_SEED + NUM_SEEDS - 1)) ($NUM_SEEDS total)"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Profiling
#
# NOTE: Phase 1 MUST be re-run if the profiler has changed since the last
# profiling output was produced (e.g., new track_per_channel sites added,
# LayerStats fields changed).  Per-channel Phase 2 ablation will refuse to
# run with stale profiling data.
# ---------------------------------------------------------------------------

_print_header "Phase 1 — Baseline Activation Profiling ($NUM_IMAGES images, $NUM_SEEDS seeds)"

_start_timer
python run_phase1_profiling.py \
    $PHASE1_FLAGS \
    --num-seeds "$NUM_SEEDS" \
    --seed "$BASE_SEED" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$PHASE1_OUTPUT"
_end_timer "Phase 1"

PHASE1_STATS="${PHASE1_OUTPUT}/seed_${BASE_SEED}/profiling_result.json"
if [[ ! -f "$PHASE1_STATS" ]]; then
    echo "ERROR: Expected Phase 1 output not found: $PHASE1_STATS"
    echo "  Phase 1 may have failed. Check logs above."
    exit 1
fi
echo "  Phase 1 stats: $PHASE1_STATS"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Global Ablation
# ---------------------------------------------------------------------------

_print_header "Phase 2 — Global Granularity Ablation ($NUM_SEEDS seeds, k=$DEFAULT_SIGMAS)"

_start_timer
python run_phase2_ablation.py \
    $PHASE2_FLAGS \
    --num-seeds "$NUM_SEEDS" \
    --seed "$BASE_SEED" \
    --batch-size "$BATCH_SIZE" \
    --sigma-thresholds $DEFAULT_SIGMAS \
    --granularity global \
    --layer-stats "$PHASE1_STATS" \
    --output-dir "$PHASE2_GLOBAL_OUTPUT"
_end_timer "Phase 2 (global)"

# ---------------------------------------------------------------------------
# Phase 2: Per-Channel Ablation
# ---------------------------------------------------------------------------

_print_header "Phase 2 — Per-Channel Granularity Ablation ($NUM_SEEDS seeds, k=$DEFAULT_SIGMAS)"

_start_timer
python run_phase2_ablation.py \
    $PHASE2_FLAGS \
    --num-seeds "$NUM_SEEDS" \
    --seed "$BASE_SEED" \
    --batch-size "$BATCH_SIZE" \
    --sigma-thresholds $DEFAULT_SIGMAS \
    --granularity per_channel \
    --layer-stats "$PHASE1_STATS" \
    --output-dir "$PHASE2_PER_CHANNEL_OUTPUT"
_end_timer "Phase 2 (per_channel)"

# ---------------------------------------------------------------------------
# Phase 2: Per-Channel Ablation Modes (RQ2 — mean_only, var_only at k=3)
# ---------------------------------------------------------------------------

_print_header "Phase 2 — Per-Channel Ablation Modes at k=3 (RQ2) ($NUM_SEEDS seeds)"

P2_MEAN_ONLY_OUTPUT="${OUTPUT_ROOT}/phase2-per-channel-mean-only"
_start_timer
python run_phase2_ablation.py \
    $PHASE2_FLAGS \
    --num-seeds "$NUM_SEEDS" \
    --seed "$BASE_SEED" \
    --batch-size "$BATCH_SIZE" \
    --sigma-thresholds 3.0 \
    --granularity per_channel \
    --ablation-mode mean_only \
    --per-channel-sites pre_gelu \
    --layer-stats "$PHASE1_STATS" \
    --output-dir "$P2_MEAN_ONLY_OUTPUT"
_end_timer "Phase 2 (mean_only)"

P2_VAR_ONLY_OUTPUT="${OUTPUT_ROOT}/phase2-per-channel-var-only"
_start_timer
python run_phase2_ablation.py \
    $PHASE2_FLAGS \
    --num-seeds "$NUM_SEEDS" \
    --seed "$BASE_SEED" \
    --batch-size "$BATCH_SIZE" \
    --sigma-thresholds 3.0 \
    --granularity per_channel \
    --ablation-mode var_only \
    --per-channel-sites pre_gelu \
    --layer-stats "$PHASE1_STATS" \
    --output-dir "$P2_VAR_ONLY_OUTPUT"
_end_timer "Phase 2 (var_only)"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

_print_header "Experiment Complete"

echo "Outputs:"
echo "  Phase 1:              $PHASE1_OUTPUT"
echo "  Phase 2 (global):     $PHASE2_GLOBAL_OUTPUT"
echo "  Phase 2 (per_channel): $PHASE2_PER_CHANNEL_OUTPUT"
echo "  Phase 2 (mean_only):  $P2_MEAN_ONLY_OUTPUT"
echo "  Phase 2 (var_only):   $P2_VAR_ONLY_OUTPUT"
echo ""
echo "To regenerate all plots:"
echo "  bash scripts/regenerate_all.sh --run-dir $OUTPUT_ROOT"
echo ""