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
# Override defaults via environment variables:
#   NUM_SEEDS=5 bash scripts/run_full_experiment.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

SMOKE=false
RUN_NAME=""

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
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/run_full_experiment.sh [--smoke] [--name <name>]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Configuration — override via environment
# ---------------------------------------------------------------------------

DATA_DIR="${DATA_DIR:-data}"

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

BASE_SEED="${BASE_SEED:-42}"
PHASE1_OUTPUT="${PHASE1_OUTPUT:-${OUTPUT_ROOT}/phase1-profiling}"
PHASE2_GLOBAL_OUTPUT="${PHASE2_GLOBAL_OUTPUT:-${OUTPUT_ROOT}/phase2-global}"
PHASE2_PER_CHANNEL_OUTPUT="${PHASE2_PER_CHANNEL_OUTPUT:-${OUTPUT_ROOT}/phase2-per-channel}"

# Sigma thresholds: Phase 1 convention {3, 4, 6}.
DEFAULT_SIGMAS="3.0 4.0 6.0"

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
echo "To regenerate plots:"
echo "  # Phase 1 plots"
echo "  python scripts/regenerate_plots.py \\"
echo "    --phase1-json ${PHASE1_OUTPUT}/seed_${BASE_SEED}/profiling_result.json \\"
echo "    --output-dir ${PHASE1_OUTPUT}/seed_${BASE_SEED}/"
echo ""
echo "  # Phase 2 comparison (global vs per-channel)"
echo "  python scripts/regenerate_plots.py \\"
echo "    --phase2-csv-a ${PHASE2_GLOBAL_OUTPUT}/seed_${BASE_SEED}/ablation_results.csv \\"
echo "    --phase2-csv-b ${PHASE2_PER_CHANNEL_OUTPUT}/seed_${BASE_SEED}/ablation_results.csv \\"
echo "    --output-dir ${OUTPUT_ROOT}/phase2-comparison/"
echo ""
echo "  # Poster plots"
echo "  python scripts/generate_poster_plots.py \\"
echo "    --phase1-json ${PHASE1_OUTPUT}/seed_${BASE_SEED}/profiling_result.json \\"
echo "    --phase2-csv-a ${PHASE2_GLOBAL_OUTPUT}/seed_${BASE_SEED}/ablation_results.csv \\"
echo "    --phase2-csv-b ${PHASE2_PER_CHANNEL_OUTPUT}/seed_${BASE_SEED}/ablation_results.csv \\"
echo "    --output-dir ${OUTPUT_ROOT}/poster-plots"