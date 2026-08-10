#!/usr/bin/env bash
# =============================================================================
# Unified plot regeneration script
#
# Regenerates all plots from a single run directory with one command.
#
# Usage:
#   bash scripts/regenerate_all.sh --run-dir outputs/full-run-2026-8-4
#
# What it does:
#   1. Auto-discovers profiling JSON and ablation CSVs under <run-dir>.
#   2. Combines mean_only/var_only/outlier per-channel CSVs for the waterfall.
#   3. Calls regenerate_plots.py, generate_poster_plots.py,
#      analyze_ablation_results.py, analyze_effective_gain.py, and
#      analyze_layernorm_gamma.py.
#   4. Writes all output into <run-dir>/plots/ with subdirectories.
#   5. Prints a summary of what was generated and where.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

RUN_DIR=""
OUTPUT_NAME="plots"
HELP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/regenerate_all.sh --run-dir <path> [--output-name <name>]"
            exit 1
            ;;
    esac
done

if $HELP; then
    echo "Usage: bash scripts/regenerate_all.sh --run-dir <path> [--output-name <name>]"
    echo ""
    echo "Regenerates all plots from a single run directory."
    echo ""
    echo "Required:"
    echo "  --run-dir <path>       Path to the run directory (e.g. outputs/full-run-2026-8-4)"
    echo ""
    echo "Optional:"
    echo "  --output-name <name>   Name of the output subdirectory (default: plots)"
    echo ""
    echo "The script auto-discovers:"
    echo "  - Phase 1 profiling JSON under <run-dir>/phase1-profiling/seed_*/"
    echo "  - Phase 2 global CSV under <run-dir>/phase2-global/seed_*/"
    echo "  - Phase 2 per-channel CSVs under <run-dir>/phase2-per-channel*/seed_*/"
    echo "    (outlier, mean_only, and var_only are merged for the waterfall)"
    echo ""
    echo "Output is written to <run-dir>/<output-name>/ with subdirectories:"
    echo "  phase1/     — Phase 1 profiling plots"
    echo "  phase2/     — Phase 2 single-run and comparison plots"
    echo "  poster/     — Poster-quality plots"
    echo "  analysis/   — Ablation analysis, effective gain, layernorm gamma"
    exit 0
fi

if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: --run-dir is required."
    echo "Usage: bash scripts/regenerate_all.sh --run-dir <path>"
    exit 1
fi

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-discover files
# ---------------------------------------------------------------------------

echo "=== Auto-discovering files in $RUN_DIR ==="

# Phase 1: first seed_*/profiling_result.json
PHASE1_JSON=""
PHASE1_DIR="$RUN_DIR/phase1-profiling"
if [[ -d "$PHASE1_DIR" ]]; then
    for seed_dir in "$PHASE1_DIR"/seed_*/; do
        candidate="${seed_dir}profiling_result.json"
        if [[ -f "$candidate" ]]; then
            PHASE1_JSON="$candidate"
            echo "  Phase 1 JSON: $PHASE1_JSON"
            break
        fi
    done
fi

if [[ -z "$PHASE1_JSON" ]]; then
    echo "WARNING: No profiling_result.json found under $PHASE1_DIR"
fi

# Phase 2 global (csv-a)
CSV_A=""
GLOBAL_DIR="$RUN_DIR/phase2-global"
if [[ -d "$GLOBAL_DIR" ]]; then
    for seed_dir in "$GLOBAL_DIR"/seed_*/; do
        candidate="${seed_dir}ablation_results.csv"
        if [[ -f "$candidate" ]]; then
            CSV_A="$candidate"
            echo "  CSV A (global): $CSV_A"
            break
        fi
    done
fi

if [[ -z "$CSV_A" ]]; then
    echo "WARNING: No global ablation_results.csv found under $GLOBAL_DIR"
fi

# Phase 2 per-channel (csv-b) — collect all modes
CSV_B_FILES=()
for pc_subdir in "phase2-per-channel" "phase2-per-channel-mean-only" "phase2-per-channel-var-only"; do
    PC_DIR="$RUN_DIR/$pc_subdir"
    if [[ -d "$PC_DIR" ]]; then
        for seed_dir in "$PC_DIR"/seed_*/; do
            candidate="${seed_dir}ablation_results.csv"
            if [[ -f "$candidate" ]]; then
                CSV_B_FILES+=("$candidate")
                echo "  CSV B ($pc_subdir): $candidate"
                break
            fi
        done
    fi
done

if [[ ${#CSV_B_FILES[@]} -eq 0 ]]; then
    echo "WARNING: No per-channel ablation_results.csv files found."
fi

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

PLOTS_DIR="$RUN_DIR/$OUTPUT_NAME"
mkdir -p "$PLOTS_DIR"

PHASE1_OUT="$PLOTS_DIR/phase1"
PHASE2_OUT="$PLOTS_DIR/phase2"
POSTER_OUT="$PLOTS_DIR/poster"
ANALYSIS_OUT="$PLOTS_DIR/analysis"

mkdir -p "$PHASE1_OUT" "$PHASE2_OUT" "$POSTER_OUT" "$ANALYSIS_OUT"

# ---------------------------------------------------------------------------
# 1. Regenerate standard plots (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

echo ""
echo "=== [1/5] Regenerating standard plots ==="

REGENERATE_ARGS=()
if [[ -n "$PHASE1_JSON" ]]; then
    REGENERATE_ARGS+=(--layer-stats "$PHASE1_JSON")
fi
if [[ -n "$CSV_A" ]]; then
    REGENERATE_ARGS+=(--csv-a "$CSV_A")
fi
if [[ ${#CSV_B_FILES[@]} -gt 0 ]]; then
    # Use the first per-channel CSV (outlier) as csv-b for comparison plots.
    REGENERATE_ARGS+=(--csv-b "${CSV_B_FILES[0]}")
fi

python scripts/regenerate_plots.py \
    "${REGENERATE_ARGS[@]}" \
    --output-dir "$PHASE2_OUT"

# Phase 1 plots go to phase1 subdir if we have profiling data.
if [[ -n "$PHASE1_JSON" ]]; then
    python scripts/regenerate_plots.py \
        --layer-stats "$PHASE1_JSON" \
        --output-dir "$PHASE1_OUT"
fi

# Phase 2 single-run plots for global.
if [[ -n "$CSV_A" ]]; then
    python scripts/regenerate_plots.py \
        --csv "$CSV_A" \
        --output-dir "$PHASE2_OUT/global"
fi

# Phase 2 single-run plots for per-channel (outlier).
if [[ ${#CSV_B_FILES[@]} -gt 0 ]]; then
    python scripts/regenerate_plots.py \
        --csv "${CSV_B_FILES[0]}" \
        --output-dir "$PHASE2_OUT/per-channel"
fi

echo "  Done: $PHASE1_OUT, $PHASE2_OUT"

# ---------------------------------------------------------------------------
# 2. Generate poster plots
# ---------------------------------------------------------------------------

echo ""
echo "=== [2/5] Generating poster plots ==="

POSTER_ARGS=()
if [[ -n "$PHASE1_JSON" ]]; then
    POSTER_ARGS+=(--layer-stats "$PHASE1_JSON")
fi
if [[ -n "$CSV_A" ]]; then
    POSTER_ARGS+=(--csv-a "$CSV_A")
fi
for csv_b in "${CSV_B_FILES[@]}"; do
    POSTER_ARGS+=(--csv-b "$csv_b")
done

if [[ -n "$CSV_A" ]] && [[ ${#CSV_B_FILES[@]} -gt 0 ]]; then
    python scripts/generate_poster_plots.py \
        "${POSTER_ARGS[@]}" \
        --output-dir "$POSTER_OUT"
    echo "  Done: $POSTER_OUT"
else
    echo "  Skipped (need both global and per-channel CSVs)."
fi

# ---------------------------------------------------------------------------
# 3. Ablation analysis
# ---------------------------------------------------------------------------

echo ""
echo "=== [3/5] Running ablation analysis ==="

if [[ -n "$CSV_A" ]] && [[ ${#CSV_B_FILES[@]} -gt 0 ]]; then
    python scripts/analyze_ablation_results.py \
        --csv-a "$CSV_A" \
        --csv-b "${CSV_B_FILES[0]}" \
        --output-dir "$ANALYSIS_OUT/ablation"
    echo "  Done: $ANALYSIS_OUT/ablation"
else
    echo "  Skipped (need both global and per-channel CSVs)."
fi

# ---------------------------------------------------------------------------
# 4. Effective gain analysis
# ---------------------------------------------------------------------------

echo ""
echo "=== [4/5] Running effective gain analysis ==="

if [[ -n "$PHASE1_JSON" ]]; then
    python scripts/analyze_effective_gain.py \
        --layer-stats "$PHASE1_JSON" \
        --output-dir "$ANALYSIS_OUT/effective-gain"
    echo "  Done: $ANALYSIS_OUT/effective-gain"
else
    echo "  Skipped (need Phase 1 profiling JSON)."
fi

# ---------------------------------------------------------------------------
# 5. LayerNorm gamma analysis
# ---------------------------------------------------------------------------

echo ""
echo "=== [5/5] Running LayerNorm gamma analysis ==="

if [[ -n "$PHASE1_JSON" ]]; then
    python scripts/analyze_layernorm_gamma.py \
        --layer-stats "$PHASE1_JSON" \
        --output-dir "$ANALYSIS_OUT/layernorm-gamma"
    echo "  Done: $ANALYSIS_OUT/layernorm-gamma"
else
    echo "  Skipped (need Phase 1 profiling JSON)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Plot Regeneration Complete"
echo "================================================================"
echo ""
echo "Output directory: $PLOTS_DIR"
echo ""

_count_pngs() {
    local d="$1"
    if [[ -d "$d" ]]; then
        find "$d" -name "*.png" -o -name "*.json" 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

echo "  phase1/     ($(_count_pngs "$PHASE1_OUT") files)"
echo "  phase2/     ($(_count_pngs "$PHASE2_OUT") files)"
echo "  poster/     ($(_count_pngs "$POSTER_OUT") files)"
echo "  analysis/   ($(_count_pngs "$ANALYSIS_OUT") files)"
echo ""
echo "To regenerate again:"
echo "  bash scripts/regenerate_all.sh --run-dir $RUN_DIR --output-name $OUTPUT_NAME"
echo ""