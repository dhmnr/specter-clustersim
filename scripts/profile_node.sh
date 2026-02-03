#!/bin/bash
# SPECTER Node Profiling Script
# Captures traces from multi-GPU benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECTER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPECTER_DIR"

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Detect GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

echo "========================================"
echo "  SPECTER Node Profiling"
echo "========================================"
echo ""
echo "GPUs detected: $NUM_GPUS"
nvidia-smi --query-gpu=index,name --format=csv
echo ""

# Parse arguments
OUTPUT_DIR="./traces"
SYNC_MODE=""
QUICK=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sync)
            SYNC_MODE="--sync"
            shift
            ;;
        --quick)
            QUICK="--quick"
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Clean previous traces
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Sync mode: ${SYNC_MODE:-off}"
echo "Quick mode: ${QUICK:-off}"
echo ""

# Run profiling
echo "Starting trace capture..."
echo ""

specter capture $SYNC_MODE -o "$OUTPUT_DIR" -- \
    torchrun --nproc_per_node="$NUM_GPUS" \
    scripts/bench_multigpu.py $QUICK $EXTRA_ARGS

echo ""
echo "========================================"
echo "  Trace Capture Complete"
echo "========================================"
echo ""

# List captured files
echo "Captured traces:"
ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "No trace files found!"
echo ""

# Quick analysis
echo "Running analysis..."
echo ""
specter analyze "$OUTPUT_DIR"

# Show replay command
echo ""
echo "========================================"
echo "  Next Steps"
echo "========================================"
echo ""
echo "View scaling analysis:"
echo "  specter replay $OUTPUT_DIR --gpu h100_sxm --network dgx_h100_400g"
echo ""
echo "Copy traces to local machine:"
echo "  scp -r user@host:$SPECTER_DIR/$OUTPUT_DIR ./traces"
echo ""
