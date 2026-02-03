#!/bin/bash
# SPECTER Remote Setup Script
# Run this on a multi-GPU node to set up profiling

set -e

echo "========================================"
echo "  SPECTER Setup"
echo "========================================"
echo ""

# Check for NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check for required tools
echo "Checking dependencies..."

if ! command -v gcc &> /dev/null; then
    echo "ERROR: gcc not found. Install with: apt install build-essential"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

echo "  gcc: $(gcc --version | head -1)"
echo "  python: $(python3 --version)"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECTER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPECTER_DIR"
echo "SPECTER directory: $SPECTER_DIR"
echo ""

# Build tracer libraries
echo "Building tracer libraries..."
cd specter/capture
make clean 2>/dev/null || true
make
echo ""

# Check if libraries built
if [ ! -f "../../build/libspecter.so" ]; then
    echo "ERROR: Failed to build libspecter.so"
    exit 1
fi

echo "Built:"
ls -la ../../build/*.so
echo ""

# Set up Python environment
cd "$SPECTER_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install torch -q
pip install -e . -q

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To run profiling:"
echo ""
echo "  source .venv/bin/activate"
echo "  ./scripts/profile_node.sh"
echo ""
echo "Or manually:"
echo ""
echo "  specter capture -o ./traces -- torchrun --nproc_per_node=$NUM_GPUS scripts/bench_multigpu.py"
echo "  specter analyze ./traces"
echo ""
