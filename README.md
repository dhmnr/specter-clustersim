# SPECTER-CLUSTERSIM

Trace NCCL and cuBLAS operations from real GPU workloads, then simulate them on clusters of any size.

## Quick Start

```bash
# Build the tracer
cd specter/capture && make

# Install the CLI
uv pip install -e .

# Capture traces from a training run
specter capture -o ./traces -- torchrun --nproc_per_node=8 train.py

# Analyze the traces
specter analyze ./traces

# Simulate on larger cluster
specter replay ./traces --gpu h100_sxm --network dgx_h100_400g
```

## Remote Node Profiling

For profiling on a multi-GPU node (DGX, cloud instance, etc.):

```bash
# 1. Clone to remote machine
ssh user@gpu-node
git clone <repo> specter && cd specter

# 2. Run setup (builds tracer, installs deps)
./scripts/setup.sh

# 3. Run profiling
./scripts/profile_node.sh

# 4. Copy traces back
exit
scp -r user@gpu-node:specter/traces ./traces

# 5. Analyze locally
specter analyze ./traces
specter replay ./traces --gpu h100_sxm
```

### Profile Script Options

```bash
# Quick mode (fewer tests)
./scripts/profile_node.sh --quick

# Sync mode (accurate timing, slower)
./scripts/profile_node.sh --sync

# Custom output directory
./scripts/profile_node.sh -o /data/traces

# Limit GPU count
./scripts/profile_node.sh --gpus 4
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CAPTURE (on real hardware)                                  │
│     LD_PRELOAD intercepts NCCL + cuBLAS calls                  │
│     Records: operation, size, timing, comm group               │
│                                                                 │
│  2. ANALYZE (trace parsing)                                     │
│     Compute patterns, communication patterns                    │
│     Bottleneck identification                                   │
│                                                                 │
│  3. SIMULATE (any cluster size)                                 │
│     Replay trace on simulated cluster                          │
│     cuBLAS: from trace or GPU profile                          │
│     NCCL: analytical model (bandwidth + latency)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

### specter capture

```bash
# Basic capture
specter capture -o ./traces -- python train.py

# With torchrun (multi-GPU)
specter capture -o ./traces -- torchrun --nproc_per_node=8 train.py

# Sync mode for accurate timing (slower)
specter capture --sync -o ./traces -- torchrun train.py
```

### specter analyze

```bash
# Analyze traces
specter analyze ./traces

# Specific rank
specter analyze ./traces --rank 0

# Export to JSON
specter analyze ./traces --json analysis.json
```

### specter replay

```bash
# Scaling analysis (default)
specter replay ./traces --gpu h100_sxm --network dgx_h100_400g

# Specific scale factor
specter replay ./traces --gpu h100_sxm --scale 8

# Different hardware
specter replay ./traces --gpu a100_sxm --network dgx_a100_200g
```

### specter profiles

```bash
# List available hardware profiles
specter profiles

# Output:
# === GPU Profiles ===
#   h100_sxm        NVIDIA H100 SXM (1979 TFLOPS FP16)
#   a100_sxm        NVIDIA A100 SXM (312 TFLOPS FP16)
#   rtx_4090        NVIDIA RTX 4090 (165 TFLOPS FP16)
#   ...
#
# === Network Profiles ===
#   dgx_h100_400g   DGX H100 (400G IB, 900G NVLink)
#   dgx_a100_200g   DGX A100 (200G IB, 600G NVLink)
#   ...
```

## Hardware Profiles

### GPUs

| Profile | GPU | FP16 TFLOPS | Memory | Bandwidth |
|---------|-----|-------------|--------|-----------|
| `h100_sxm` | H100 SXM | 1979 | 80 GB | 3350 GB/s |
| `h100_pcie` | H100 PCIe | 1513 | 80 GB | 2000 GB/s |
| `a100_sxm` | A100 SXM | 312 | 80 GB | 2039 GB/s |
| `a100_pcie` | A100 PCIe | 312 | 80 GB | 1935 GB/s |
| `rtx_4090` | RTX 4090 | 165 | 24 GB | 1008 GB/s |
| `rtx_3090` | RTX 3090 | 71 | 24 GB | 936 GB/s |
| `mi300x` | MI300X | 1307 | 192 GB | 5300 GB/s |

### Networks

| Profile | Description | Intra-node | Inter-node |
|---------|-------------|------------|------------|
| `dgx_h100_400g` | DGX H100 cluster | 900 Gbps (NVLink) | 400 Gbps (IB) |
| `dgx_a100_200g` | DGX A100 cluster | 600 Gbps (NVLink) | 200 Gbps (IB) |
| `cloud_pcie_100g` | Cloud instance | 64 Gbps (PCIe) | 100 Gbps (Eth) |

## Trace Format

Traces are stored as JSON Lines (`.jsonl`) files:

```jsonl
{"event":"init","rank":0,"world_size":8,"sync_mode":0,"pid":12345}
{"id":0,"type":"nccl","op":"AllReduce","start_us":1000.0,"duration_us":1234.5,"bytes":134217728,"dtype":"fp16","comm_size":8}
{"id":1,"type":"cublas","op":"GemmEx","start_us":2500.0,"duration_us":72.3,"m":4096,"n":4096,"k":4096,"tflops":1901.2}
{"event":"shutdown","time_us":500000.0,"total_ops":1234}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SPECTER_OUTPUT` | Directory for trace files | `.` |
| `SPECTER_SYNC` | Set to `1` for synchronous timing | `0` |

## Building from Source

```bash
# Build tracer libraries
cd specter/capture && make

# Install Python package (with uv)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or with pip
pip install -e .
```

## Architecture

```
specter/
├── specter/
│   ├── cli.py              # Command line interface
│   ├── sim.py              # Simulation engine
│   ├── capture/            # C tracer libraries
│   │   ├── nccl_tracer.c
│   │   ├── cublas_tracer.c
│   │   └── combined_tracer.c
│   └── profiles/           # Hardware profiles
│       ├── gpu.py          # GPU performance models
│       └── nccl.py         # NCCL analytical model
├── scripts/
│   ├── setup.sh            # Remote setup script
│   ├── profile_node.sh     # Run profiling
│   └── bench_multigpu.py   # Multi-GPU benchmark
└── build/
    └── libspecter.so       # Compiled tracer
```

## License

MIT
