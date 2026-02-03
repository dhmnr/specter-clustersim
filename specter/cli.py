#!/usr/bin/env python3
"""
SPECTER CLI

Commands:
  capture  - Run a command with tracing enabled
  analyze  - Analyze captured traces
  replay   - Replay traces on simulated cluster
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class NCCLOp:
    """A single NCCL operation from the trace."""
    id: int
    op: str
    start_us: float
    duration_us: float
    count: int
    bytes: int
    dtype: str
    comm_rank: int = 0
    comm_size: int = 1
    peer: int = -1
    redop: str = ""
    in_group: bool = False


@dataclass
class CublasOp:
    """A single cuBLAS operation from the trace."""
    id: int
    op: str
    start_us: float
    duration_us: float
    m: int
    n: int
    k: int
    batch: int = 1
    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    compute: str = "fp32"
    trans_a: str = "N"
    trans_b: str = "N"
    flops: float = 0
    tflops: float = 0


@dataclass
class Trace:
    """Complete trace from one rank."""
    rank: int
    world_size: int
    nccl_ops: list[NCCLOp] = field(default_factory=list)
    cublas_ops: list[CublasOp] = field(default_factory=list)
    total_time_us: float = 0


# ============================================================================
# Trace Parsing
# ============================================================================

def parse_trace_file(path: Path) -> Trace:
    """Parse a single trace file (JSONL format)."""
    trace = Trace(rank=0, world_size=1)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Handle init event
            if data.get("event") == "init":
                trace.rank = data.get("rank", 0)
                trace.world_size = data.get("world_size", 1)
                continue

            # Handle shutdown event
            if data.get("event") == "shutdown":
                trace.total_time_us = data.get("time_us", 0)
                continue

            # Skip other events
            if "event" in data:
                continue

            # Parse operation based on type
            op_type = data.get("type", "")

            # Combined trace format
            if op_type == "nccl":
                trace.nccl_ops.append(NCCLOp(
                    id=data.get("id", 0),
                    op=data.get("op", ""),
                    start_us=data.get("start_us", 0),
                    duration_us=data.get("duration_us", 0),
                    count=data.get("count", 0),
                    bytes=data.get("bytes", 0),
                    dtype=data.get("dtype", ""),
                    comm_rank=data.get("comm_rank", 0),
                    comm_size=data.get("comm_size", 1),
                    peer=data.get("peer", -1),
                    redop=data.get("redop", ""),
                    in_group=data.get("in_group", False),
                ))
            elif op_type == "cublas":
                trace.cublas_ops.append(CublasOp(
                    id=data.get("id", 0),
                    op=data.get("op", ""),
                    start_us=data.get("start_us", 0),
                    duration_us=data.get("duration_us", 0),
                    m=data.get("m", 0),
                    n=data.get("n", 0),
                    k=data.get("k", 0),
                    batch=data.get("batch", 1),
                    dtype_a=data.get("dtype_a", ""),
                    dtype_b=data.get("dtype_b", ""),
                    dtype_c=data.get("dtype_c", ""),
                    compute=data.get("compute", ""),
                    trans_a=data.get("trans_a", ""),
                    trans_b=data.get("trans_b", ""),
                    flops=data.get("flops", 0),
                    tflops=data.get("tflops", 0),
                ))
            # Standalone NCCL trace format
            elif "op" in data and data["op"] in ("AllReduce", "Broadcast", "AllGather",
                                                   "ReduceScatter", "Send", "Recv"):
                trace.nccl_ops.append(NCCLOp(
                    id=data.get("id", len(trace.nccl_ops)),
                    op=data.get("op", ""),
                    start_us=data.get("start_us", 0),
                    duration_us=data.get("duration_us", 0),
                    count=data.get("count", 0),
                    bytes=data.get("bytes", 0),
                    dtype=data.get("dtype", ""),
                    comm_rank=data.get("comm_rank", 0),
                    comm_size=data.get("comm_size", 1),
                    peer=data.get("peer", -1),
                    redop=data.get("redop", ""),
                    in_group=data.get("in_group", False),
                ))
            # Standalone cuBLAS trace format
            elif "op" in data and "Gemm" in data.get("op", ""):
                trace.cublas_ops.append(CublasOp(
                    id=data.get("id", len(trace.cublas_ops)),
                    op=data.get("op", ""),
                    start_us=data.get("start_us", 0),
                    duration_us=data.get("duration_us", 0),
                    m=data.get("m", 0),
                    n=data.get("n", 0),
                    k=data.get("k", 0),
                    batch=data.get("batch", 1),
                    dtype_a=data.get("dtype_a", ""),
                    dtype_b=data.get("dtype_b", ""),
                    dtype_c=data.get("dtype_c", ""),
                    compute=data.get("compute", ""),
                    trans_a=data.get("trans_a", ""),
                    trans_b=data.get("trans_b", ""),
                    flops=data.get("flops", 0),
                    tflops=data.get("tflops", 0),
                ))

    return trace


def load_traces(trace_dir: Path) -> list[Trace]:
    """Load all trace files from a directory."""
    traces = []

    # Find all trace files
    patterns = [
        "specter_trace_rank*.jsonl",
        "specter_nccl_rank*.jsonl",
        "specter_cublas_rank*.jsonl",
    ]

    files = set()
    for pattern in patterns:
        files.update(trace_dir.glob(pattern))

    # Group by rank for combined loading
    rank_files: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        # Extract rank from filename
        name = f.stem
        for part in name.split("_"):
            if part.startswith("rank"):
                try:
                    rank = int(part[4:])
                    rank_files[rank].append(f)
                except ValueError:
                    pass

    # Load and merge traces per rank
    for rank, paths in sorted(rank_files.items()):
        combined = Trace(rank=rank, world_size=1)
        for path in paths:
            trace = parse_trace_file(path)
            combined.world_size = max(combined.world_size, trace.world_size)
            combined.nccl_ops.extend(trace.nccl_ops)
            combined.cublas_ops.extend(trace.cublas_ops)
            combined.total_time_us = max(combined.total_time_us, trace.total_time_us)
        traces.append(combined)

    return traces


# ============================================================================
# Analysis
# ============================================================================

def analyze_traces(traces: list[Trace]) -> dict:
    """Analyze traces and return summary statistics."""
    if not traces:
        return {}

    # Use rank 0 as reference
    trace = traces[0]

    # NCCL analysis
    nccl_by_op = defaultdict(list)
    for op in trace.nccl_ops:
        nccl_by_op[op.op].append(op)

    nccl_summary = {}
    for op_name, ops in nccl_by_op.items():
        total_time_ms = sum(op.duration_us for op in ops) / 1000
        total_bytes = sum(op.bytes for op in ops)
        avg_bw = (total_bytes / 1e9) / (total_time_ms / 1000) if total_time_ms > 0 else 0

        # Get unique comm sizes
        comm_sizes = set(op.comm_size for op in ops)

        nccl_summary[op_name] = {
            "count": len(ops),
            "total_time_ms": total_time_ms,
            "total_bytes": total_bytes,
            "total_gb": total_bytes / 1e9,
            "avg_bandwidth_gbps": avg_bw,
            "comm_sizes": sorted(comm_sizes),
        }

    # cuBLAS analysis
    total_compute_time_ms = sum(op.duration_us for op in trace.cublas_ops) / 1000
    total_flops = sum(op.flops for op in trace.cublas_ops)
    avg_tflops = (total_flops / 1e12) / (total_compute_time_ms / 1000) if total_compute_time_ms > 0 else 0

    # Group GEMMs by shape
    gemm_shapes = defaultdict(list)
    for op in trace.cublas_ops:
        shape = (op.m, op.n, op.k, op.batch)
        gemm_shapes[shape].append(op)

    cublas_summary = {
        "total_ops": len(trace.cublas_ops),
        "total_time_ms": total_compute_time_ms,
        "total_tflops": total_flops / 1e12,
        "avg_tflops": avg_tflops,
        "unique_shapes": len(gemm_shapes),
        "top_shapes": [],
    }

    # Top shapes by total time
    shape_times = []
    for shape, ops in gemm_shapes.items():
        total_time = sum(op.duration_us for op in ops)
        shape_times.append((shape, len(ops), total_time))
    shape_times.sort(key=lambda x: -x[2])

    for shape, count, total_us in shape_times[:5]:
        m, n, k, batch = shape
        cublas_summary["top_shapes"].append({
            "shape": f"{m}x{n}x{k}" + (f"x{batch}" if batch > 1 else ""),
            "count": count,
            "total_time_ms": total_us / 1000,
        })

    # Overall summary
    total_nccl_ms = sum(s["total_time_ms"] for s in nccl_summary.values())
    total_time_ms = total_nccl_ms + total_compute_time_ms

    return {
        "rank": trace.rank,
        "world_size": trace.world_size,
        "nccl": nccl_summary,
        "cublas": cublas_summary,
        "total_nccl_ms": total_nccl_ms,
        "total_compute_ms": total_compute_time_ms,
        "total_time_ms": total_time_ms,
        "compute_fraction": total_compute_time_ms / total_time_ms if total_time_ms > 0 else 0,
        "comm_fraction": total_nccl_ms / total_time_ms if total_time_ms > 0 else 0,
    }


def print_analysis(analysis: dict):
    """Pretty-print analysis results."""
    print()
    print("=" * 60)
    print("  SPECTER Trace Analysis")
    print("=" * 60)
    print()
    print(f"  Rank: {analysis['rank']}  World Size: {analysis['world_size']}")
    print()

    # NCCL
    print("-" * 60)
    print("  NCCL Operations")
    print("-" * 60)

    nccl = analysis.get("nccl", {})
    if nccl:
        for op_name, stats in sorted(nccl.items()):
            print(f"\n  {op_name}:")
            print(f"    Count:      {stats['count']}")
            print(f"    Total time: {stats['total_time_ms']:.2f} ms")
            print(f"    Total data: {stats['total_gb']:.2f} GB")
            print(f"    Bandwidth:  {stats['avg_bandwidth_gbps']:.1f} GB/s")
            if len(stats['comm_sizes']) > 1:
                print(f"    Comm sizes: {stats['comm_sizes']}")
    else:
        print("  No NCCL operations recorded")

    # cuBLAS
    print()
    print("-" * 60)
    print("  cuBLAS (Compute)")
    print("-" * 60)

    cublas = analysis.get("cublas", {})
    if cublas["total_ops"] > 0:
        print(f"\n  Total GEMMs:   {cublas['total_ops']}")
        print(f"  Total time:    {cublas['total_time_ms']:.2f} ms")
        print(f"  Total TFLOPs:  {cublas['total_tflops']:.1f}")
        print(f"  Avg TFLOPS:    {cublas['avg_tflops']:.1f}")

        if cublas["top_shapes"]:
            print(f"\n  Top GEMM shapes by time:")
            for i, shape in enumerate(cublas["top_shapes"], 1):
                print(f"    {i}. {shape['shape']:20s} x{shape['count']:4d}  {shape['total_time_ms']:8.2f} ms")
    else:
        print("  No cuBLAS operations recorded")

    # Summary
    print()
    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    print(f"\n  Total time:      {analysis['total_time_ms']:.2f} ms")
    print(f"  Compute:         {analysis['total_compute_ms']:.2f} ms ({analysis['compute_fraction']*100:.1f}%)")
    print(f"  Communication:   {analysis['total_nccl_ms']:.2f} ms ({analysis['comm_fraction']*100:.1f}%)")
    print()


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_capture(args):
    """Run a command with SPECTER tracing enabled."""
    # Find the tracer library
    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"

    lib_paths = [
        build_dir / "libspecter.so",
        build_dir / "libspecter_nccl.so",
        script_dir / "capture" / "libspecter.so",
    ]

    lib_path = None
    for p in lib_paths:
        if p.exists():
            lib_path = p
            break

    if lib_path is None:
        print("Error: SPECTER tracer library not found.")
        print("Please build it first:")
        print("  cd specter/capture && make")
        return 1

    # Set up environment
    env = os.environ.copy()

    # LD_PRELOAD
    existing_preload = env.get("LD_PRELOAD", "")
    if existing_preload:
        env["LD_PRELOAD"] = f"{lib_path}:{existing_preload}"
    else:
        env["LD_PRELOAD"] = str(lib_path)

    # Output directory
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env["SPECTER_OUTPUT"] = str(output_dir)

    # Sync mode
    if args.sync:
        env["SPECTER_SYNC"] = "1"

    print(f"[SPECTER] Tracing enabled")
    print(f"[SPECTER] Output: {output_dir}")
    print(f"[SPECTER] Sync mode: {'on' if args.sync else 'off'}")
    print()

    # Run the command
    cmd = args.cmd
    result = subprocess.run(cmd, env=env)

    return result.returncode


def cmd_analyze(args):
    """Analyze captured traces."""
    trace_dir = Path(args.trace_dir)

    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        return 1

    traces = load_traces(trace_dir)

    if not traces:
        print(f"Error: No trace files found in {trace_dir}")
        return 1

    print(f"[SPECTER] Loaded {len(traces)} trace(s) from {trace_dir}")

    # Analyze rank 0 (or specified rank)
    rank = args.rank if args.rank is not None else 0
    trace = None
    for t in traces:
        if t.rank == rank:
            trace = t
            break

    if trace is None:
        print(f"Error: Rank {rank} not found in traces")
        return 1

    analysis = analyze_traces([trace])
    print_analysis(analysis)

    # Export JSON if requested
    if args.json:
        with open(args.json, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"[SPECTER] Analysis exported to {args.json}")

    return 0


def cmd_replay(args):
    """Replay traces on simulated cluster."""
    from .sim import Simulator, print_scaling_analysis
    from .profiles.gpu import list_gpu_profiles
    from .profiles.nccl import NETWORK_PROFILES

    trace_dir = Path(args.trace_dir)

    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        return 1

    # Use provided profiles or defaults
    gpu_name = args.gpu or "h100_sxm"
    network_name = args.network or "dgx_h100_400g"

    print(f"[SPECTER] Loading traces from {trace_dir}")
    print(f"[SPECTER] Target GPU: {gpu_name}")
    print(f"[SPECTER] Target network: {network_name}")

    try:
        sim = Simulator.from_trace(
            trace_dir,
            gpu_name=gpu_name,
            network_name=network_name,
            calibrate=not args.no_calibrate,
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable GPUs: {', '.join(list_gpu_profiles())}")
        print(f"Available networks: {', '.join(NETWORK_PROFILES.keys())}")
        return 1

    print(f"[SPECTER] Loaded {len(sim.trace_gemms)} GEMMs, {len(sim.trace_nccl)} NCCL ops")

    if args.scale:
        # Single scale simulation
        result = sim.replay_trace(scale=args.scale)
        print(result.summary())
    else:
        # Scaling analysis
        scales = [1, 2, 4, 8, 16, 32]
        if args.max_scale:
            scales = [s for s in scales if s <= args.max_scale]

        results = sim.scaling_analysis(scales)
        print_scaling_analysis(results, base_world=1)

    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="specter",
        description="SPECTER - Ghost-run your GPU cluster at any scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture traces from a training run
  specter capture -o ./traces -- torchrun --nproc_per_node=8 train.py

  # Analyze captured traces
  specter analyze ./traces

  # Analyze with accurate timing (slower)
  specter capture --sync -o ./traces -- torchrun train.py
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # capture
    p_capture = subparsers.add_parser("capture", help="Run command with tracing")
    p_capture.add_argument("-o", "--output", default="./specter_traces",
                          help="Output directory for traces")
    p_capture.add_argument("--sync", action="store_true",
                          help="Enable sync mode for accurate timing (slower)")
    p_capture.add_argument("cmd", nargs=argparse.REMAINDER,
                          help="Command to run")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze traces")
    p_analyze.add_argument("trace_dir", help="Directory containing trace files")
    p_analyze.add_argument("--rank", type=int, default=None,
                          help="Rank to analyze (default: 0)")
    p_analyze.add_argument("--json", help="Export analysis to JSON file")

    # replay
    p_replay = subparsers.add_parser("replay", help="Replay traces on simulated cluster")
    p_replay.add_argument("trace_dir", help="Directory containing trace files")
    p_replay.add_argument("--gpu", help="GPU profile (e.g., h100_sxm, a100_sxm, rtx_4090)")
    p_replay.add_argument("--network", help="Network profile (e.g., dgx_h100_400g)")
    p_replay.add_argument("--scale", type=int, default=None,
                         help="Scale factor (default: show scaling analysis)")
    p_replay.add_argument("--max-scale", type=int, default=32,
                         help="Max scale for analysis (default: 32)")
    p_replay.add_argument("--no-calibrate", action="store_true",
                         help="Don't calibrate GPU profile from trace")

    # profiles (list available)
    p_profiles = subparsers.add_parser("profiles", help="List available hardware profiles")

    args = parser.parse_args()

    if args.command == "capture":
        if not args.cmd or args.cmd == ["--"]:
            parser.error("No command specified to capture")
        # Remove leading "--" if present
        if args.cmd and args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
        return cmd_capture(args)

    elif args.command == "analyze":
        return cmd_analyze(args)

    elif args.command == "replay":
        return cmd_replay(args)

    elif args.command == "profiles":
        return cmd_profiles(args)

    else:
        parser.print_help()
        return 0


def cmd_profiles(args):
    """List available hardware profiles."""
    from .profiles.gpu import GPU_PROFILES
    from .profiles.nccl import NETWORK_PROFILES

    print("\n=== GPU Profiles ===\n")
    for name, profile in GPU_PROFILES.items():
        print(f"  {name:<15} {profile.name}")
        print(f"                  FP16: {profile.peak_tflops_fp16:.0f} TFLOPS, "
              f"Mem: {profile.memory_gb:.0f} GB @ {profile.memory_bandwidth_gbps:.0f} GB/s")

    print("\n=== Network Profiles ===\n")
    for name, profile in NETWORK_PROFILES.items():
        print(f"  {name:<20} {profile.name}")
        print(f"                       Intra: {profile.intra_node_bandwidth_gbps:.0f} Gbps, "
              f"Inter: {profile.inter_node_bandwidth_gbps:.0f} Gbps")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
