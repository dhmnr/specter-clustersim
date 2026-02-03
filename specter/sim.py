"""
SPECTER Simulator

Simulate GPU workloads on arbitrary cluster configurations.

Usage:
    # From trace (recorded on real hardware)
    sim = Simulator.from_trace("./traces", cluster_config)
    result = sim.replay(scale=4)

    # From model (no real hardware needed)
    sim = Simulator.from_profile("h100_sxm", "dgx_h100_400g", world_size=256)
    result = sim.estimate_gemm(4096, 4096, 4096, count=100)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

from .profiles.gpu import GPUProfile, load_gpu_profile, calibrate_from_trace
from .profiles.nccl import NCCLModel, NetworkProfile, load_network_profile


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    # Timing breakdown (all in milliseconds)
    compute_time_ms: float = 0.0
    nccl_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Efficiency metrics
    compute_efficiency: float = 0.0  # Achieved vs peak TFLOPS
    network_efficiency: float = 0.0  # Achieved vs peak bandwidth

    # Scaling info
    world_size: int = 1
    gpus_per_node: int = 8
    num_nodes: int = 1

    # Operation counts
    gemm_count: int = 0
    nccl_op_count: int = 0

    # Detailed breakdown (optional)
    gemm_breakdown: list[dict] = field(default_factory=list)
    nccl_breakdown: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "",
            "=" * 60,
            "  SPECTER Simulation Result",
            "=" * 60,
            "",
            f"  Cluster: {self.num_nodes} nodes × {self.gpus_per_node} GPUs = {self.world_size} total",
            "",
            "  Timing:",
            f"    Compute:       {self.compute_time_ms:10.2f} ms",
            f"    Communication: {self.nccl_time_ms:10.2f} ms",
            f"    Total:         {self.total_time_ms:10.2f} ms",
            "",
            "  Efficiency:",
            f"    Compute:       {self.compute_efficiency * 100:10.1f}%",
            f"    Network:       {self.network_efficiency * 100:10.1f}%",
            "",
            "  Operations:",
            f"    GEMMs:         {self.gemm_count:10d}",
            f"    NCCL ops:      {self.nccl_op_count:10d}",
            "",
        ]
        return "\n".join(lines)


class Simulator:
    """
    SPECTER Workload Simulator.

    Can simulate workloads using:
    1. Real traces (captured from actual hardware)
    2. Pre-stored GPU profiles (when no hardware access)
    3. Analytical models (pure calculation)
    """

    def __init__(
        self,
        gpu: GPUProfile,
        network: NetworkProfile,
        world_size: int = 1,
    ):
        self.gpu = gpu
        self.network = network
        self.world_size = world_size
        self.nccl = NCCLModel(network)

        # Optional: trace data for replay
        self.trace_gemms: list[dict] = []
        self.trace_nccl: list[dict] = []

    @classmethod
    def from_profile(
        cls,
        gpu_name: str,
        network_name: str,
        world_size: int = 1,
    ) -> "Simulator":
        """
        Create simulator from pre-stored profiles.

        Args:
            gpu_name: GPU profile name (e.g., "h100_sxm", "rtx_4090")
            network_name: Network profile name (e.g., "dgx_h100_400g")
            world_size: Number of GPUs
        """
        gpu = load_gpu_profile(gpu_name)
        network = load_network_profile(network_name)
        return cls(gpu, network, world_size)

    @classmethod
    def from_trace(
        cls,
        trace_dir: str | Path,
        gpu_name: str,
        network_name: str,
        world_size: int = 1,
        calibrate: bool = True,
    ) -> "Simulator":
        """
        Create simulator from captured traces.

        Args:
            trace_dir: Directory containing trace files
            gpu_name: GPU profile to use (for scaling)
            network_name: Network profile name
            world_size: Target world size for simulation
            calibrate: Whether to calibrate GPU profile from trace
        """
        trace_dir = Path(trace_dir)
        gpu = load_gpu_profile(gpu_name)
        network = load_network_profile(network_name)

        sim = cls(gpu, network, world_size)

        # Load trace files
        for trace_file in trace_dir.glob("specter_*.jsonl"):
            with open(trace_file) as f:
                for line in f:
                    try:
                        op = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if op.get("type") == "cublas" or "Gemm" in op.get("op", ""):
                        sim.trace_gemms.append(op)
                    elif op.get("type") == "nccl" or op.get("op") in (
                        "AllReduce", "AllGather", "ReduceScatter",
                        "Broadcast", "Send", "Recv"
                    ):
                        sim.trace_nccl.append(op)

        # Optionally calibrate GPU profile from trace
        if calibrate and sim.trace_gemms:
            sim.gpu = calibrate_from_trace(gpu, sim.trace_gemms)

        return sim

    def estimate_gemm(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "fp16",
        batch: int = 1,
        count: int = 1,
    ) -> SimulationResult:
        """
        Estimate time for GEMM operations.

        Args:
            m, n, k: Matrix dimensions
            dtype: Data type
            batch: Batch size
            count: Number of GEMM operations
        """
        time_per_op_us = self.gpu.estimate_gemm_time_us(m, n, k, dtype, batch)
        total_time_us = time_per_op_us * count

        # Calculate efficiency
        flops = 2.0 * m * n * k * batch * count
        achieved_tflops = flops / (total_time_us * 1e6)
        peak_tflops = self.gpu.get_peak_tflops(dtype)
        efficiency = achieved_tflops / peak_tflops if peak_tflops > 0 else 0

        return SimulationResult(
            compute_time_ms=total_time_us / 1000,
            total_time_ms=total_time_us / 1000,
            compute_efficiency=efficiency,
            world_size=1,
            gpus_per_node=1,
            num_nodes=1,
            gemm_count=count,
        )

    def estimate_allreduce(
        self,
        size_bytes: int,
        count: int = 1,
    ) -> SimulationResult:
        """Estimate time for AllReduce operations."""
        time_per_op_us = self.nccl.allreduce(size_bytes, self.world_size)
        total_time_us = time_per_op_us * count

        # Calculate bandwidth efficiency
        # Ring AllReduce moves 2(n-1)/n * size bytes
        data_factor = 2 * (self.world_size - 1) / self.world_size
        total_bytes = size_bytes * data_factor * count
        achieved_bw = total_bytes / (total_time_us * 1e-6) / 1e9  # GB/s

        num_nodes = (self.world_size + self.network.gpus_per_node - 1) // self.network.gpus_per_node
        if num_nodes > 1:
            peak_bw = self.network.inter_node_bandwidth_gbps / 8  # GB/s
        else:
            peak_bw = self.network.intra_node_bandwidth_gbps / 8  # GB/s

        efficiency = achieved_bw / peak_bw if peak_bw > 0 else 0

        return SimulationResult(
            nccl_time_ms=total_time_us / 1000,
            total_time_ms=total_time_us / 1000,
            network_efficiency=min(1.0, efficiency),
            world_size=self.world_size,
            gpus_per_node=self.network.gpus_per_node,
            num_nodes=num_nodes,
            nccl_op_count=count,
        )

    def replay_trace(self, scale: int = 1) -> SimulationResult:
        """
        Replay captured trace on simulated cluster.

        Args:
            scale: Scale factor for world size (original × scale)

        Returns:
            SimulationResult with timing estimates
        """
        if not self.trace_gemms and not self.trace_nccl:
            raise ValueError("No trace data loaded. Use from_trace() to load traces.")

        # Scale world size
        # Trace was captured at original_world_size
        # We simulate at original_world_size × scale
        original_world = 1
        for op in self.trace_nccl:
            if "comm_size" in op:
                original_world = max(original_world, op["comm_size"])

        scaled_world = original_world * scale
        self.world_size = scaled_world

        result = SimulationResult(
            world_size=scaled_world,
            gpus_per_node=self.network.gpus_per_node,
            num_nodes=(scaled_world + self.network.gpus_per_node - 1) // self.network.gpus_per_node,
        )

        # Replay GEMMs (compute time doesn't change with scale)
        total_compute_us = 0
        total_flops = 0

        for op in self.trace_gemms:
            m = op.get("m", 0)
            n = op.get("n", 0)
            k = op.get("k", 0)
            batch = op.get("batch", 1)
            dtype = op.get("dtype_a", "fp16")

            if m == 0:
                continue

            time_us = self.gpu.estimate_gemm_time_us(m, n, k, dtype, batch)
            total_compute_us += time_us
            total_flops += 2 * m * n * k * batch

            result.gemm_breakdown.append({
                "m": m, "n": n, "k": k, "batch": batch,
                "time_us": time_us,
            })

        result.compute_time_ms = total_compute_us / 1000
        result.gemm_count = len(self.trace_gemms)

        # Calculate compute efficiency
        if total_compute_us > 0:
            achieved_tflops = total_flops / (total_compute_us * 1e6)
            peak_tflops = self.gpu.get_peak_tflops("fp16")
            result.compute_efficiency = achieved_tflops / peak_tflops if peak_tflops > 0 else 0

        # Replay NCCL (communication time changes with scale)
        total_nccl_us = 0
        total_bytes = 0

        for op in self.trace_nccl:
            op_type = op.get("op", "")
            size_bytes = op.get("bytes", 0)
            orig_comm_size = op.get("comm_size", 1)

            if size_bytes == 0:
                continue

            # Scale comm_size if this was a data-parallel collective
            # (Tensor parallel collectives don't scale with DP)
            # Heuristic: if original comm_size == original_world, it's DP
            if orig_comm_size == original_world:
                scaled_comm_size = scaled_world
            else:
                scaled_comm_size = orig_comm_size  # TP stays same size

            time_us = self.nccl.estimate_op(op_type, size_bytes, scaled_comm_size)
            total_nccl_us += time_us
            total_bytes += size_bytes

            result.nccl_breakdown.append({
                "op": op_type,
                "bytes": size_bytes,
                "orig_comm_size": orig_comm_size,
                "scaled_comm_size": scaled_comm_size,
                "time_us": time_us,
            })

        result.nccl_time_ms = total_nccl_us / 1000
        result.nccl_op_count = len(self.trace_nccl)

        # Calculate network efficiency
        if total_nccl_us > 0 and total_bytes > 0:
            achieved_bw = total_bytes / (total_nccl_us * 1e-6) / 1e9
            if result.num_nodes > 1:
                peak_bw = self.network.inter_node_bandwidth_gbps / 8
            else:
                peak_bw = self.network.intra_node_bandwidth_gbps / 8
            result.network_efficiency = min(1.0, achieved_bw / peak_bw) if peak_bw > 0 else 0

        # Total time (assuming sequential - no overlap)
        result.total_time_ms = result.compute_time_ms + result.nccl_time_ms

        return result

    def scaling_analysis(
        self,
        scales: list[int] = [1, 2, 4, 8, 16],
    ) -> list[SimulationResult]:
        """
        Analyze scaling behavior across multiple cluster sizes.

        Returns list of results for each scale factor.
        """
        results = []
        for scale in scales:
            result = self.replay_trace(scale)
            results.append(result)
        return results


def print_scaling_analysis(results: list[SimulationResult], base_world: int = 8):
    """Print scaling analysis results."""
    print()
    print("=" * 80)
    print("  SPECTER Scaling Analysis")
    print("=" * 80)
    print()
    print(f"{'GPUs':<10} {'Compute':<12} {'Comm':<12} {'Total':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 80)

    base_time = results[0].total_time_ms if results else 1

    for result in results:
        speedup = base_time / result.total_time_ms if result.total_time_ms > 0 else 0
        scale = result.world_size / base_world
        ideal_speedup = scale
        efficiency = speedup / ideal_speedup if ideal_speedup > 0 else 0

        print(f"{result.world_size:<10} "
              f"{result.compute_time_ms:>8.2f} ms  "
              f"{result.nccl_time_ms:>8.2f} ms  "
              f"{result.total_time_ms:>8.2f} ms  "
              f"{speedup:>6.2f}x     "
              f"{efficiency * 100:>6.1f}%")

    print()
