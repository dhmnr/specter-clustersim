"""
NCCL Collective Operation Models

Analytical models for NCCL collective operations.
Based on published algorithms and calibrated with real benchmarks.

References:
- NCCL documentation
- "Bandwidth Optimal All-reduce Algorithms" (Patarasuk & Yuan)
- NVIDIA GTC presentations on NCCL internals
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


class NCCLAlgorithm(Enum):
    """NCCL algorithm selection."""
    RING = "ring"
    TREE = "tree"
    COLLNET = "collnet"  # Sharp/InfiniBand collective offload
    AUTO = "auto"


@dataclass
class NetworkProfile:
    """Network interconnect profile."""

    name: str

    # Intra-node interconnect (GPU-to-GPU within same server)
    intra_node_bandwidth_gbps: float  # e.g., 900 for NVLink 4.0
    intra_node_latency_us: float      # e.g., 1-2 us for NVLink

    # Inter-node interconnect (server-to-server)
    inter_node_bandwidth_gbps: float  # e.g., 400 for 400G InfiniBand
    inter_node_latency_us: float      # e.g., 2-5 us for IB

    # Topology info
    gpus_per_node: int = 8
    nvlink_topology: str = "full"  # "full", "hybrid", "none"

    # Collective offload support (InfiniBand SHARP, etc.)
    has_collnet: bool = False
    collnet_speedup: float = 1.5  # Typical SHARP speedup


# Pre-defined network profiles
NETWORK_PROFILES = {
    # DGX H100 cluster with 400G InfiniBand
    "dgx_h100_400g": NetworkProfile(
        name="DGX H100 (400G IB)",
        intra_node_bandwidth_gbps=900.0,  # NVLink 4.0
        intra_node_latency_us=1.5,
        inter_node_bandwidth_gbps=400.0,  # 400G HDR InfiniBand
        inter_node_latency_us=2.5,
        gpus_per_node=8,
        nvlink_topology="full",
        has_collnet=True,
        collnet_speedup=1.8,
    ),

    # DGX A100 cluster with 200G InfiniBand
    "dgx_a100_200g": NetworkProfile(
        name="DGX A100 (200G IB)",
        intra_node_bandwidth_gbps=600.0,  # NVLink 3.0
        intra_node_latency_us=2.0,
        inter_node_bandwidth_gbps=200.0,
        inter_node_latency_us=3.0,
        gpus_per_node=8,
        nvlink_topology="full",
        has_collnet=True,
        collnet_speedup=1.5,
    ),

    # Cloud instances with PCIe + ethernet
    "cloud_pcie_100g": NetworkProfile(
        name="Cloud PCIe (100G Ethernet)",
        intra_node_bandwidth_gbps=64.0,  # PCIe Gen4
        intra_node_latency_us=5.0,
        inter_node_bandwidth_gbps=100.0,
        inter_node_latency_us=10.0,
        gpus_per_node=8,
        nvlink_topology="none",
        has_collnet=False,
    ),

    # Consumer/research setup
    "consumer_pcie": NetworkProfile(
        name="Consumer PCIe",
        intra_node_bandwidth_gbps=32.0,  # PCIe Gen4 realistic
        intra_node_latency_us=8.0,
        inter_node_bandwidth_gbps=10.0,  # 10G ethernet
        inter_node_latency_us=50.0,
        gpus_per_node=2,
        nvlink_topology="none",
        has_collnet=False,
    ),

    # Single node (no network)
    "single_node_nvlink": NetworkProfile(
        name="Single Node NVLink",
        intra_node_bandwidth_gbps=900.0,
        intra_node_latency_us=1.5,
        inter_node_bandwidth_gbps=0.0,
        inter_node_latency_us=0.0,
        gpus_per_node=8,
        nvlink_topology="full",
        has_collnet=False,
    ),
}


def load_network_profile(name: str) -> NetworkProfile:
    """Load a network profile by name."""
    name_lower = name.lower().replace(" ", "_").replace("-", "_")

    if name_lower not in NETWORK_PROFILES:
        available = ", ".join(NETWORK_PROFILES.keys())
        raise ValueError(f"Unknown network profile: {name}. Available: {available}")

    return NETWORK_PROFILES[name_lower]


class NCCLModel:
    """
    Analytical model for NCCL collective operations.

    Models the time for various collectives based on:
    - Message size
    - World size
    - Network topology
    - Algorithm selection
    """

    def __init__(self, network: NetworkProfile):
        self.network = network

    def _get_bandwidth_and_latency(self, world_size: int) -> tuple[float, float]:
        """
        Get effective bandwidth and latency for a collective.

        For multi-node, we're limited by inter-node bandwidth.
        """
        num_nodes = math.ceil(world_size / self.network.gpus_per_node)

        if num_nodes == 1:
            # Single node: use intra-node interconnect
            return (
                self.network.intra_node_bandwidth_gbps,
                self.network.intra_node_latency_us,
            )
        else:
            # Multi-node: limited by inter-node (usually slower)
            # But intra-node still matters for hierarchical algorithms
            return (
                self.network.inter_node_bandwidth_gbps,
                self.network.inter_node_latency_us,
            )

    def allreduce(
        self,
        size_bytes: int,
        world_size: int,
        algorithm: NCCLAlgorithm = NCCLAlgorithm.AUTO,
    ) -> float:
        """
        Estimate AllReduce time.

        Ring AllReduce: 2(n-1)/n * size / bandwidth + 2(n-1) * latency
        Tree AllReduce: 2 * log(n) * size / bandwidth + 2 * log(n) * latency

        Returns time in microseconds.
        """
        if world_size <= 1:
            return 0.0

        bw_gbps, lat_us = self._get_bandwidth_and_latency(world_size)
        bw_bytes_per_us = bw_gbps * 1e9 / 8 / 1e6  # Convert to bytes/us

        # Algorithm selection
        if algorithm == NCCLAlgorithm.AUTO:
            # NCCL typically uses ring for large messages, tree for small
            if size_bytes < 256 * 1024:  # < 256KB
                algorithm = NCCLAlgorithm.TREE
            else:
                algorithm = NCCLAlgorithm.RING

        if algorithm == NCCLAlgorithm.RING:
            # Ring: 2(n-1)/n * size for data, 2(n-1) steps
            data_factor = 2.0 * (world_size - 1) / world_size
            num_steps = 2 * (world_size - 1)
            time_us = (size_bytes * data_factor) / bw_bytes_per_us + num_steps * lat_us

        elif algorithm == NCCLAlgorithm.TREE:
            # Tree: 2 * log(n) phases, each sends full data
            num_steps = 2 * math.ceil(math.log2(world_size))
            time_us = (size_bytes * 2) / bw_bytes_per_us + num_steps * lat_us

        elif algorithm == NCCLAlgorithm.COLLNET:
            # SHARP/CollNet: network does the reduction
            if self.network.has_collnet:
                # Roughly like tree but faster
                base_time = (size_bytes * 2) / bw_bytes_per_us
                time_us = base_time / self.network.collnet_speedup + lat_us * 4
            else:
                # Fallback to ring
                return self.allreduce(size_bytes, world_size, NCCLAlgorithm.RING)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return time_us

    def allgather(self, size_bytes: int, world_size: int) -> float:
        """
        Estimate AllGather time.

        Ring: (n-1)/n * total_size / bandwidth + (n-1) * latency
        where total_size = size_bytes * world_size

        Returns time in microseconds.
        """
        if world_size <= 1:
            return 0.0

        bw_gbps, lat_us = self._get_bandwidth_and_latency(world_size)
        bw_bytes_per_us = bw_gbps * 1e9 / 8 / 1e6

        # Each rank contributes size_bytes, total output is size_bytes * world_size
        # Ring algorithm: (n-1) steps, each sends size_bytes
        total_data = size_bytes * (world_size - 1)
        num_steps = world_size - 1

        time_us = total_data / bw_bytes_per_us + num_steps * lat_us
        return time_us

    def reduce_scatter(self, size_bytes: int, world_size: int) -> float:
        """
        Estimate ReduceScatter time.

        Similar to AllGather but with reduction.
        Ring: (n-1)/n * total_size / bandwidth + (n-1) * latency

        Returns time in microseconds.
        """
        if world_size <= 1:
            return 0.0

        bw_gbps, lat_us = self._get_bandwidth_and_latency(world_size)
        bw_bytes_per_us = bw_gbps * 1e9 / 8 / 1e6

        # Input is size_bytes * world_size, output is size_bytes per rank
        total_data = size_bytes * (world_size - 1)
        num_steps = world_size - 1

        time_us = total_data / bw_bytes_per_us + num_steps * lat_us
        return time_us

    def broadcast(self, size_bytes: int, world_size: int) -> float:
        """
        Estimate Broadcast time.

        Tree: log(n) steps, each sends full size.

        Returns time in microseconds.
        """
        if world_size <= 1:
            return 0.0

        bw_gbps, lat_us = self._get_bandwidth_and_latency(world_size)
        bw_bytes_per_us = bw_gbps * 1e9 / 8 / 1e6

        num_steps = math.ceil(math.log2(world_size))
        time_us = size_bytes / bw_bytes_per_us + num_steps * lat_us
        return time_us

    def reduce(self, size_bytes: int, world_size: int, root: int = 0) -> float:
        """
        Estimate Reduce time (all-to-one).

        Tree: log(n) steps with reduction.

        Returns time in microseconds.
        """
        if world_size <= 1:
            return 0.0

        bw_gbps, lat_us = self._get_bandwidth_and_latency(world_size)
        bw_bytes_per_us = bw_gbps * 1e9 / 8 / 1e6

        num_steps = math.ceil(math.log2(world_size))
        time_us = size_bytes / bw_bytes_per_us + num_steps * lat_us
        return time_us

    def send(self, size_bytes: int, src_rank: int, dst_rank: int) -> float:
        """
        Estimate point-to-point Send time.

        Returns time in microseconds.
        """
        # Check if same node
        src_node = src_rank // self.network.gpus_per_node
        dst_node = dst_rank // self.network.gpus_per_node

        if src_node == dst_node:
            bw_gbps = self.network.intra_node_bandwidth_gbps
            lat_us = self.network.intra_node_latency_us
        else:
            bw_gbps = self.network.inter_node_bandwidth_gbps
            lat_us = self.network.inter_node_latency_us

        bw_bytes_per_us = bw_gbps * 1e9 / 8 / 1e6
        time_us = size_bytes / bw_bytes_per_us + lat_us
        return time_us

    def recv(self, size_bytes: int, src_rank: int, dst_rank: int) -> float:
        """Estimate Recv time (same as Send)."""
        return self.send(size_bytes, src_rank, dst_rank)

    def estimate_op(
        self,
        op_type: str,
        size_bytes: int,
        world_size: int,
        **kwargs,
    ) -> float:
        """
        Estimate any NCCL operation time.

        Args:
            op_type: "AllReduce", "AllGather", "ReduceScatter", "Broadcast", "Send", "Recv"
            size_bytes: Data size in bytes
            world_size: Number of participants

        Returns:
            Estimated time in microseconds
        """
        op_type_lower = op_type.lower()

        if "allreduce" in op_type_lower:
            return self.allreduce(size_bytes, world_size)
        elif "allgather" in op_type_lower:
            return self.allgather(size_bytes, world_size)
        elif "reducescatter" in op_type_lower:
            return self.reduce_scatter(size_bytes, world_size)
        elif "broadcast" in op_type_lower:
            return self.broadcast(size_bytes, world_size)
        elif "reduce" in op_type_lower:
            return self.reduce(size_bytes, world_size)
        elif "send" in op_type_lower:
            peer = kwargs.get("peer", 1)
            return self.send(size_bytes, 0, peer)
        elif "recv" in op_type_lower:
            peer = kwargs.get("peer", 0)
            return self.recv(size_bytes, peer, 0)
        else:
            raise ValueError(f"Unknown NCCL operation: {op_type}")


def print_nccl_estimates(world_size: int = 8, network: str = "dgx_h100_400g"):
    """Print NCCL timing estimates for various message sizes."""
    net = load_network_profile(network)
    model = NCCLModel(net)

    sizes = [
        (1024, "1 KB"),
        (1024 * 1024, "1 MB"),
        (16 * 1024 * 1024, "16 MB"),
        (128 * 1024 * 1024, "128 MB"),
        (1024 * 1024 * 1024, "1 GB"),
    ]

    print(f"\nNCCL Timing Estimates ({net.name}, {world_size} GPUs)")
    print("=" * 70)
    print(f"{'Size':<12} {'AllReduce':<12} {'AllGather':<12} {'ReduceScatter':<14} {'Broadcast':<12}")
    print("-" * 70)

    for size, name in sizes:
        ar = model.allreduce(size, world_size)
        ag = model.allgather(size, world_size)
        rs = model.reduce_scatter(size, world_size)
        bc = model.broadcast(size, world_size)

        def fmt(us):
            if us < 1000:
                return f"{us:.1f} us"
            else:
                return f"{us/1000:.2f} ms"

        print(f"{name:<12} {fmt(ar):<12} {fmt(ag):<12} {fmt(rs):<14} {fmt(bc):<12}")

    print()


if __name__ == "__main__":
    # Demo
    print_nccl_estimates(8, "dgx_h100_400g")
    print_nccl_estimates(64, "dgx_h100_400g")
    print_nccl_estimates(256, "dgx_h100_400g")
