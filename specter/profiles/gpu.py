"""
GPU Performance Profiles

Pre-calibrated performance data for common GPUs.
Data sourced from official specs + real benchmarks.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class GPUProfile:
    """Performance profile for a GPU."""

    name: str

    # Peak theoretical TFLOPS
    peak_tflops_fp16: float
    peak_tflops_bf16: float
    peak_tflops_fp32: float

    # Memory
    memory_gb: float
    memory_bandwidth_gbps: float

    # Optional fields with defaults
    peak_tflops_tf32: float = 0.0  # Ampere+ only
    peak_tflops_fp8: float = 0.0   # Hopper+ only

    # Interconnect (intra-node)
    nvlink_bandwidth_gbps: float = 0.0  # 0 = no NVLink (PCIe only)
    pcie_bandwidth_gbps: float = 64.0   # PCIe Gen4 x16 default

    # Calibrated efficiency curves
    # Maps arithmetic intensity (FLOPs/byte) to efficiency (0-1)
    # Derived from real benchmarks
    gemm_efficiency: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    # Known GEMM overrides (shape -> efficiency)
    # For shapes that don't follow the curve well
    gemm_overrides: dict[str, float] = field(default_factory=dict)

    # Kernel launch overhead
    launch_overhead_us: float = 3.0

    def get_peak_tflops(self, dtype: str) -> float:
        """Get peak TFLOPS for a given dtype."""
        dtype_map = {
            "fp16": self.peak_tflops_fp16,
            "bf16": self.peak_tflops_bf16,
            "fp32": self.peak_tflops_fp32,
            "tf32": self.peak_tflops_tf32,
            "fp8": self.peak_tflops_fp8,
        }
        return dtype_map.get(dtype, self.peak_tflops_fp16)

    def get_dtype_bytes(self, dtype: str) -> int:
        """Get bytes per element for dtype."""
        dtype_bytes = {
            "fp8": 1,
            "fp16": 2,
            "bf16": 2,
            "fp32": 4,
            "tf32": 4,  # stored as fp32
            "fp64": 8,
        }
        return dtype_bytes.get(dtype, 2)

    def estimate_gemm_time_us(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "fp16",
        batch: int = 1,
    ) -> float:
        """
        Estimate GEMM execution time using roofline model.

        Args:
            m, n, k: Matrix dimensions (C[m,n] = A[m,k] @ B[k,n])
            dtype: Data type (fp16, bf16, fp32, etc.)
            batch: Batch size for batched GEMM

        Returns:
            Estimated time in microseconds
        """
        # Check for override first
        shape_key = f"{m}x{n}x{k}"
        if batch > 1:
            shape_key += f"x{batch}"

        # Total FLOPs: 2*M*N*K per GEMM (multiply + add)
        flops = 2.0 * m * n * k * batch

        # Bytes accessed (assuming no reuse from cache)
        # A: m*k, B: k*n, C: m*n (read + write)
        bytes_per_elem = self.get_dtype_bytes(dtype)
        bytes_accessed = (m * k + k * n + 2 * m * n) * bytes_per_elem * batch

        # Arithmetic intensity (FLOPs per byte)
        intensity = flops / bytes_accessed if bytes_accessed > 0 else float('inf')

        # Get efficiency from calibration curve or use default
        efficiency = self._get_efficiency(intensity, dtype, shape_key)

        # Peak performance for this dtype
        peak_tflops = self.get_peak_tflops(dtype)

        # Roofline: time = max(compute_time, memory_time)
        # But we fold memory into efficiency, so:
        achieved_tflops = peak_tflops * efficiency

        if achieved_tflops <= 0:
            return float('inf')

        time_us = flops / (achieved_tflops * 1e6)

        # Add kernel launch overhead
        time_us += self.launch_overhead_us

        return time_us

    def _get_efficiency(self, intensity: float, dtype: str, shape_key: str) -> float:
        """Get efficiency from calibration curve."""

        # Check override first
        if shape_key in self.gemm_overrides:
            return self.gemm_overrides[shape_key]

        # Get curve for this dtype (or default)
        curve = self.gemm_efficiency.get(dtype, self.gemm_efficiency.get("default", []))

        if not curve:
            # Default efficiency curve if not calibrated
            # Based on typical GPU behavior
            if intensity < 10:
                return 0.3  # Memory bound
            elif intensity < 50:
                return 0.5 + (intensity - 10) * 0.01  # Transitional
            elif intensity < 200:
                return 0.7 + (intensity - 50) * 0.001  # Compute bound
            else:
                return 0.85  # Near peak

        # Interpolate from curve
        for i, (thresh, eff) in enumerate(curve):
            if intensity <= thresh:
                if i == 0:
                    return eff
                prev_thresh, prev_eff = curve[i - 1]
                # Linear interpolation
                t = (intensity - prev_thresh) / (thresh - prev_thresh)
                return prev_eff + t * (eff - prev_eff)

        # Beyond last threshold
        return curve[-1][1]


# =============================================================================
# Pre-defined GPU Profiles
# Data from official specs + MLPerf/public benchmarks
# =============================================================================

GPU_PROFILES = {
    # NVIDIA H100 SXM (Hopper)
    "h100_sxm": GPUProfile(
        name="NVIDIA H100 SXM",
        peak_tflops_fp16=1979.0,
        peak_tflops_bf16=1979.0,
        peak_tflops_fp32=67.0,
        peak_tflops_tf32=989.0,
        peak_tflops_fp8=3958.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=3350.0,
        nvlink_bandwidth_gbps=900.0,  # NVLink 4.0, 18 links
        pcie_bandwidth_gbps=128.0,     # PCIe Gen5
        launch_overhead_us=2.0,
        gemm_efficiency={
            "default": [
                (10, 0.35),
                (50, 0.65),
                (100, 0.80),
                (200, 0.88),
                (500, 0.92),
            ],
        },
    ),

    # NVIDIA H100 PCIe
    "h100_pcie": GPUProfile(
        name="NVIDIA H100 PCIe",
        peak_tflops_fp16=1513.0,
        peak_tflops_bf16=1513.0,
        peak_tflops_fp32=51.0,
        peak_tflops_tf32=756.0,
        peak_tflops_fp8=3026.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2000.0,
        nvlink_bandwidth_gbps=0.0,  # No NVLink on PCIe variant
        pcie_bandwidth_gbps=128.0,
        launch_overhead_us=2.0,
        gemm_efficiency={
            "default": [
                (10, 0.30),
                (50, 0.60),
                (100, 0.75),
                (200, 0.85),
                (500, 0.90),
            ],
        },
    ),

    # NVIDIA A100 SXM (Ampere)
    "a100_sxm": GPUProfile(
        name="NVIDIA A100 SXM",
        peak_tflops_fp16=312.0,
        peak_tflops_bf16=312.0,
        peak_tflops_fp32=19.5,
        peak_tflops_tf32=156.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2039.0,
        nvlink_bandwidth_gbps=600.0,  # NVLink 3.0, 12 links
        pcie_bandwidth_gbps=64.0,
        launch_overhead_us=2.5,
        gemm_efficiency={
            "default": [
                (10, 0.30),
                (50, 0.60),
                (100, 0.78),
                (200, 0.85),
                (500, 0.90),
            ],
        },
    ),

    # NVIDIA A100 PCIe
    "a100_pcie": GPUProfile(
        name="NVIDIA A100 PCIe",
        peak_tflops_fp16=312.0,
        peak_tflops_bf16=312.0,
        peak_tflops_fp32=19.5,
        peak_tflops_tf32=156.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=1935.0,
        nvlink_bandwidth_gbps=0.0,
        pcie_bandwidth_gbps=64.0,
        launch_overhead_us=2.5,
        gemm_efficiency={
            "default": [
                (10, 0.28),
                (50, 0.55),
                (100, 0.72),
                (200, 0.82),
                (500, 0.88),
            ],
        },
    ),

    # NVIDIA RTX 4090 (Ada Lovelace)
    "rtx_4090": GPUProfile(
        name="NVIDIA RTX 4090",
        peak_tflops_fp16=165.0,  # With sparsity: 330
        peak_tflops_bf16=165.0,
        peak_tflops_fp32=82.6,
        peak_tflops_tf32=82.6,
        memory_gb=24.0,
        memory_bandwidth_gbps=1008.0,
        nvlink_bandwidth_gbps=0.0,  # Consumer card, no NVLink
        pcie_bandwidth_gbps=64.0,
        launch_overhead_us=3.0,
        gemm_efficiency={
            "default": [
                (10, 0.25),
                (50, 0.55),
                (100, 0.75),
                (200, 0.85),
                (500, 0.90),
            ],
        },
    ),

    # NVIDIA RTX 3090 (Ampere consumer)
    "rtx_3090": GPUProfile(
        name="NVIDIA RTX 3090",
        peak_tflops_fp16=71.0,
        peak_tflops_bf16=71.0,
        peak_tflops_fp32=35.6,
        peak_tflops_tf32=35.6,
        memory_gb=24.0,
        memory_bandwidth_gbps=936.0,
        nvlink_bandwidth_gbps=0.0,
        pcie_bandwidth_gbps=64.0,
        launch_overhead_us=3.5,
        gemm_efficiency={
            "default": [
                (10, 0.22),
                (50, 0.50),
                (100, 0.70),
                (200, 0.80),
                (500, 0.85),
            ],
        },
    ),

    # NVIDIA V100 SXM (Volta)
    "v100_sxm": GPUProfile(
        name="NVIDIA V100 SXM",
        peak_tflops_fp16=125.0,
        peak_tflops_bf16=0.0,  # No BF16 on Volta
        peak_tflops_fp32=15.7,
        memory_gb=32.0,
        memory_bandwidth_gbps=900.0,
        nvlink_bandwidth_gbps=300.0,  # NVLink 2.0
        pcie_bandwidth_gbps=32.0,
        launch_overhead_us=4.0,
        gemm_efficiency={
            "default": [
                (10, 0.25),
                (50, 0.55),
                (100, 0.70),
                (200, 0.80),
                (500, 0.85),
            ],
        },
    ),

    # AMD MI300X
    "mi300x": GPUProfile(
        name="AMD MI300X",
        peak_tflops_fp16=1307.0,
        peak_tflops_bf16=1307.0,
        peak_tflops_fp32=163.4,
        peak_tflops_fp8=2614.0,
        memory_gb=192.0,
        memory_bandwidth_gbps=5300.0,
        nvlink_bandwidth_gbps=0.0,  # Uses Infinity Fabric
        pcie_bandwidth_gbps=128.0,
        launch_overhead_us=3.0,
        gemm_efficiency={
            "default": [
                (10, 0.30),
                (50, 0.58),
                (100, 0.75),
                (200, 0.85),
                (500, 0.88),
            ],
        },
    ),
}


def load_gpu_profile(name: str) -> GPUProfile:
    """Load a GPU profile by name."""
    name_lower = name.lower().replace(" ", "_").replace("-", "_")

    if name_lower not in GPU_PROFILES:
        available = ", ".join(GPU_PROFILES.keys())
        raise ValueError(f"Unknown GPU profile: {name}. Available: {available}")

    return GPU_PROFILES[name_lower]


def list_gpu_profiles() -> list[str]:
    """List available GPU profile names."""
    return list(GPU_PROFILES.keys())


def calibrate_from_trace(profile: GPUProfile, trace_ops: list[dict]) -> GPUProfile:
    """
    Calibrate a GPU profile using real trace data.

    This adjusts the efficiency curve to match observed performance.
    """
    # Collect (intensity, efficiency) points from trace
    points = []

    for op in trace_ops:
        if op.get("type") != "cublas" or "Gemm" not in op.get("op", ""):
            continue

        m = op.get("m", 0)
        n = op.get("n", 0)
        k = op.get("k", 0)
        batch = op.get("batch", 1)
        duration_us = op.get("duration_us", 0)

        if m == 0 or duration_us == 0:
            continue

        # Calculate actual TFLOPS achieved
        flops = 2.0 * m * n * k * batch
        actual_tflops = flops / (duration_us * 1e6)

        # Calculate efficiency vs peak
        dtype = op.get("dtype_a", "fp16")
        peak = profile.get_peak_tflops(dtype)
        efficiency = actual_tflops / peak if peak > 0 else 0

        # Calculate arithmetic intensity
        bytes_per_elem = profile.get_dtype_bytes(dtype)
        bytes_accessed = (m * k + k * n + 2 * m * n) * bytes_per_elem * batch
        intensity = flops / bytes_accessed if bytes_accessed > 0 else 0

        if 0 < efficiency <= 1.0:  # Sanity check
            points.append((intensity, efficiency))

    if not points:
        return profile

    # Sort by intensity and compute rolling average
    points.sort(key=lambda x: x[0])

    # Build new efficiency curve
    # Group into buckets and average
    buckets = [10, 25, 50, 100, 200, 500, 1000]
    new_curve = []

    for thresh in buckets:
        bucket_points = [e for i, e in points if i <= thresh]
        if bucket_points:
            avg_eff = sum(bucket_points) / len(bucket_points)
            new_curve.append((thresh, min(0.95, avg_eff)))  # Cap at 95%

    # Create calibrated profile
    calibrated = GPUProfile(
        name=f"{profile.name} (calibrated)",
        peak_tflops_fp16=profile.peak_tflops_fp16,
        peak_tflops_bf16=profile.peak_tflops_bf16,
        peak_tflops_fp32=profile.peak_tflops_fp32,
        peak_tflops_tf32=profile.peak_tflops_tf32,
        peak_tflops_fp8=profile.peak_tflops_fp8,
        memory_gb=profile.memory_gb,
        memory_bandwidth_gbps=profile.memory_bandwidth_gbps,
        nvlink_bandwidth_gbps=profile.nvlink_bandwidth_gbps,
        pcie_bandwidth_gbps=profile.pcie_bandwidth_gbps,
        launch_overhead_us=profile.launch_overhead_us,
        gemm_efficiency={"default": new_curve},
        gemm_overrides=profile.gemm_overrides,
    )

    return calibrated
