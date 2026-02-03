#!/usr/bin/env python3
"""
SPECTER Multi-GPU Benchmark

Exercises cuBLAS and NCCL operations for comprehensive tracing.
Run with: torchrun --nproc_per_node=N bench_multigpu.py

Options:
  --steps N        Number of iterations per test (default: 10)
  --warmup N       Warmup iterations (default: 3)
  --quick          Quick mode (fewer tests)
  --gemm-only      Only run GEMM tests
  --nccl-only      Only run NCCL tests
"""

import argparse
import os
import time
import torch
import torch.distributed as dist


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def log(msg):
    if get_rank() == 0:
        print(msg, flush=True)


def sync():
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()


def bench_gemm(steps=10, warmup=3, quick=False):
    """Benchmark various GEMM shapes."""
    log("\n" + "=" * 60)
    log("  cuBLAS GEMM Benchmark")
    log("=" * 60)

    device = torch.device("cuda")

    if quick:
        shapes = [
            (4096, 4096, 4096, 1),
            (8192, 8192, 8192, 1),
        ]
    else:
        shapes = [
            # Square GEMMs (common in attention)
            (2048, 2048, 2048, 1),
            (4096, 4096, 4096, 1),
            (8192, 8192, 8192, 1),

            # Wide GEMMs (FFN up projection)
            (4096, 11008, 4096, 1),    # LLaMA-7B FFN
            (4096, 14336, 4096, 1),    # LLaMA-13B FFN
            (8192, 28672, 8192, 1),    # LLaMA-70B FFN

            # Tall GEMMs (FFN down projection)
            (4096, 4096, 11008, 1),
            (8192, 8192, 28672, 1),

            # Batched GEMMs (attention)
            (2048, 2048, 128, 32),     # seq=2048, heads=32, head_dim=128
            (4096, 4096, 128, 32),     # seq=4096
            (2048, 2048, 128, 64),     # heads=64 (larger model)
        ]

    log(f"\n{'Shape':<30} {'Time (ms)':<12} {'TFLOPS':<10}")
    log("-" * 60)

    for m, n, k, batch in shapes:
        # Allocate
        if batch > 1:
            a = torch.randn(batch, m, k, dtype=torch.float16, device=device)
            b = torch.randn(batch, k, n, dtype=torch.float16, device=device)
        else:
            a = torch.randn(m, k, dtype=torch.float16, device=device)
            b = torch.randn(k, n, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(warmup):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(steps):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Calculate metrics
        time_ms = elapsed / steps * 1000
        flops = 2.0 * m * n * k * batch
        tflops = flops / (time_ms / 1000) / 1e12

        shape_str = f"{m}x{n}x{k}" + (f"x{batch}" if batch > 1 else "")
        log(f"{shape_str:<30} {time_ms:<12.3f} {tflops:<10.1f}")

    sync()


def bench_nccl(steps=10, warmup=3, quick=False):
    """Benchmark NCCL collective operations."""
    if not dist.is_initialized():
        log("\n[SKIP] NCCL benchmark requires distributed mode")
        return

    world_size = get_world_size()
    rank = get_rank()

    log("\n" + "=" * 60)
    log(f"  NCCL Benchmark (world_size={world_size})")
    log("=" * 60)

    device = torch.device("cuda")

    if quick:
        sizes_mb = [1, 128]
    else:
        sizes_mb = [1, 4, 16, 64, 128, 256, 512]

    # AllReduce
    log(f"\n--- AllReduce ---")
    log(f"{'Size':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<16} {'Bus BW (GB/s)':<14}")
    log("-" * 60)

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        count = size_bytes // 2  # fp16

        tensor = torch.randn(count, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(steps):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_ms = elapsed / steps * 1000

        # Algobw: size / time
        algobw = size_bytes / (time_ms / 1000) / 1e9

        # Busbw: 2*(n-1)/n * size / time (for ring allreduce)
        busbw = algobw * 2 * (world_size - 1) / world_size

        log(f"{size_mb:<12} MB {time_ms:<12.3f} {algobw:<16.1f} {busbw:<14.1f}")

    sync()

    # AllGather
    log(f"\n--- AllGather ---")
    log(f"{'Size':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<16}")
    log("-" * 60)

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        count = size_bytes // 2

        send_tensor = torch.randn(count, dtype=torch.float16, device=device)
        recv_tensors = [torch.zeros(count, dtype=torch.float16, device=device)
                        for _ in range(world_size)]

        # Warmup
        for _ in range(warmup):
            dist.all_gather(recv_tensors, send_tensor)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(steps):
            dist.all_gather(recv_tensors, send_tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_ms = elapsed / steps * 1000
        total_bytes = size_bytes * world_size
        bw = total_bytes / (time_ms / 1000) / 1e9

        log(f"{size_mb:<12} MB {time_ms:<12.3f} {bw:<16.1f}")

    sync()

    # ReduceScatter
    log(f"\n--- ReduceScatter ---")
    log(f"{'Size':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<16}")
    log("-" * 60)

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        count = size_bytes // 2

        # Input: list of tensors, one per rank's contribution
        input_tensors = [torch.randn(count // world_size, dtype=torch.float16, device=device)
                         for _ in range(world_size)]
        output_tensor = torch.zeros(count // world_size, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(warmup):
            dist.reduce_scatter(output_tensor, input_tensors)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(steps):
            dist.reduce_scatter(output_tensor, input_tensors)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_ms = elapsed / steps * 1000
        bw = size_bytes / (time_ms / 1000) / 1e9

        log(f"{size_mb:<12} MB {time_ms:<12.3f} {bw:<16.1f}")

    sync()

    # Point-to-point (if >= 2 GPUs)
    if world_size >= 2:
        log(f"\n--- Send/Recv (rank 0 <-> rank 1) ---")
        log(f"{'Size':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<16}")
        log("-" * 60)

        for size_mb in sizes_mb[:4]:  # Only test smaller sizes for p2p
            size_bytes = size_mb * 1024 * 1024
            count = size_bytes // 2

            tensor = torch.randn(count, dtype=torch.float16, device=device)

            # Warmup
            for _ in range(warmup):
                if rank == 0:
                    dist.send(tensor, dst=1)
                    dist.recv(tensor, src=1)
                elif rank == 1:
                    dist.recv(tensor, src=0)
                    dist.send(tensor, dst=0)
                dist.barrier()
            torch.cuda.synchronize()

            # Benchmark (round trip)
            start = time.perf_counter()
            for _ in range(steps):
                if rank == 0:
                    dist.send(tensor, dst=1)
                    dist.recv(tensor, src=1)
                elif rank == 1:
                    dist.recv(tensor, src=0)
                    dist.send(tensor, dst=0)
                dist.barrier()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            time_ms = elapsed / steps * 1000
            # Round trip = 2x the data
            bw = (2 * size_bytes) / (time_ms / 1000) / 1e9

            if rank == 0:
                log(f"{size_mb:<12} MB {time_ms:<12.3f} {bw:<16.1f}")

    sync()


def bench_simulated_training(steps=5, warmup=2):
    """Simulate a training step with interleaved compute and communication."""
    if not dist.is_initialized():
        log("\n[SKIP] Training simulation requires distributed mode")
        return

    world_size = get_world_size()
    rank = get_rank()

    log("\n" + "=" * 60)
    log(f"  Simulated Training Step (world_size={world_size})")
    log("=" * 60)

    device = torch.device("cuda")

    # Simulate a transformer layer
    hidden_size = 4096
    ffn_size = 11008
    seq_len = 2048
    batch_size = 2

    # Allocations
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
    w_qkv = torch.randn(hidden_size, 3 * hidden_size, dtype=torch.float16, device=device)
    w_out = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device=device)
    w_up = torch.randn(hidden_size, ffn_size, dtype=torch.float16, device=device)
    w_down = torch.randn(ffn_size, hidden_size, dtype=torch.float16, device=device)

    grad_buffer = torch.randn(hidden_size * hidden_size, dtype=torch.float16, device=device)

    log(f"\nLayer config: hidden={hidden_size}, ffn={ffn_size}, seq={seq_len}, batch={batch_size}")
    log(f"Running {steps} steps...")

    # Warmup
    for _ in range(warmup):
        # Forward
        qkv = torch.matmul(x.view(-1, hidden_size), w_qkv)
        out = torch.matmul(qkv[:, :hidden_size], w_out)
        dist.all_reduce(out)

        up = torch.matmul(x.view(-1, hidden_size), w_up)
        down = torch.matmul(up, w_down)
        dist.all_reduce(down)

    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    start = time.perf_counter()

    for step in range(steps):
        # Forward pass
        qkv = torch.matmul(x.view(-1, hidden_size), w_qkv)
        out = torch.matmul(qkv[:, :hidden_size], w_out)
        dist.all_reduce(out)

        up = torch.matmul(x.view(-1, hidden_size), w_up)
        down = torch.matmul(up, w_down)
        dist.all_reduce(down)

        # Backward (simplified - just more matmuls)
        grad = torch.matmul(down.view(-1, hidden_size), w_up.t())
        dist.all_reduce(grad)

        # Gradient sync (simulated)
        dist.all_reduce(grad_buffer)

    torch.cuda.synchronize()
    dist.barrier()

    elapsed = time.perf_counter() - start
    time_per_step = elapsed / steps * 1000

    log(f"\nResults:")
    log(f"  Time per step: {time_per_step:.2f} ms")
    log(f"  Throughput: {1000 / time_per_step:.1f} steps/s")


def main():
    parser = argparse.ArgumentParser(description="SPECTER Multi-GPU Benchmark")
    parser.add_argument("--steps", type=int, default=10, help="Iterations per test")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--gemm-only", action="store_true", help="Only GEMM tests")
    parser.add_argument("--nccl-only", action="store_true", help="Only NCCL tests")
    args = parser.parse_args()

    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        log(f"\n[SPECTER] Distributed mode: rank {rank}, world_size {dist.get_world_size()}")
    else:
        torch.cuda.set_device(0)
        log("\n[SPECTER] Single GPU mode")

    # Show GPU info
    if get_rank() == 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"[SPECTER] GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    # Run benchmarks
    if not args.nccl_only:
        bench_gemm(args.steps, args.warmup, args.quick)

    if not args.gemm_only:
        bench_nccl(args.steps, args.warmup, args.quick)

    if not args.gemm_only and not args.nccl_only and not args.quick:
        bench_simulated_training(steps=5, warmup=2)

    log("\n" + "=" * 60)
    log("  Benchmark Complete!")
    log("=" * 60 + "\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
