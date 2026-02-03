#!/usr/bin/env python3
"""
SPECTER Test Script

Tests both cuBLAS (GEMM) and NCCL operations for tracing.

Single GPU:   specter capture -o ./traces -- python test_trace.py
Multi GPU:    specter capture -o ./traces -- torchrun --nproc_per_node=N test_trace.py
"""

import os
import torch
import torch.distributed as dist


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def test_cublas():
    """Run some matrix multiplications to generate cuBLAS traces."""
    print(f"[Rank {get_rank()}] Running cuBLAS tests...")

    device = torch.device("cuda")

    # Different GEMM shapes to trace
    shapes = [
        (4096, 4096, 4096),    # Square
        (4096, 16384, 4096),   # Wide (like FFN)
        (4096, 4096, 16384),   # Tall
        (2048, 2048, 2048),    # Smaller
        (8192, 8192, 8192),    # Larger
    ]

    for m, n, k in shapes:
        # FP16 GEMM (most common in training)
        a = torch.randn(m, k, dtype=torch.float16, device=device)
        b = torch.randn(k, n, dtype=torch.float16, device=device)

        # Warmup
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Traced runs
        for _ in range(5):
            c = torch.matmul(a, b)

        torch.cuda.synchronize()
        print(f"  GEMM {m}x{n}x{k}: done")

    # Batched GEMM (like attention)
    batch_size = 32
    seq_len = 2048
    heads = 32
    head_dim = 128

    q = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.float16, device=device)

    # Attention scores (batched GEMM)
    for _ in range(5):
        scores = torch.matmul(q, k.transpose(-2, -1))

    torch.cuda.synchronize()
    print(f"  Batched GEMM (attention): done")


def test_nccl():
    """Run NCCL collective operations to generate traces."""
    if not dist.is_initialized():
        print("[SPECTER] Skipping NCCL tests (not distributed)")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda")

    if world_size < 2:
        print(f"[Rank {rank}] Skipping NCCL tests (world_size={world_size}, need >= 2)")
        return

    print(f"[Rank {rank}] Running NCCL tests (world_size={world_size})...")

    # Different sizes to trace
    sizes = [
        1024 * 1024,        # 1M elements = 2MB fp16
        16 * 1024 * 1024,   # 16M elements = 32MB fp16
        64 * 1024 * 1024,   # 64M elements = 128MB fp16
    ]

    for size in sizes:
        tensor = torch.randn(size, dtype=torch.float16, device=device)

        # AllReduce
        for _ in range(3):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        print(f"  AllReduce {size//1024//1024}M: done")

        # AllGather
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        for _ in range(3):
            dist.all_gather(gathered, tensor)
        torch.cuda.synchronize()
        print(f"  AllGather {size//1024//1024}M: done")

        # ReduceScatter
        input_list = [torch.randn(size // world_size, dtype=torch.float16, device=device)
                      for _ in range(world_size)]
        output = torch.zeros(size // world_size, dtype=torch.float16, device=device)
        for _ in range(3):
            dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        print(f"  ReduceScatter {size//1024//1024}M: done")

    # Point-to-point (Send/Recv)
    if world_size >= 2:
        tensor = torch.randn(1024 * 1024, dtype=torch.float16, device=device)

        for _ in range(3):
            if rank == 0:
                dist.send(tensor, dst=1)
                dist.recv(tensor, src=1)
            else:
                dist.recv(tensor, src=0)
                dist.send(tensor, dst=0)

        torch.cuda.synchronize()
        print(f"  Send/Recv: done")


def main():
    # Check if we're running distributed
    distributed = "RANK" in os.environ or "LOCAL_RANK" in os.environ

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Started on GPU {local_rank}")
    else:
        rank = 0
        torch.cuda.set_device(0)
        print(f"[Single GPU] Started on GPU 0")

    # Run tests
    test_cublas()

    if distributed:
        dist.barrier()

    test_nccl()

    if distributed:
        dist.barrier()

    if rank == 0:
        print("\n[SPECTER] All tests complete!")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
