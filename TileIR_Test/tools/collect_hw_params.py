"""Collect hardware profile for unified search framework."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp


def _measure_peak_tflops_fp16() -> float:
    import torch

    N = 8192
    A = torch.randn(N, N, dtype=torch.float16, device="cuda")
    B = torch.randn(N, N, dtype=torch.float16, device="cuda")

    for _ in range(5):
        torch.mm(A, B)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    flops = 2 * N * N * N
    return float(flops / elapsed / 1e12)


def _measure_mem_bandwidth_gbs() -> float:
    V = 2**28
    try:
        src = cp.ones(V, dtype=cp.float16)
        dst = cp.empty(V, dtype=cp.float16)
    except Exception:
        # Fallback for smaller memory GPUs.
        V = 2**26
        src = cp.ones(V, dtype=cp.float16)
        dst = cp.empty(V, dtype=cp.float16)

    for _ in range(3):
        cp.copyto(dst, src)
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        cp.copyto(dst, src)
    cp.cuda.Device().synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    # read + write traffic
    bytes_moved = V * 2 * 2
    return float(bytes_moved / elapsed / 1e9)


def collect_hw_profile() -> dict:
    dev = cp.cuda.Device(0)
    attrs = dev.attributes

    profile = {
        "num_sms": int(attrs["MultiProcessorCount"]),
        "l2_cache_bytes": int(attrs["L2CacheSize"]),
        "smem_per_sm_bytes": int(attrs["MaxSharedMemoryPerBlock"]),
        "peak_tflops_fp16": None,
        "mem_bandwidth_gbs": None,
        "ridge_point": None,
        "tc_alignment": 16,
        "atomic_latency_cycles": None,
        "dtype_bytes": 2,
    }

    profile["peak_tflops_fp16"] = _measure_peak_tflops_fp16()
    profile["mem_bandwidth_gbs"] = _measure_mem_bandwidth_gbs()
    profile["ridge_point"] = (
        profile["peak_tflops_fp16"] * 1e12 / (profile["mem_bandwidth_gbs"] * 1e9)
    )

    out_path = Path("results/hw_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))
    print(f"Saved: {out_path}")
    return profile


if __name__ == "__main__":
    collect_hw_profile()
