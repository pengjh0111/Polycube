"""Attribution benchmark for tile/splitk selection vs wave ordering contribution."""

from __future__ import annotations

import json
import os
import sys
import time

import cupy as cp
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cutile_gemm.wave_tiling_gemm import launch_wave_tiling
from wave_tiling.benefit_model import HardwareParams
from wave_tiling.unified_score import UnifiedWeights
from wave_tiling.unified_search import unified_search

LLM_DECODE_SHAPES = [
    (1, 4096, 4096, "LLaMA-2 Attn, bs=1"),
    (4, 4096, 4096, "LLaMA-2 Attn, bs=4"),
    (8, 4096, 4096, "LLaMA-2 Attn, bs=8"),
    (16, 4096, 4096, "LLaMA-2 Attn, bs=16"),
    (1, 4096, 11008, "LLaMA-2 FFN, bs=1"),
    (4, 4096, 11008, "LLaMA-2 FFN, bs=4"),
    (1, 14336, 4096, "LLaMA-3 FFN, bs=1"),
    (4, 14336, 4096, "LLaMA-3 FFN, bs=4"),
    (1, 8192, 8192, "Mistral-7B, bs=1"),
    (8, 8192, 8192, "Mistral-7B, bs=8"),
]


def timed_launch_torch(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def timed_launch_cupy(fn, *args, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(*args)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _get_compiled_mm():
    try:
        return torch.compile(torch.mm)
    except Exception:
        return torch.mm


def run_attribution_benchmark(hw, weights, num_sms, smem_bytes, l2_bytes):
    results = []

    compiled_mm = _get_compiled_mm()

    print(
        f"\n{'Shape':30} | {'cuBLAS':>9} | {'flat-same':>9} | {'unified':>9} | "
        f"{'tile+sk':>8} | {'wave-only':>9} | {'total':>8}"
    )
    print("-" * 108)

    for M, N, K, label in LLM_DECODE_SHAPES:
        A_cp = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B_cp = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
        A_t = torch.as_tensor(A_cp, device="cuda")
        B_t = torch.as_tensor(B_cp, device="cuda")

        for _ in range(30):
            compiled_mm(A_t, B_t)
        torch.cuda.synchronize()
        ms_A = timed_launch_torch(lambda: compiled_mm(A_t, B_t), warmup=20, iters=200)

        candidates = unified_search(
            M,
            N,
            K,
            num_sms=num_sms,
            smem_bytes=smem_bytes,
            hw=hw,
            weights=weights,
            top_k_final=10,
        )
        if not candidates:
            continue
        best = candidates[0]

        ms_B = timed_launch_cupy(
            launch_wave_tiling,
            A_cp,
            B_cp,
            best.tile_m,
            best.tile_n,
            best.tile_k,
            best.splitk,
            1,
            1,
            1,
            warmup=3,
            iters=50,
        )

        ms_C = timed_launch_cupy(
            launch_wave_tiling,
            A_cp,
            B_cp,
            best.tile_m,
            best.tile_n,
            best.tile_k,
            best.splitk,
            best.ws_k,
            best.ws_m,
            best.ws_n,
            warmup=3,
            iters=50,
        )

        A_bytes = M * K * 2
        B_bytes = K * N * 2

        row = {
            "label": label,
            "M": M,
            "N": N,
            "K": K,
            "ms_cublas": ms_A,
            "ms_flat_same_tile": ms_B,
            "ms_unified": ms_C,
            "tile_splitk_speedup": ms_A / ms_B,
            "wave_ordering_speedup": ms_B / ms_C,
            "total_speedup": ms_A / ms_C,
            "best_tile": [best.tile_m, best.tile_n, best.tile_k],
            "best_splitk": best.splitk,
            "best_wave": [best.ws_k, best.ws_m, best.ws_n],
            "hw_waves": best.hw_waves,
            "fits_l2": (A_bytes + B_bytes) < l2_bytes,
            "total_bytes_MB": (A_bytes + B_bytes) / 1e6,
        }
        results.append(row)

        print(
            f"{label:30} | {ms_A:>9.4f} | {ms_B:>9.4f} | {ms_C:>9.4f} | "
            f"{ms_A / ms_B:>8.3f}x | {ms_B / ms_C:>9.3f}x | {ms_A / ms_C:>8.3f}x"
        )

    tile_speedups = [r["tile_splitk_speedup"] for r in results]
    wave_speedups = [r["wave_ordering_speedup"] for r in results]
    total_speedups = [r["total_speedup"] for r in results]

    if results:
        print("\n=== Attribution Summary ===")
        print(
            f"Tile+SplitK selection (vs cuBLAS):  mean={np.mean(tile_speedups):.3f}x "
            f"max={np.max(tile_speedups):.3f}x"
        )
        print(
            f"Wave ordering (on top of tile):     mean={np.mean(wave_speedups):.3f}x "
            f"max={np.max(wave_speedups):.3f}x"
        )
        print(
            f"Total (vs cuBLAS):                  mean={np.mean(total_speedups):.3f}x "
            f"max={np.max(total_speedups):.3f}x"
        )

    out_path = "results/attribution_results.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")
    return results


def main() -> None:
    hw_raw = json.load(open("calibrated_hw_params.json"))
    hw_profile = json.load(open("results/hw_profile.json"))
    w_raw = json.load(open("unified_weights.json"))

    hw = HardwareParams(
        alpha=float(hw_raw.get("alpha", 1.0)),
        beta=float(hw_raw.get("beta", 2.0)),
        gamma=float(hw_raw.get("gamma", 2.0)),
        delta=float(hw_raw.get("delta", 0.0)),
        sm_count=int(hw_profile["num_sms"]),
        l2_cache_bytes=int(hw_profile["l2_cache_bytes"]),
        shared_mem_bytes=int(hw_profile["smem_per_sm_bytes"]),
    )
    uw = UnifiedWeights(
        lambda1=float(w_raw["lambda1"]),
        lambda2=float(w_raw["lambda2"]),
        lambda3=float(w_raw["lambda3"]),
        lambda4=float(w_raw["lambda4"]),
        c_atomic=float(w_raw.get("c_atomic", 0.05)),
    )

    run_attribution_benchmark(
        hw,
        uw,
        num_sms=int(hw_profile["num_sms"]),
        smem_bytes=int(hw_profile["smem_per_sm_bytes"]),
        l2_bytes=int(hw_profile["l2_cache_bytes"]),
    )


if __name__ == "__main__":
    main()
