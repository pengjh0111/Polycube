"""E2E benchmark with optimized baseline and isolated wave-ordering attribution."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

import cupy as cp
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cutile_gemm.optimized_baseline import launch_optimized
from wave_tiling.benefit_model import HardwareParams
from wave_tiling.unified_score import UnifiedWeights
from wave_tiling.unified_search import unified_search

FIXED_NAIVE_CONFIG = {
    "tile_m": 32,
    "tile_n": 128,
    "tile_k": 32,
    "splitk": 1,
}

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


def timed_torch(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def timed_cupy(fn, *args, warmup=5, iters=40):
    for _ in range(warmup):
        fn(*args)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def run_e2e_v2(hw, weights, num_sms, smem_bytes, l2_bytes):
    del l2_bytes
    results = []

    print(
        f"\n{'Shape':28} | {'cuBLAS':>7} | {'C-naive':>8} | {'B-opt':>7} | {'D-wave':>7} | "
        f"{'B/A':>6} | {'D/B':>6} | {'D/A':>6}"
    )
    print("-" * 104)

    for M, N, K, label in LLM_DECODE_SHAPES:
        A_cp = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B_cp = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
        A_t = torch.as_tensor(A_cp, device="cuda")
        B_t = torch.as_tensor(B_cp, device="cuda")

        for _ in range(30):
            torch.mm(A_t, B_t)
        torch.cuda.synchronize()
        ms_A = timed_torch(lambda: torch.mm(A_t, B_t), warmup=20, iters=200)

        try:
            ms_C = timed_cupy(
                launch_optimized,
                A_cp,
                B_cp,
                FIXED_NAIVE_CONFIG["tile_m"],
                FIXED_NAIVE_CONFIG["tile_n"],
                FIXED_NAIVE_CONFIG["tile_k"],
                FIXED_NAIVE_CONFIG["splitk"],
                1,
                1,
                1,
                8,
                warmup=3,
                iters=40,
            )
        except Exception as exc:
            print(f"  C-naive failed on {label}: {exc}")
            ms_C = float("nan")

        candidates = unified_search(M, N, K, num_sms, smem_bytes, hw, weights, top_k_final=10)
        if not candidates:
            continue

        ms_B = float("inf")
        best_tile_cfg = None
        for cfg in candidates:
            try:
                ms = timed_cupy(
                    launch_optimized,
                    A_cp,
                    B_cp,
                    cfg.tile_m,
                    cfg.tile_n,
                    cfg.tile_k,
                    cfg.splitk,
                    1,
                    1,
                    1,
                    8,
                    warmup=3,
                    iters=40,
                )
                if ms < ms_B:
                    ms_B = ms
                    best_tile_cfg = cfg
            except Exception:
                continue

        ms_D = float("inf")
        best_full_cfg = None
        for cfg in candidates:
            try:
                ms = timed_cupy(
                    launch_optimized,
                    A_cp,
                    B_cp,
                    cfg.tile_m,
                    cfg.tile_n,
                    cfg.tile_k,
                    cfg.splitk,
                    cfg.ws_k,
                    cfg.ws_m,
                    cfg.ws_n,
                    8,
                    warmup=3,
                    iters=40,
                )
                if ms < ms_D:
                    ms_D = ms
                    best_full_cfg = cfg
            except Exception:
                continue

        if best_tile_cfg is None or best_full_cfg is None or not np.isfinite(ms_B) or not np.isfinite(ms_D):
            continue

        ratio_BA = ms_A / ms_B
        ratio_DB = ms_B / ms_D
        ratio_DA = ms_A / ms_D

        print(
            f"{label:28} | {ms_A:>7.3f} | {ms_C:>8.3f} | {ms_B:>7.3f} | {ms_D:>7.3f} | "
            f"{ratio_BA:>6.3f}x | {ratio_DB:>6.3f}x | {ratio_DA:>6.3f}x"
        )

        results.append(
            {
                "label": label,
                "M": M,
                "N": N,
                "K": K,
                "ms_cublas": ms_A,
                "ms_naive": ms_C,
                "ms_opt_no_wave": ms_B,
                "ms_opt_wave": ms_D,
                "speedup_tile_search_vs_cublas": ratio_BA,
                "speedup_wave_ordering": ratio_DB,
                "speedup_total_vs_cublas": ratio_DA,
                "best_tile_cfg": asdict(best_tile_cfg),
                "best_full_cfg": asdict(best_full_cfg),
            }
        )

    ba = [r["speedup_tile_search_vs_cublas"] for r in results]
    db = [r["speedup_wave_ordering"] for r in results]
    da = [r["speedup_total_vs_cublas"] for r in results]

    print("\n=== Attribution Summary ===")
    if results:
        print(f"Tile+SplitK search (B/A, vs cuBLAS): mean={np.mean(ba):.3f}x max={np.max(ba):.3f}x")
        print(f"Wave ordering (D/B, on top of opt):  mean={np.mean(db):.3f}x max={np.max(db):.3f}x")
        print(f"Total framework (D/A, vs cuBLAS):    mean={np.mean(da):.3f}x max={np.max(da):.3f}x")

    json.dump(results, open("results/e2e_v2.json", "w"), indent=2)
    print("Saved: results/e2e_v2.json")
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

    run_e2e_v2(
        hw,
        uw,
        num_sms=int(hw_profile["num_sms"]),
        smem_bytes=int(hw_profile["smem_per_sm_bytes"]),
        l2_bytes=int(hw_profile["l2_cache_bytes"]),
    )


if __name__ == "__main__":
    main()
