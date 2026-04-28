"""End-to-end benchmark: cuBLAS vs cutile flat vs cutile unified search."""

from __future__ import annotations

import json
import os
import time
import sys

import cupy as cp
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling.benefit_model import HardwareParams
from wave_tiling.unified_score import UnifiedWeights
from wave_tiling.unified_search import unified_search
from cutile_gemm.baseline_gemm import launch_baseline
from cutile_gemm.wave_tiling_gemm import launch_wave_tiling


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

FLAT_TILE_OPTIONS = [
    (16, 32),
    (16, 64),
    (16, 128),
    (16, 256),
    (32, 64),
    (32, 128),
    (32, 256),
    (64, 64),
    (64, 128),
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


def run_e2e_benchmark(hw: HardwareParams, weights: UnifiedWeights, num_sms: int, smem_bytes: int, l2_bytes: int):
    results = []

    compiled_mm = _get_compiled_mm()

    print(
        f"\n{'Shape':30} | {'cuBLAS':>9} | {'flat-best':>10} | {'unified':>9} | "
        f"{'vs-cuBLAS':>10} | {'vs-flat':>9}"
    )
    print("-" * 98)

    for M, N, K, label in LLM_DECODE_SHAPES:
        A_cp = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B_cp = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
        A_t = torch.as_tensor(A_cp, device="cuda")
        B_t = torch.as_tensor(B_cp, device="cuda")

        for _ in range(30):
            compiled_mm(A_t, B_t)
        torch.cuda.synchronize()
        ms_cublas = timed_launch_torch(lambda: compiled_mm(A_t, B_t), warmup=20, iters=200)

        ms_flat_best = float("inf")
        best_flat_cfg = None
        for sk in (1, 2, 4, 8, 16, 32, 64):
            for tm, tn in FLAT_TILE_OPTIONS:
                if sk * 32 > K:
                    continue
                ms = timed_launch_cupy(launch_baseline, A_cp, B_cp, tm, tn, 32, sk, warmup=3, iters=40)
                if ms < ms_flat_best:
                    ms_flat_best = ms
                    best_flat_cfg = (tm, tn, 32, sk)

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

        ms_unified_best = float("inf")
        best_unified_cfg = None
        for cfg in candidates:
            ms = timed_launch_cupy(
                launch_wave_tiling,
                A_cp,
                B_cp,
                cfg.tile_m,
                cfg.tile_n,
                cfg.tile_k,
                cfg.splitk,
                cfg.ws_k,
                cfg.ws_m,
                cfg.ws_n,
                warmup=3,
                iters=40,
            )
            if ms < ms_unified_best:
                ms_unified_best = ms
                best_unified_cfg = cfg

        if best_unified_cfg is None:
            continue

        speedup_vs_cublas = ms_cublas / ms_unified_best
        speedup_vs_flat = ms_flat_best / ms_unified_best

        A_bytes = M * K * 2
        B_bytes = K * N * 2
        fits_l2 = (A_bytes + B_bytes) < l2_bytes

        row = {
            "label": label,
            "M": M,
            "N": N,
            "K": K,
            "ms_cublas": ms_cublas,
            "ms_flat": ms_flat_best,
            "ms_unified": ms_unified_best,
            "speedup_vs_cublas": speedup_vs_cublas,
            "speedup_vs_flat": speedup_vs_flat,
            "best_unified_cfg": {
                "tile_m": best_unified_cfg.tile_m,
                "tile_n": best_unified_cfg.tile_n,
                "tile_k": best_unified_cfg.tile_k,
                "splitk": best_unified_cfg.splitk,
                "ws_k": best_unified_cfg.ws_k,
                "ws_m": best_unified_cfg.ws_m,
                "ws_n": best_unified_cfg.ws_n,
                "rho_tile": best_unified_cfg.rho_tile,
                "rho_wave": best_unified_cfg.rho_wave,
                "hw_waves": best_unified_cfg.hw_waves,
                "score": best_unified_cfg.score,
            },
            "best_flat_cfg": best_flat_cfg,
            "fits_l2": fits_l2,
            "total_bytes_MB": (A_bytes + B_bytes) / 1e6,
        }
        results.append(row)

        print(
            f"{label:30} | {ms_cublas:>9.4f} | {ms_flat_best:>10.4f} | {ms_unified_best:>9.4f} | "
            f"{speedup_vs_cublas:>10.3f}x | {speedup_vs_flat:>9.3f}x"
        )

    print("\n=== Summary ===")
    speedups_cb = [r["speedup_vs_cublas"] for r in results]
    speedups_fl = [r["speedup_vs_flat"] for r in results]
    if speedups_cb:
        print(
            f"vs cuBLAS: mean={np.mean(speedups_cb):.3f}x max={np.max(speedups_cb):.3f}x "
            f"min={np.min(speedups_cb):.3f}x % > 1.0x: {sum(s > 1.0 for s in speedups_cb) / len(speedups_cb):.0%}"
        )
    if speedups_fl:
        print(
            f"vs flat:   mean={np.mean(speedups_fl):.3f}x max={np.max(speedups_fl):.3f}x "
            f"% > 1.05x: {sum(s > 1.05 for s in speedups_fl) / len(speedups_fl):.0%}"
        )

    out_path = "results/e2e_benchmark.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")
    return results


def validate_score_prediction(results) -> float:
    scores = [r["best_unified_cfg"]["score"] for r in results if r["best_unified_cfg"]["score"] is not None]
    latencies = [r["ms_unified"] for r in results if r["best_unified_cfg"]["score"] is not None]
    if len(scores) < 3:
        print("Not enough data for correlation analysis")
        return 0.0
    r_val = float(np.corrcoef(np.asarray(scores), np.asarray(latencies))[0, 1])
    print("\nScore Prediction Validation:")
    print(f"  r(unified_score, latency) = {r_val:.3f}")
    print("  Expected: r < -0.3 (higher score -> lower latency)")
    print("  " + ("PASS" if r_val < -0.3 else "NEEDS IMPROVEMENT"))
    return r_val


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

    results = run_e2e_benchmark(
        hw,
        uw,
        num_sms=int(hw_profile["num_sms"]),
        smem_bytes=int(hw_profile["smem_per_sm_bytes"]),
        l2_bytes=int(hw_profile["l2_cache_bytes"]),
    )
    score_r = validate_score_prediction(results)
    json.dump({"score_latency_r": score_r}, open("results/e2e_score_validation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
