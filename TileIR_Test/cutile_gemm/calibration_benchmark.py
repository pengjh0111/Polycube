"""Calibration benchmark: optimized cuTile baseline vs cuBLAS."""

from __future__ import annotations

import json
import time
import os
import sys

import cupy as cp
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cutile_gemm.optimized_baseline import launch_optimized

CALIBRATION_SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (2048, 8192, 8192),
    (4096, 8192, 4096),
]

PROBE_CONFIGS = [
    (128, 256, 64),
    (64, 128, 64),
    (128, 128, 64),
    (64, 256, 32),
    (64, 128, 32),
]


def timed_torch(fn, warmup=20, iters=100):
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


def run_calibration():
    rows = []
    print(
        f"\n{'Shape':25} | {'cuBLAS(ms)':>11} | {'cutile(ms)':>11} | "
        f"{'ratio':>7} | {'TFLOPS_ct':>10} | {'TFLOPS_cb':>10}"
    )
    print("-" * 85)

    for M, N, K in CALIBRATION_SHAPES:
        A_cp = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B_cp = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
        A_t = torch.as_tensor(A_cp, device="cuda")
        B_t = torch.as_tensor(B_cp, device="cuda")

        for _ in range(30):
            torch.mm(A_t, B_t)
        torch.cuda.synchronize()
        ms_cublas = timed_torch(lambda: torch.mm(A_t, B_t), warmup=20, iters=80)

        best_ms = float("inf")
        best_cfg = None
        fail_msgs = []
        for tm, tn, tk in PROBE_CONFIGS:
            try:
                for _ in range(2):
                    launch_optimized(A_cp, B_cp, tm, tn, tk, splitk_factor=1, ws_k=1, ws_m=1, ws_n=1)
                ms = timed_cupy(launch_optimized, A_cp, B_cp, tm, tn, tk, 1, 1, 1, 1, 8, warmup=2, iters=20)
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = (tm, tn, tk)
            except Exception as exc:
                fail_msgs.append(f"{(tm, tn, tk)} failed: {exc}")

        if best_cfg is None:
            for msg in fail_msgs:
                print("  " + msg)
            continue

        flops = 2.0 * M * N * K
        tflops_ct = flops / best_ms / 1e9
        tflops_cb = flops / ms_cublas / 1e9
        ratio = ms_cublas / best_ms

        print(
            f"M={M} N={N} K={K}         | {ms_cublas:>11.3f} | {best_ms:>11.3f} | "
            f"{ratio:>7.3f}x | {tflops_ct:>10.1f} | {tflops_cb:>10.1f}"
        )
        print(f"  best_cfg: {best_cfg}")

        rows.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "ms_cublas": ms_cublas,
                "ms_cutile_best": best_ms,
                "ratio_cutile_over_cublas": ratio,
                "tflops_cutile": tflops_ct,
                "tflops_cublas": tflops_cb,
                "best_cfg": list(best_cfg),
            }
        )

    out = {
        "rows": rows,
        "mean_ratio": (sum(r["ratio_cutile_over_cublas"] for r in rows) / len(rows)) if rows else 0.0,
        "max_ratio": max((r["ratio_cutile_over_cublas"] for r in rows), default=0.0),
    }
    json.dump(out, open("results/calibration_results.json", "w"), indent=2)
    print("\nSaved: results/calibration_results.json")

    print("\nInterpretation:")
    print("  ratio > 0.7: baseline is mature enough for scheduling experiments")
    print("  ratio < 0.5: kernel has issues (check mma path, tile sizes)")
    print("  ratio < 0.3: fundamental problem")
    return out


if __name__ == "__main__":
    run_calibration()
