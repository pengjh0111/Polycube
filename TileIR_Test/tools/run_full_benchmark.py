"""Five-variant full attribution benchmark for LLM GEMM shapes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cutile_gemm.polycube_runtime_gemm import (
    build_optimized_polycube_plan,
    build_optimized_rowmajor_plan,
    build_rowmajor_compiled_plan,
    launch_optimized_compiled,
    launch_polycube_compiled,
)
from wave_tiling.benefit_model import HardwareParams
from wave_tiling.search import search_optimal_wave_shape
from wave_tiling.wave_shape import PolycubeWaveShape

BENCHMARK_SHAPES = [
    (1, 4096, 4096, "llama2-7b-attn-bs1"),
    (32, 4096, 4096, "llama2-7b-attn-bs32"),
    (512, 4096, 4096, "llama2-7b-attn-bs512"),
    (2048, 4096, 4096, "llama2-7b-attn-bs2k"),
    (1, 4096, 11008, "llama2-7b-ffn-bs1"),
    (32, 4096, 11008, "llama2-7b-ffn-bs32"),
    (512, 4096, 11008, "llama2-7b-ffn-bs512"),
    (1, 8192, 8192, "llama2-70b-attn-bs1"),
    (32, 8192, 8192, "llama2-70b-attn-bs32"),
    (512, 8192, 8192, "llama2-70b-attn-bs512"),
    (1, 8192, 28672, "llama2-70b-ffn-bs1"),
    (32, 8192, 28672, "llama2-70b-ffn-bs32"),
    (1, 4096, 14336, "mistral-7b-ffn-bs1"),
    (32, 4096, 14336, "mistral-7b-ffn-bs32"),
    (1024, 1024, 1024, "sq-1k"),
    (2048, 2048, 2048, "sq-2k"),
    (4096, 4096, 4096, "sq-4k"),
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


def timed_median(fn, *args, warmup=10, iters=200, repeats=3):
    for _ in range(warmup):
        fn(*args)
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        cp.cuda.Device().synchronize()
        times.append((time.perf_counter() - t0) / iters * 1000.0)
    return float(np.median(times))


def select_heuristic_tile(M, N, K, smem_bytes=232448):
    """
    Tile heuristic with hard constraint: tile_m <= M, tile_n <= N.
    Larger tiles for large compute-bound shapes,
    smaller for memory-bound small-M shapes.
    """

    del K
    if M <= 16:
        candidates = [(16, 128, 32), (16, 256, 32)]
    elif M <= 64:
        candidates = [(32, 128, 32), (32, 256, 32)]
    elif M <= 256:
        candidates = [(64, 128, 32), (64, 256, 32)]
    else:
        candidates = [(64, 128, 32), (128, 128, 32), (64, 256, 32)]

    valid = [
        (tm, tn, tk)
        for tm, tn, tk in candidates
        if (tm * tk + tk * tn) * 2 + tm * tn * 4 <= smem_bytes and tm <= M and tn <= N
    ]

    if not valid:
        for tm in [64, 32, 16, 8, 4, 2, 1]:
            if tm <= M:
                tn = min(128, N)
                if (tm * 32 + 32 * tn) * 2 + tm * tn * 4 <= smem_bytes:
                    return tm, tn, 32
        return max(1, min(16, M)), min(128, N), 32

    return max(valid, key=lambda x: x[0] * x[1])


def select_heuristic_splitk(M, N, K, tile_m, tile_n, num_sms=170):
    """
    splitk heuristic: maximize SM utilization.
    """

    tm = math.ceil(M / tile_m)
    tn = math.ceil(N / tile_n)
    base = tm * tn
    best_s, best_util = 1, base / (math.ceil(base / num_sms) * num_sms)
    for s in [1, 2, 4, 8, 16, 32, 64]:
        if s * 32 > K:
            break
        total = base * s
        hw_w = math.ceil(total / num_sms)
        util = total / (hw_w * num_sms)
        if util > best_util:
            best_util, best_s = util, s
    return best_s


def gpu_name():
    return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


def tflops(M, N, K, ms):
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def run_shape(M, N, K, label, hw, num_sms, max_shapes, top_eval):
    result = {
        "label": label,
        "M": M,
        "N": N,
        "K": K,
        "ms": {},
        "tflops": {},
        "speedup": {},
        "configs": {},
    }

    A_cp = cp.random.standard_normal((M, K)).astype(cp.float16)
    B_cp = cp.random.standard_normal((K, N)).astype(cp.float16)
    A_t = torch.as_tensor(A_cp, device="cuda")
    B_t = torch.as_tensor(B_cp, device="cuda")

    ms_a = timed_torch(lambda: torch.mm(A_t, B_t), warmup=20, iters=200)
    result["ms"]["cublas"] = ms_a
    result["tflops"]["cublas"] = tflops(M, N, K, ms_a)

    try:
        b_plan = build_rowmajor_compiled_plan(M, N, K, 16, 128, 32, 1)
        ms_b = timed_median(launch_polycube_compiled, A_cp, B_cp, b_plan, warmup=10, iters=200, repeats=3)
    except Exception as e:
        print(f"  [B] failed: {e}", flush=True)
        ms_b = float("nan")
    result["ms"]["naive"] = ms_b
    result["tflops"]["naive"] = tflops(M, N, K, ms_b) if not math.isnan(ms_b) else float("nan")
    result["configs"]["naive"] = {"tile_m": 16, "tile_n": 128, "tile_k": 32, "splitk": 1}

    try:
        tm_c, tn_c, tk_c = select_heuristic_tile(M, N, K)
        sk_c = select_heuristic_splitk(M, N, K, tm_c, tn_c, num_sms)
        c_plan = build_optimized_rowmajor_plan(M, N, K, tm_c, tn_c, tk_c, sk_c)
        ms_c = timed_median(launch_optimized_compiled, A_cp, B_cp, c_plan, warmup=10, iters=200, repeats=3)
    except Exception as e:
        print(f"  [C] failed: {e}", flush=True)
        ms_c = float("nan")
        tm_c = tn_c = tk_c = sk_c = None
    result["ms"]["opt_heur"] = ms_c
    result["tflops"]["opt_heur"] = tflops(M, N, K, ms_c) if not math.isnan(ms_c) else float("nan")
    result["configs"]["opt_heur"] = {
        "tile_m": tm_c,
        "tile_n": tn_c,
        "tile_k": tk_c,
        "splitk": sk_c,
    }

    frontier = []
    try:
        tile_m_opts = [t for t in [16, 32, 64, 128] if t <= M or t == 16]
        tile_n_opts = [t for t in [64, 128, 256] if t <= N or t == 64]
        frontier = search_optimal_wave_shape(
            M=M,
            N=N,
            K=K,
            tile_m_candidates=tile_m_opts,
            tile_n_candidates=tile_n_opts,
            splitk_candidates=[1, 2, 4, 8, 16, 32, 64],
            hw=hw,
            shape_type="polycube",
            max_shapes=max_shapes,
        )
    except Exception as e:
        print(f"  [search] failed: {e}", flush=True)

    polycube_candidates = [cfg for cfg in frontier[:top_eval] if isinstance(cfg.wave_shape, PolycubeWaveShape)]

    ms_d = float("inf")
    best_d_cfg = None
    for cfg in polycube_candidates:
        try:
            tk = getattr(cfg, "tile_k", 32)
            d_plan = build_optimized_polycube_plan(
                M,
                N,
                K,
                cfg.tile_m,
                cfg.tile_n,
                tk,
                cfg.splitk_factor,
                cfg.wave_shape,
            )
            ms = timed_median(launch_optimized_compiled, A_cp, B_cp, d_plan, warmup=10, iters=200, repeats=3)
            if ms < ms_d:
                ms_d = ms
                best_d_cfg = cfg
        except Exception as ex:
            print(f"  [D] candidate failed: {ex}", flush=True)
            continue
    if not math.isfinite(ms_d):
        ms_d = float("nan")

    ms_e = float("inf")
    best_e_cfg = best_d_cfg
    if best_d_cfg is not None:
        try:
            tk = getattr(best_d_cfg, "tile_k", 32)
            e_plan = build_optimized_rowmajor_plan(
                M,
                N,
                K,
                best_d_cfg.tile_m,
                best_d_cfg.tile_n,
                tk,
                best_d_cfg.splitk_factor,
            )
            ms_e = timed_median(launch_optimized_compiled, A_cp, B_cp, e_plan, warmup=10, iters=200, repeats=3)
        except Exception as ex:
            print(f"  [E] paired eval failed: {ex}", flush=True)
            ms_e = float("inf")
            best_e_cfg = None
            for cfg in polycube_candidates:
                try:
                    tk = getattr(cfg, "tile_k", 32)
                    e_plan = build_optimized_rowmajor_plan(M, N, K, cfg.tile_m, cfg.tile_n, tk, cfg.splitk_factor)
                    ms = timed_median(launch_optimized_compiled, A_cp, B_cp, e_plan, warmup=10, iters=200, repeats=3)
                    if ms < ms_e:
                        ms_e = ms
                        best_e_cfg = cfg
                except Exception as inner_ex:
                    print(f"  [E] candidate failed: {inner_ex}", flush=True)
                    continue
    else:
        for cfg in polycube_candidates:
            try:
                tk = getattr(cfg, "tile_k", 32)
                e_plan = build_optimized_rowmajor_plan(M, N, K, cfg.tile_m, cfg.tile_n, tk, cfg.splitk_factor)
                ms = timed_median(launch_optimized_compiled, A_cp, B_cp, e_plan, warmup=10, iters=200, repeats=3)
                if ms < ms_e:
                    ms_e = ms
                    best_e_cfg = cfg
            except Exception as ex:
                print(f"  [E] candidate failed: {ex}", flush=True)
                continue
    if not math.isfinite(ms_e):
        ms_e = float("nan")
    result["ms"]["opt_frametile"] = ms_e
    result["tflops"]["opt_frametile"] = tflops(M, N, K, ms_e) if not math.isnan(ms_e) else float("nan")
    result["configs"]["opt_frametile"] = {
        "tile_m": best_e_cfg.tile_m if best_e_cfg else None,
        "tile_n": best_e_cfg.tile_n if best_e_cfg else None,
        "tile_k": getattr(best_e_cfg, "tile_k", 32) if best_e_cfg else None,
        "splitk": best_e_cfg.splitk_factor if best_e_cfg else None,
    }

    result["ms"]["opt_wave"] = ms_d
    result["tflops"]["opt_wave"] = tflops(M, N, K, ms_d) if not math.isnan(ms_d) else float("nan")
    result["configs"]["opt_wave"] = {
        "tile_m": best_d_cfg.tile_m if best_d_cfg else None,
        "tile_n": best_d_cfg.tile_n if best_d_cfg else None,
        "tile_k": getattr(best_d_cfg, "tile_k", 32) if best_d_cfg else None,
        "splitk": best_d_cfg.splitk_factor if best_d_cfg else None,
        "wave_shape": best_d_cfg.to_dict()["wave_shape"] if best_d_cfg else None,
    }

    result["speedup"] = {
        "C_over_A": ms_a / ms_c if math.isfinite(ms_c) else float("nan"),
        "E_over_C": ms_c / ms_e if (math.isfinite(ms_c) and math.isfinite(ms_e)) else float("nan"),
        "D_over_E": ms_e / ms_d if (math.isfinite(ms_e) and math.isfinite(ms_d)) else float("nan"),
        "D_over_A": ms_a / ms_d if (math.isfinite(ms_a) and math.isfinite(ms_d)) else float("nan"),
        "C_over_A_ratio": ms_a / ms_c if math.isfinite(ms_c) else float("nan"),
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shapes", type=int, default=50)
    parser.add_argument("--top-eval", type=int, default=8)
    parser.add_argument("--num-sms", type=int, default=0, help="Override SM count (0=auto-detect)")
    args = parser.parse_args()

    hw_profile_path = Path("results/hw_profile.json")
    hw_raw_path = Path("calibrated_hw_params.json")

    num_sms = args.num_sms
    if num_sms == 0:
        if hw_profile_path.exists():
            num_sms = json.loads(hw_profile_path.read_text())["num_sms"]
        else:
            num_sms = 170

    hw = HardwareParams(sm_count=num_sms)
    if hw_raw_path.exists():
        raw = json.loads(hw_raw_path.read_text())
        hw = HardwareParams(
            alpha=float(raw.get("alpha", 1.0)),
            beta=float(raw.get("beta", 2.0)),
            gamma=float(raw.get("gamma", 2.0)),
            delta=float(raw.get("delta", 0.0)),
            sm_count=num_sms,
        )

    gname = gpu_name()
    print(f"GPU: {gname}  SMs: {num_sms}", flush=True)
    print(
        f"\n{'Shape':<26} | {'A(cuBLAS)':>10} | {'B(Naive)':>9} | {'C(OptHeur)':>10} | {'E(OptFT)':>9} | {'D(OptWave)':>10} | {'C/A':>6} | {'E/C':>6} | {'D/E':>6} | {'D/A':>6}",
        flush=True,
    )
    print("-" * 115, flush=True)

    all_results = []

    for M, N, K, label in BENCHMARK_SHAPES:
        print(f"[{label}] M={M} N={N} K={K}", flush=True)
        try:
            r = run_shape(M, N, K, label, hw, num_sms, args.max_shapes, args.top_eval)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            continue
        all_results.append(r)

        ms = r["ms"]
        sp = r["speedup"]

        def fmt_ms(v):
            return f"{v:9.3f}" if math.isfinite(v) else "    nan  "

        def fmt_sp(v):
            return f"{v:6.3f}x" if math.isfinite(v) else "   nan  "

        print(
            f"{label:<26} | {fmt_ms(ms['cublas'])} | {fmt_ms(ms['naive'])} | {fmt_ms(ms['opt_heur'])} | {fmt_ms(ms['opt_frametile'])} | {fmt_ms(ms['opt_wave'])} | {fmt_sp(sp['C_over_A_ratio'])} | {fmt_sp(sp['E_over_C'])} | {fmt_sp(sp['D_over_E'])} | {fmt_sp(sp['D_over_A'])}",
            flush=True,
        )

    print("-" * 115, flush=True)

    def geomean(key):
        vals = [
            r["speedup"][key]
            for r in all_results
            if math.isfinite(r["speedup"].get(key, float("nan")))
        ]
        if not vals:
            return float("nan")
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    gm_ca = geomean("C_over_A_ratio")
    gm_ec = geomean("E_over_C")
    gm_de = geomean("D_over_E")
    gm_da = geomean("D_over_A")

    def fmt_sp(v):
        return f"{v:6.3f}x" if math.isfinite(v) else "   nan  "

    print(f"\nAttribution summary (geomean over {len(all_results)} shapes):", flush=True)
    print(f"  C/A  base optimizations (MMA+num_ctas+latency+swizzle) vs cuBLAS : {fmt_sp(gm_ca)}", flush=True)
    print(f"  E/C  framework tile/splitk selection                              : {fmt_sp(gm_ec)}", flush=True)
    print(f"  D/E  framework polycube block order                               : {fmt_sp(gm_de)}", flush=True)
    print(f"  D/A  total (framework + optimization) vs cuBLAS                  : {fmt_sp(gm_da)}", flush=True)

    Path("results").mkdir(exist_ok=True)
    payload = {
        "gpu": gname,
        "sm_count": num_sms,
        "shapes": all_results,
        "geomean": {
            "C_over_A": gm_ca,
            "E_over_C": gm_ec,
            "D_over_E": gm_de,
            "D_over_A": gm_da,
        },
    }
    Path("results/full_benchmark.json").write_text(json.dumps(payload, indent=2))
    print("\nSaved: results/full_benchmark.json", flush=True)


if __name__ == "__main__":
    main()
