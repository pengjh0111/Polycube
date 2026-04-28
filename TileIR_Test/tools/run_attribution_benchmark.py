"""Run A/B/D attribution benchmark following the polycube full-test pattern."""

from __future__ import annotations

import argparse
import inspect
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
    PolycubeRuntimePlan,
    build_polycube_compiled_plan,
    build_polycube_plan,
    build_rowmajor_compiled_plan,
    launch_polycube_compiled,
    launch_polycube_plan,
    polycube_runtime_kernel,
)
from wave_tiling.benefit_model import HardwareParams
from wave_tiling.search import search_optimal_wave_shape
from wave_tiling.task_space import TaskSpace3D
from wave_tiling.wave_shape import PolycubeWaveShape

BENCHMARK_SHAPES = [
    (1024, 1024, 1024, "sq-1k"),
    (2048, 2048, 2048, "sq-2k"),
    (4096, 4096, 4096, "sq-4k"),
    (128, 4096, 8192, "tall-128"),
    (256, 4096, 8192, "tall-256"),
    (4096, 512, 2048, "fat-4k"),
    (8192, 256, 2048, "fat-8k"),
    (1, 4096, 4096, "llm-decode-1"),
    (4, 4096, 4096, "llm-decode-4"),
    (8, 4096, 4096, "llm-decode-8"),
    (16, 4096, 4096, "llm-decode-16"),
    (1, 4096, 11008, "llm-ffn-1"),
    (4, 4096, 11008, "llm-ffn-4"),
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


def timed_median(fn, *args, warmup=10, iters=200, repeats=3):
    """Run timing repeats times, return median ms."""

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


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def _build_default_rowmajor_plan(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk_factor: int,
) -> PolycubeRuntimePlan:
    """Build a no-extra-order baseline plan with default row-major mapping."""

    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    ordered = [
        (sk, m, n)
        for m in range(task.tm_dim)
        for n in range(task.tn_dim)
        for sk in range(task.sk_dim)
    ]

    pid_sk = cp.asarray(np.asarray([q[0] for q in ordered], dtype=np.int32))
    pid_m = cp.asarray(np.asarray([q[1] for q in ordered], dtype=np.int32))
    pid_n = cp.asarray(np.asarray([q[2] for q in ordered], dtype=np.int32))

    return PolycubeRuntimePlan(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        splitk_factor=splitk_factor,
        M=M,
        N=N,
        K=K,
        pid_sk_lut=pid_sk,
        pid_m_lut=pid_m,
        pid_n_lut=pid_n,
    )


def _safe_name(label: str) -> str:
    out = []
    for ch in label.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", ","}:
            out.append("_")
    name = "".join(out).strip("_")
    return "_".join([tok for tok in name.split("_") if tok])


def _kernel_source_or_note(fn) -> str:
    try:
        return inspect.getsource(fn)
    except Exception as exc:
        return f"# Source introspection unavailable: {type(exc).__name__}: {exc}"


def _dump_case_kernel_compare(
    dump_dir: Path,
    label: str,
    M: int,
    N: int,
    K: int,
    best_d_cfg,
    b_plan: PolycubeRuntimePlan,
    d_plan: PolycubeRuntimePlan,
    ms_a: float,
    ms_b: float,
    ms_d: float,
) -> str:
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(label)}__greedy"
    out_path = dump_dir / f"{stem}.txt"

    tile_k = best_d_cfg.tile_k if hasattr(best_d_cfg, "tile_k") else 32
    runtime_source = _kernel_source_or_note(polycube_runtime_kernel)
    launch_source = _kernel_source_or_note(launch_polycube_plan)

    payload = [
        f"label: {label}",
        "strategy: greedy",
        f"shape: M={M}, N={N}, K={K}",
        f"timing_ms: cublas={ms_a:.6f}, B_same_cfg={ms_b:.6f}, D_poly_order={ms_d:.6f}, D_over_B={ms_b / ms_d:.6f}x",
        "",
        "[B same-cfg launch parameters | same kernel with default row-major LUT]",
        f"tile_m={best_d_cfg.tile_m}, tile_n={best_d_cfg.tile_n}, tile_k={tile_k}, splitk_factor={best_d_cfg.splitk_factor}",
        f"lut_blocks={int(b_plan.pid_sk_lut.shape[0])}",
        f"lut_preview_pid_sk={cp.asnumpy(b_plan.pid_sk_lut[:16]).tolist()}",
        f"lut_preview_pid_m={cp.asnumpy(b_plan.pid_m_lut[:16]).tolist()}",
        f"lut_preview_pid_n={cp.asnumpy(b_plan.pid_n_lut[:16]).tolist()}",
        f"lut_full_pid_sk={cp.asnumpy(b_plan.pid_sk_lut).tolist()}",
        f"lut_full_pid_m={cp.asnumpy(b_plan.pid_m_lut).tolist()}",
        f"lut_full_pid_n={cp.asnumpy(b_plan.pid_n_lut).tolist()}",
        "",
        "[D poly-order launch parameters | same kernel with optimized polycube LUT]",
        f"tile_m={d_plan.tile_m}, tile_n={d_plan.tile_n}, tile_k={d_plan.tile_k}, splitk_factor={d_plan.splitk_factor}",
        f"lut_blocks={int(d_plan.pid_sk_lut.shape[0])}",
        f"lut_preview_pid_sk={cp.asnumpy(d_plan.pid_sk_lut[:16]).tolist()}",
        f"lut_preview_pid_m={cp.asnumpy(d_plan.pid_m_lut[:16]).tolist()}",
        f"lut_preview_pid_n={cp.asnumpy(d_plan.pid_n_lut[:16]).tolist()}",
        f"lut_full_pid_sk={cp.asnumpy(d_plan.pid_sk_lut).tolist()}",
        f"lut_full_pid_m={cp.asnumpy(d_plan.pid_m_lut).tolist()}",
        f"lut_full_pid_n={cp.asnumpy(d_plan.pid_n_lut).tolist()}",
        "",
        "[Shared runtime kernel source: polycube_runtime_kernel]",
        runtime_source,
        "",
        "[Shared launcher source: launch_polycube_plan]",
        launch_source,
    ]
    out_path.write_text("\n".join(payload))
    return str(out_path)


def _format_ms(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "nan"


def _format_speedup(value: float) -> str:
    return f"{value:.3f}x" if np.isfinite(value) else "nan"


def _gpu_name() -> str:
    props = cp.cuda.runtime.getDeviceProperties(0)
    return props["name"].decode("utf-8")


def run_attribution_benchmark(
    hw: HardwareParams,
    num_workers: int,
    max_shapes: int,
    top_eval: int,
    dump_kernel_compare: bool,
    kernel_dump_root: str,
):
    tile_m_options = [16, 32, 64]
    tile_n_options = [32, 64, 128, 256]
    splitk_options = [1, 2, 4, 8, 16, 32, 64]

    rows: list[dict] = []
    print(
        f"GPU: {_gpu_name()}  SMs: {hw.sm_count}",
        flush=True,
    )
    print(
        f"\n{'Shape':28} | {'cuBLAS(ms)':>10} | {'cuBLAS(TF)':>10} | {'B(ms)':>8} | {'D(ms)':>8} | {'D/B':>6} | {'D/A':>6} | {'D(TF)':>8} | {'search(s)':>9}"
        ,
        flush=True,
    )
    print("-" * 132, flush=True)

    for M, N, K, label in BENCHMARK_SHAPES:
        try:
            print(f"[{label}] preparing tensors and cuBLAS baseline", flush=True)
            A_cp = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
            B_cp = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
            A_t = torch.as_tensor(A_cp, device="cuda")
            B_t = torch.as_tensor(B_cp, device="cuda")

            for _ in range(30):
                torch.mm(A_t, B_t)
            torch.cuda.synchronize()
            ms_a = timed_torch(lambda: torch.mm(A_t, B_t), warmup=20, iters=200)

            t0 = time.perf_counter()
            print(f"[{label}] searching up to {max_shapes} unit shapes", flush=True)
            frontier = search_optimal_wave_shape(
                M=M,
                N=N,
                K=K,
                tile_m_candidates=tile_m_options,
                tile_n_candidates=tile_n_options,
                splitk_candidates=splitk_options,
                hw=hw,
                shape_type="polycube",
                max_shapes=max_shapes,
                num_workers=num_workers,
                polycube_strategy="greedy",
            )
            search_s = time.perf_counter() - t0

            if not frontier:
                print(f"{label:28} | search produced no candidates")
                rows.append(
                    {
                        "label": label,
                        "M": M,
                        "N": N,
                        "K": K,
                        "ms_cublas": ms_a,
                        "ms_no_wave": float("nan"),
                        "ms_wave": float("nan"),
                        "speedup_D_over_B": float("nan"),
                        "speedup_D_over_A": float("nan"),
                        "search_time_s": search_s,
                        "best_config": None,
                    }
                )
                continue

            ms_d = float("inf")
            ms_b_for_best_d = float("inf")
            best_d = None
            best_b_plan = None
            best_d_plan = None

            for cfg in frontier[:top_eval]:
                if not isinstance(cfg.wave_shape, PolycubeWaveShape):
                    continue
                try:
                    tile_k = cfg.tile_k if hasattr(cfg, "tile_k") else 32
                    b_plan = build_rowmajor_compiled_plan(
                        M,
                        N,
                        K,
                        cfg.tile_m,
                        cfg.tile_n,
                        tile_k,
                        cfg.splitk_factor,
                    )
                    ms_b = timed_median(
                        launch_polycube_compiled,
                        A_cp,
                        B_cp,
                        b_plan,
                        warmup=10,
                        iters=200,
                        repeats=3,
                    )

                    try:
                        d_plan = build_polycube_compiled_plan(
                            M,
                            N,
                            K,
                            cfg.tile_m,
                            cfg.tile_n,
                            tile_k,
                            cfg.splitk_factor,
                            cfg.wave_shape,
                        )
                    except NotImplementedError as exc:
                        print(
                            f"[{label}] skip candidate tile=({cfg.tile_m},{cfg.tile_n},{tile_k}) splitk={cfg.splitk_factor}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    ms_d_candidate = timed_median(
                        launch_polycube_compiled,
                        A_cp,
                        B_cp,
                        d_plan,
                        warmup=10,
                        iters=200,
                        repeats=3,
                    )

                    if ms_d_candidate < ms_d:
                        ms_d = ms_d_candidate
                        ms_b_for_best_d = ms_b
                        best_d = cfg
                        best_b_plan = b_plan
                        best_d_plan = d_plan
                    print(
                        f"[{label}] evaluated candidate tile=({cfg.tile_m},{cfg.tile_n},{tile_k}) splitk={cfg.splitk_factor} wave={cfg.to_dict()['wave_shape']} B={ms_b:.3f}ms D={ms_d_candidate:.3f}ms",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"{label}: candidate failed: {type(exc).__name__}: {exc}")
                    continue

            if best_d is None or not np.isfinite(ms_b_for_best_d) or not np.isfinite(ms_d):
                print(f"{label:28} | no valid polycube candidate")
                rows.append(
                    {
                        "label": label,
                        "M": M,
                        "N": N,
                        "K": K,
                        "ms_cublas": ms_a,
                        "ms_no_wave": float("nan"),
                        "ms_wave": float("nan"),
                        "speedup_D_over_B": float("nan"),
                        "speedup_D_over_A": float("nan"),
                        "search_time_s": search_s,
                        "best_config": None,
                    }
                )
                continue

            db = ms_b_for_best_d / ms_d
            da = ms_a / ms_d
            tflops_a = tflops(M, N, K, ms_a)
            tflops_d = tflops(M, N, K, ms_d)

            print(
                f"{label:28} | {_format_ms(ms_a):>10} | {_format_ms(tflops_a):>10} | {_format_ms(ms_b_for_best_d):>8} | {_format_ms(ms_d):>8} | {_format_speedup(db):>6} | {_format_speedup(da):>6} | {_format_ms(tflops_d):>8} | {search_s:>9.2f}"
                ,
                flush=True,
            )

            kernel_compare_dump = None
            if dump_kernel_compare:
                kernel_compare_dump = _dump_case_kernel_compare(
                    Path(kernel_dump_root) / "kernel_compare_greedy",
                    label,
                    M,
                    N,
                    K,
                    best_d,
                    best_b_plan,
                    best_d_plan,
                    ms_a,
                    ms_b_for_best_d,
                    ms_d,
                )

            rows.append(
                {
                    "label": label,
                    "M": M,
                    "N": N,
                    "K": K,
                    "ms_cublas": ms_a,
                    "ms_no_wave": ms_b_for_best_d,
                    "ms_wave": ms_d,
                    "speedup_D_over_B": db,
                    "speedup_D_over_A": da,
                    "search_time_s": search_s,
                    "best_config": {
                        "tile_m": best_d.tile_m,
                        "tile_n": best_d.tile_n,
                        "tile_k": best_d.tile_k if hasattr(best_d, "tile_k") else 32,
                        "splitk_factor": best_d.splitk_factor,
                        "wave_shape": best_d.to_dict()["wave_shape"],
                        "wave_count": best_d.wave_count,
                        "benefit": best_d.benefit,
                    },
                    "kernel_compare_dump": kernel_compare_dump,
                }
            )
        except Exception as exc:
            print(f"{label}: failed: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "label": label,
                    "M": M,
                    "N": N,
                    "K": K,
                    "ms_cublas": float("nan"),
                    "ms_no_wave": float("nan"),
                    "ms_wave": float("nan"),
                    "speedup_D_over_B": float("nan"),
                    "speedup_D_over_A": float("nan"),
                    "search_time_s": float("nan"),
                    "best_config": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    finite_db = [r["speedup_D_over_B"] for r in rows if np.isfinite(r["speedup_D_over_B"])]
    finite_da = [r["speedup_D_over_A"] for r in rows if np.isfinite(r["speedup_D_over_A"])]
    mean_db = float(np.mean(finite_db)) if finite_db else float("nan")
    max_db = float(np.max(finite_db)) if finite_db else float("nan")
    mean_da = float(np.mean(finite_da)) if finite_da else float("nan")

    print("-" * 132)
    print(
        f"{'MEAN':28} | {'':>10} | {'':>10} | {'':>8} | {'':>8} | {_format_speedup(mean_db):>6} | {_format_speedup(mean_da):>6} | {'':>8} | {'':>9}"
        ,
        flush=True,
    )
    print("Block Ordering (D/B) breakdown:", flush=True)
    if finite_db:
        best = max(rows, key=lambda r: r["speedup_D_over_B"] if np.isfinite(r["speedup_D_over_B"]) else -1)
        worst = min(rows, key=lambda r: r["speedup_D_over_B"] if np.isfinite(r["speedup_D_over_B"]) else 1e9)
        print(f"Best  case: {best['label']}  +{(best['speedup_D_over_B'] - 1.0) * 100.0:.1f}%", flush=True)
        print(f"Worst case: {worst['label']}  +{(worst['speedup_D_over_B'] - 1.0) * 100.0:.1f}%", flush=True)
        print(f"Shapes where D/B > 1.05x: {sum(1 for r in rows if np.isfinite(r['speedup_D_over_B']) and r['speedup_D_over_B'] > 1.05)}/{len(rows)}", flush=True)
        print(f"Shapes where D/B > 1.10x: {sum(1 for r in rows if np.isfinite(r['speedup_D_over_B']) and r['speedup_D_over_B'] > 1.10)}/{len(rows)}", flush=True)
    else:
        print(f"Best  case: n/a  +0.0%", flush=True)
        print(f"Worst case: n/a  +0.0%", flush=True)
        print(f"Shapes where D/B > 1.05x: 0/{len(rows)}", flush=True)
        print(f"Shapes where D/B > 1.10x: 0/{len(rows)}", flush=True)

    payload = {
        "gpu": _gpu_name(),
        "sm_count": int(hw.sm_count),
        "shapes": rows,
        "summary": {
            "mean_D_over_B": mean_db,
            "max_D_over_B": max_db,
            "mean_D_over_A": mean_da,
            "note": "B and D use identical polycube_runtime_kernel; only LUT order differs.",
        },
    }
    Path("results/attribution_raw.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shapes", type=int, default=50)
    parser.add_argument("--top-eval", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--dump-kernel-compare", action="store_true")
    parser.add_argument("--kernel-dump-root", type=str, default="results")
    args = parser.parse_args()

    hw_raw = json.load(open("calibrated_hw_params.json"))
    hw_profile = json.load(open("results/hw_profile.json"))
    hw = HardwareParams(
        alpha=float(hw_raw.get("alpha", 1.0)),
        beta=float(hw_raw.get("beta", 2.0)),
        gamma=float(hw_raw.get("gamma", 2.0)),
        delta=float(hw_raw.get("delta", 0.0)),
        sm_count=int(hw_profile["num_sms"]),
        l2_cache_bytes=int(hw_profile["l2_cache_bytes"]),
        shared_mem_bytes=int(hw_profile["smem_per_sm_bytes"]),
    )

    run_attribution_benchmark(
        hw=hw,
        num_workers=max(1, args.num_workers),
        max_shapes=max(1, args.max_shapes),
        top_eval=max(1, args.top_eval),
        dump_kernel_compare=args.dump_kernel_compare,
        kernel_dump_root=args.kernel_dump_root,
    )


if __name__ == "__main__":
    main()
