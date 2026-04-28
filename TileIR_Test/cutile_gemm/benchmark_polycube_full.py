"""Full polycube-first benchmark with optional CPU multi-process search.

Search stage uses true polycube candidates.
Execution stage uses native polycube runtime mapping via per-block LUT decode.
"""

from __future__ import annotations

import argparse
import inspect
import json
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
    build_polycube_plan,
    launch_polycube_plan,
    polycube_runtime_kernel,
)
from wave_tiling.benefit_model import HardwareParams
from wave_tiling.search import search_optimal_wave_shape
from wave_tiling.task_space import TaskSpace3D
from wave_tiling.wave_shape import PolycubeWaveShape

LLM_DECODE_SHAPES = [
    (1, 4096, 4096, "LLaMA-2 Attn, bs=1"),
    (4, 4096, 4096, "LLaMA-2 Attn, bs=4"),
    (8, 4096, 4096, "LLaMA-2 Attn, bs=8"),
    (16, 4096, 4096, "LLaMA-2 Attn, bs=16"),
    (1, 4096, 11008, "LLaMA-2 FFN, bs=1"),
    (4, 4096, 11008, "LLaMA-2 FFN, bs=4"),
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


def run_polycube_full_test(
    hw: HardwareParams,
    num_workers: int,
    max_shapes: int,
    top_eval: int,
    pair_samples: int,
    dump_kernel_compare: bool,
    kernel_dump_root: str,
):
    tile_m_options = [16, 32, 64]
    tile_n_options = [32, 64, 128, 256]
    splitk_options = [1, 2, 4, 8, 16, 32, 64]

    rows = []

    print(
        f"\n{'Shape':28} | {'cuBLAS':>7} | {'B same-cfg':>10} | {'D poly-order':>12} | "
        f"{'D/B':>6} | {'D/A':>6} | {'search(s)':>9}"
    )
    print("-" * 112)

    for M, N, K, label in LLM_DECODE_SHAPES:
        A_cp = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B_cp = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
        A_t = torch.as_tensor(A_cp, device="cuda")
        B_t = torch.as_tensor(B_cp, device="cuda")

        for _ in range(30):
            torch.mm(A_t, B_t)
        torch.cuda.synchronize()
        ms_a = timed_torch(lambda: torch.mm(A_t, B_t), warmup=20, iters=200)

        t0 = time.perf_counter()
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
            continue

        candidates = frontier[:top_eval]

        ms_d = float("inf")
        ms_b_for_best_d = float("inf")
        best_d = None
        best_b_plan = None
        best_d_plan = None
        best_plan_blocks = None
        pair_rows = []
        for cfg in candidates:
            if not isinstance(cfg.wave_shape, PolycubeWaveShape):
                continue
            try:
                tile_k = cfg.tile_k if hasattr(cfg, "tile_k") else 32

                b_plan = _build_default_rowmajor_plan(
                    M,
                    N,
                    K,
                    cfg.tile_m,
                    cfg.tile_n,
                    tile_k,
                    cfg.splitk_factor,
                )
                ms_b_pair = timed_cupy(
                    launch_polycube_plan,
                    A_cp,
                    B_cp,
                    b_plan,
                    warmup=2,
                    iters=20,
                )

                d_plan = build_polycube_plan(
                    M,
                    N,
                    K,
                    cfg.tile_m,
                    cfg.tile_n,
                    tile_k,
                    cfg.splitk_factor,
                    cfg.wave_shape,
                    strategy="greedy",
                )
                ms = timed_cupy(
                    launch_polycube_plan,
                    A_cp,
                    B_cp,
                    d_plan,
                    warmup=3,
                    iters=40,
                )

                if ms < ms_d:
                    ms_d = ms
                    ms_b_for_best_d = ms_b_pair
                    best_d = cfg
                    best_plan_blocks = int(d_plan.pid_sk_lut.shape[0])
                    best_b_plan = b_plan
                    best_d_plan = d_plan

                pair_rows.append(
                    {
                        "tile_m": cfg.tile_m,
                        "tile_n": cfg.tile_n,
                        "tile_k": tile_k,
                        "splitk": cfg.splitk_factor,
                        "shape_type": type(cfg.wave_shape).__name__,
                        "shape": cfg.to_dict()["wave_shape"],
                        "wave_count": cfg.wave_count,
                        "benefit": cfg.benefit,
                        "runtime_total_blocks": int(d_plan.pid_sk_lut.shape[0]),
                        "ms_b_same_cfg_no_wave": ms_b_pair,
                        "ms_d_same_cfg_polycube": ms,
                        "speedup_d_over_b_same_cfg": (ms_b_pair / ms) if np.isfinite(ms_b_pair) and np.isfinite(ms) and ms > 0 else float("nan"),
                    }
                )
            except Exception:
                continue

        if best_d is None or not np.isfinite(ms_b_for_best_d) or not np.isfinite(ms_d):
            continue

        db = ms_b_for_best_d / ms_d
        da = ms_a / ms_d

        print(
            f"{label:28} | {ms_a:>7.3f} | {ms_b_for_best_d:>10.3f} | {ms_d:>12.3f} | "
            f"{db:>6.3f}x | {da:>6.3f}x | {search_s:>9.2f}"
        )

        kernel_compare_dump = None
        if dump_kernel_compare:
            dump_dir = Path(kernel_dump_root) / "kernel_compare_greedy"
            kernel_compare_dump = _dump_case_kernel_compare(
                dump_dir,
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
                "ms_opt_no_wave_same_cfg": ms_b_for_best_d,
                "ms_polycube_ordered_same_cfg": ms_d,
                "speedup_wave_ordering_same_cfg": db,
                "speedup_total_vs_cublas": da,
                "search_time_s": search_s,
                "search_num_workers": num_workers,
                "search_max_shapes": max_shapes,
                "search_polycube_strategy": "greedy",
                "execution_mode": "native_polycube_runtime_lut",
                "same_cfg_no_wave_for_best_poly": {
                    "tile_m": best_d.tile_m,
                    "tile_n": best_d.tile_n,
                    "tile_k": best_d.tile_k if hasattr(best_d, "tile_k") else 32,
                    "splitk": best_d.splitk_factor,
                    "wave_shape": {"type": "default_row_major_lut"},
                    "note": "No-extra-order baseline using row-major LUT with the same runtime kernel and same tile/splitk as best polycube config.",
                },
                "best_poly_native": {
                    "tile_m": best_d.tile_m,
                    "tile_n": best_d.tile_n,
                    "tile_k": best_d.tile_k if hasattr(best_d, "tile_k") else 32,
                    "splitk": best_d.splitk_factor,
                    "wave_count": best_d.wave_count,
                    "benefit": best_d.benefit,
                    "shape_type": type(best_d.wave_shape).__name__,
                    "runtime_total_blocks": best_plan_blocks,
                    "shape": best_d.to_dict()["wave_shape"],
                },
                "paired_same_cfg_samples": sorted(
                    pair_rows,
                    key=lambda r: r["speedup_d_over_b_same_cfg"],
                    reverse=True,
                )[: max(1, pair_samples)],
                "kernel_compare_dump": kernel_compare_dump,
            }
        )

    dbs = [r["speedup_wave_ordering_same_cfg"] for r in rows]
    das = [r["speedup_total_vs_cublas"] for r in rows]
    search_ts = [r["search_time_s"] for r in rows]

    summary = {
        "n_shapes": len(rows),
        "polycube_strategy": "greedy",
        "mean_wave_speedup": float(np.mean(dbs)) if dbs else 0.0,
        "max_wave_speedup": float(np.max(dbs)) if dbs else 0.0,
        "mean_total_speedup_vs_cublas": float(np.mean(das)) if das else 0.0,
        "max_total_speedup_vs_cublas": float(np.max(das)) if das else 0.0,
        "mean_search_time_s": float(np.mean(search_ts)) if search_ts else 0.0,
        "max_search_time_s": float(np.max(search_ts)) if search_ts else 0.0,
        "note": "B and D use the same runtime kernel; only LUT mapping differs (default row-major vs polycube order).",
    }

    out = {"summary": summary, "results": rows}
    out_path = "results/e2e_polycube_full_greedy.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print("\n=== Polycube Full Test Summary ===")
    if rows:
        print(f"wave ordering (projected) mean D/B: {summary['mean_wave_speedup']:.3f}x max: {summary['max_wave_speedup']:.3f}x")
        print(f"total vs cuBLAS mean D/A: {summary['mean_total_speedup_vs_cublas']:.3f}x max: {summary['max_total_speedup_vs_cublas']:.3f}x")
        print(f"search time mean: {summary['mean_search_time_s']:.2f}s max: {summary['max_search_time_s']:.2f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--max-shapes", type=int, default=120)
    parser.add_argument("--top-eval", type=int, default=12)
    parser.add_argument("--pair-samples", type=int, default=3)
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

    run_polycube_full_test(
        hw,
        num_workers=max(1, args.num_workers),
        max_shapes=max(1, args.max_shapes),
        top_eval=max(1, args.top_eval),
        pair_samples=max(1, args.pair_samples),
        dump_kernel_compare=args.dump_kernel_compare,
        kernel_dump_root=args.kernel_dump_root,
    )
