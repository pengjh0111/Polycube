"""Benchmark utilities for baseline vs 3D wave-tiling cuTile GEMM."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np

from wave_tiling import HardwareParams, TaskSpace3D, search_optimal_wave_shape

from .baseline_gemm import launch_baseline
from .wave_tiling_gemm import launch_wave_tiling

SHAPES = [
    (16, 4096, 8192),
    (32, 4096, 8192),
    (64, 4096, 8192),
    (32, 8192, 8192),
    (128, 4096, 4096),
    (32, 4096, 16384),
]

TILE_CONFIGS = [
    (16, 64, 32),
    (32, 128, 32),
]


def get_hw_params() -> HardwareParams:
    dev = cp.cuda.Device(0)
    return HardwareParams(
        sm_count=dev.attributes["MultiProcessorCount"],
        l2_cache_bytes=dev.attributes["L2CacheSize"],
    )


def timed_launch(fn, *args, warmup=3, iters=50):
    for _ in range(warmup):
        fn(*args)
    cp.cuda.get_current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    cp.cuda.get_current_stream().synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def benchmark_all(hw: HardwareParams, output_path: str | Path = "benchmark_results.json"):
    results = []
    for M, N, K in SHAPES:
        for tile_m, tile_n, tile_k in TILE_CONFIGS:
            rng = cp.random.default_rng(0)
            A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
            B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

            for splitk in (1, 2, 4, 8):
                task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
                lb = task.lower_bound_waves(hw.sm_count)

                ms_base = timed_launch(
                    launch_baseline,
                    A,
                    B,
                    tile_m,
                    tile_n,
                    tile_k,
                    splitk,
                    warmup=5,
                    iters=30,
                )

                pareto = search_optimal_wave_shape(
                    M,
                    N,
                    K,
                    tile_m_candidates=[tile_m],
                    tile_n_candidates=[tile_n],
                    splitk_candidates=[splitk],
                    hw=hw,
                    max_shapes=200,
                )

                if not pareto:
                    continue

                # Per-axis references for H3
                sk_only_shape = (splitk, 1, 1)
                mn_only_shape = (1, max(1, task.tm_dim), 1)
                ms_sk_only = timed_launch(
                    launch_wave_tiling,
                    A,
                    B,
                    tile_m,
                    tile_n,
                    tile_k,
                    splitk,
                    *sk_only_shape,
                    warmup=5,
                    iters=20,
                )
                ms_mn_only = timed_launch(
                    launch_wave_tiling,
                    A,
                    B,
                    tile_m,
                    tile_n,
                    tile_k,
                    splitk,
                    *mn_only_shape,
                    warmup=5,
                    iters=20,
                )

                for pr in pareto[:3]:
                    shape = pr.wave_shape
                    ms_wt = timed_launch(
                        launch_wave_tiling,
                        A,
                        B,
                        tile_m,
                        tile_n,
                        tile_k,
                        splitk,
                        shape.sk,
                        shape.m,
                        shape.n,
                        warmup=5,
                        iters=30,
                    )
                    entry = {
                        "M": M,
                        "N": N,
                        "K": K,
                        "tile_m": tile_m,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "splitk": splitk,
                        "wave_shape": [shape.sk, shape.m, shape.n],
                        "predicted_waves": pr.wave_count,
                        "lower_bound_waves": lb,
                        "benefit_score": pr.benefit,
                        "ms_baseline": ms_base,
                        "ms_wave_tiling": ms_wt,
                        "speedup": ms_base / ms_wt,
                        "ms_sk_only": ms_sk_only,
                        "ms_mn_only": ms_mn_only,
                        "joint_beats_axis_best": ms_wt < min(ms_sk_only, ms_mn_only),
                    }
                    results.append(entry)
                    print(
                        f"M={M} N={N} K={K} sk={splitk} shape={(shape.sk, shape.m, shape.n)} "
                        f"waves={pr.wave_count}(lb={lb}) base={ms_base:.3f}ms wt={ms_wt:.3f}ms "
                        f"speedup={entry['speedup']:.2f}x"
                    )

    output_path = Path(output_path)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} benchmark rows to {output_path}")
    return results


def _pearson(x, y):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2:
        return 0.0
    x_std = x_arr.std()
    y_std = y_arr.std()
    if x_std == 0 or y_std == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def hypothesis_test(results):
    if not results:
        raise ValueError("results is empty")

    wave_excess = [
        (r["predicted_waves"] - r["lower_bound_waves"]) / r["lower_bound_waves"]
        for r in results
        if r["lower_bound_waves"] > 0
    ]
    mean_excess = float(np.mean(wave_excess)) if wave_excess else 0.0
    print(f"H1: Mean wave count excess over lower bound: {mean_excess:.2%}")

    benefits = [r["benefit_score"] for r in results]
    latencies = [r["ms_wave_tiling"] for r in results]
    corr = _pearson(benefits, latencies)
    print(f"H2: Pearson r(benefit, latency) = {corr:.3f} (expected < 0)")

    joint_wins_baseline = sum(1 for r in results if r["speedup"] > 1.05)
    joint_wins_axis = sum(1 for r in results if r["joint_beats_axis_best"])
    print(
        f"H3a: Joint faster than baseline (>5%) in {joint_wins_baseline}/{len(results)} "
        f"({joint_wins_baseline / len(results):.1%})"
    )
    print(
        f"H3b: Joint beats best axis-only in {joint_wins_axis}/{len(results)} "
        f"({joint_wins_axis / len(results):.1%})"
    )

    return {
        "h1_mean_wave_excess": mean_excess,
        "h2_pearson": corr,
        "h3_joint_win_rate_vs_baseline": joint_wins_baseline / len(results),
        "h3_joint_win_rate_vs_axis": joint_wins_axis / len(results),
    }


def _run_ncu_profile(module: str, launch_expr: str, out_prefix: str):
    launch_name = launch_expr.split("(")[0]
    py_snippet = "\n".join(
        [
            "import cupy as cp",
            f"from {module} import {launch_name}",
            "A = cp.random.standard_normal((32, 8192), dtype=cp.float32).astype(cp.float16)",
            "B = cp.random.standard_normal((8192, 4096), dtype=cp.float32).astype(cp.float16)",
            "for _ in range(5):",
            f"    {launch_expr}",
        ]
    )
    cmd = [
        "ncu",
        "--metrics",
        "l2_global_hit_rate,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum",
        "--csv",
        "-o",
        out_prefix,
        "python3",
        "-c",
        py_snippet,
    ]
    subprocess.run(cmd, check=True)


def run_ncu_comparison(
    baseline_call="launch_baseline(A, B, 16, 64, 32, 4)",
    wave_call="launch_wave_tiling(A, B, 16, 64, 32, 4, 2, 4, 4)",
    output_txt: str | Path = "ncu_comparison.txt",
):
    output_txt = Path(output_txt)
    if not shutil.which("ncu"):
        output_txt.write_text("ncu not found in PATH; profiling skipped.\n")
        print(output_txt.read_text().strip())
        return

    try:
        _run_ncu_profile("cutile_gemm.baseline_gemm", baseline_call, "profile_baseline")
        _run_ncu_profile("cutile_gemm.wave_tiling_gemm", wave_call, "profile_wave_tiling")
    except subprocess.CalledProcessError as exc:
        output_txt.write_text(f"ncu profiling failed: {exc}\n")
        print(output_txt.read_text().strip())
        return

    lines = [
        "NCU profiling completed.",
        "Compare profile_baseline.csv vs profile_wave_tiling.csv for:",
        "- l2_global_hit_rate",
        "- sm__warps_active.avg.pct_of_peak_sustained_active",
        "- gpu__time_duration.sum",
    ]
    output_txt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    hw = get_hw_params()
    rows = benchmark_all(hw)
    summary = hypothesis_test(rows)
    print(json.dumps(summary, indent=2))
    run_ncu_comparison()
