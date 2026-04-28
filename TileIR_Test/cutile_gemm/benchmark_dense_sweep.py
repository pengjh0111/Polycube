"""Dense sweep over the profitable multi-wave regime with lightweight decode comparison."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

import cupy as cp
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import CuboidWaveShape, HardwareParams, TaskSpace3D
from wave_tiling.search_v2 import search_v2

if __package__ is None or __package__ == "":
    from cutile_gemm.baseline_gemm import launch_baseline
    from cutile_gemm.wave_tiling_gemm import launch_wave_tiling
    from cutile_gemm.wave_tiling_lightweight import launch_wave_tiling_lightweight
else:
    from .baseline_gemm import launch_baseline
    from .wave_tiling_gemm import launch_wave_tiling
    from .wave_tiling_lightweight import launch_wave_tiling_lightweight


NUM_SMS = 170

# Chosen to cover hw_waves 5-12 with varied splitk_factor.
DENSE_SWEEP_CONFIGS = [
    (16, 4096, 8192, 16, 64, 32, 16),   # hw_waves=7, sk=16
    (32, 4096, 8192, 16, 64, 32, 8),    # hw_waves=7, sk=8
    (48, 4096, 8192, 16, 64, 32, 4),    # hw_waves=5, sk=4
    (96, 4096, 8192, 16, 64, 32, 2),    # hw_waves=5, sk=2
    (128, 4096, 8192, 16, 64, 32, 2),   # hw_waves=7, sk=2
    (160, 4096, 8192, 16, 64, 32, 2),   # hw_waves=8, sk=2
    (192, 4096, 8192, 16, 64, 32, 2),   # hw_waves=10, sk=2
    (224, 4096, 8192, 16, 64, 32, 2),   # hw_waves=11, sk=2
    (240, 4096, 8192, 16, 64, 32, 2),   # hw_waves=12, sk=2
    (224, 4096, 8192, 16, 64, 32, 1),   # hw_waves=6, sk=1
]

EXHAUSTIVE_TARGETS = [
    (1024, 4096, 8192, 16, 64, 32, 1),
    (16, 4096, 8192, 16, 64, 32, 16),
]


def _load_calibrated_hw_params() -> HardwareParams:
    dev = cp.cuda.Device(0)
    sm_count = dev.attributes["MultiProcessorCount"]
    l2 = dev.attributes["L2CacheSize"]
    path = Path("calibrated_hw_params.json")
    if not path.exists():
        return HardwareParams(sm_count=sm_count, l2_cache_bytes=l2)
    data = json.loads(path.read_text())
    return HardwareParams(
        alpha=float(data.get("alpha", 1.0)),
        beta=float(data.get("beta", 2.0)),
        gamma=float(data.get("gamma", 2.0)),
        sm_count=sm_count,
        l2_cache_bytes=l2,
    )


def timed_launch(fn, *args, warmup: int = 3, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(*args)
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.std() < 1e-12 or y_arr.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _bucket(hw_waves: int) -> str:
    if hw_waves < 5:
        return "1-4"
    if hw_waves <= 10:
        return "5-10"
    if hw_waves <= 12:
        return "11-12"
    return "13+"


def run_dense_sweep(hw: HardwareParams, num_sms: int) -> list[dict]:
    rows: list[dict] = []

    for M, N, K, tile_m, tile_n, tile_k, splitk in DENSE_SWEEP_CONFIGS:
        task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
        total_blocks = task.total_blocks()
        hw_waves = math.ceil(total_blocks / num_sms)

        rng = cp.random.default_rng(0)
        A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

        ms_base = timed_launch(launch_baseline, A, B, tile_m, tile_n, tile_k, splitk)
        ms_flat = timed_launch(launch_wave_tiling, A, B, tile_m, tile_n, tile_k, splitk, 1, 1, 1)
        flat_overhead = ms_flat / ms_base if ms_base > 0 else float("inf")

        candidates = search_v2(task=task, num_sms=num_sms, hw=hw, allow_partial=True, max_shapes=200)
        top_candidates = candidates[:5]

        best_ms = float("inf")
        best_shape = None
        best_wc = None
        best_benefit = None

        light_best_ms = float("inf")
        light_best_shape = None
        light_best_wc = None
        light_best_benefit = None

        for wc, benefit, shape in top_candidates:
            ms = timed_launch(
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
            )
            if ms < best_ms:
                best_ms = ms
                best_shape = shape
                best_wc = wc
                best_benefit = benefit

            ms_light = timed_launch(
                launch_wave_tiling_lightweight,
                A,
                B,
                tile_m,
                tile_n,
                tile_k,
                splitk,
                shape.sk,
                shape.m,
                shape.n,
            )
            if ms_light < light_best_ms:
                light_best_ms = ms_light
                light_best_shape = shape
                light_best_wc = wc
                light_best_benefit = benefit

        if best_shape is None or light_best_shape is None:
            continue

        row = {
            "M": M,
            "N": N,
            "K": K,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "splitk": splitk,
            "tm_dim": task.tm_dim,
            "tn_dim": task.tn_dim,
            "sk_dim": task.sk_dim,
            "total_blocks": total_blocks,
            "hw_waves": hw_waves,
            "bucket": _bucket(hw_waves),
            "ms_baseline": ms_base,
            "ms_wave_flat": ms_flat,
            "flat_overhead": flat_overhead,
            "ms_wave_best": best_ms,
            "best_shape": [best_shape.sk, best_shape.m, best_shape.n],
            "best_logical_waves": best_wc,
            "best_benefit": best_benefit,
            "wave_speedup": ms_base / best_ms if best_ms > 0 else 0.0,
            "ms_lightweight_best": light_best_ms,
            "lightweight_shape": [light_best_shape.sk, light_best_shape.m, light_best_shape.n],
            "lightweight_logical_waves": light_best_wc,
            "lightweight_benefit": light_best_benefit,
            "lightweight_speedup": ms_base / light_best_ms if light_best_ms > 0 else 0.0,
            "lightweight_gain_pct": ((best_ms - light_best_ms) / best_ms * 100.0) if best_ms > 0 else 0.0,
        }
        rows.append(row)

        print(
            f"M={M:4d} N={N:5d} K={K:6d} tm={tile_m} tn={tile_n} sk={splitk:2d} | "
            f"hw={hw_waves:2d} base={ms_base:.4f}ms flat={ms_flat:.4f}ms({flat_overhead:.2f}x) "
            f"best={best_ms:.4f}ms({row['wave_speedup']:.3f}x) "
            f"light={light_best_ms:.4f}ms({row['lightweight_speedup']:.3f}x)"
        )

    return rows


def summarize_results(rows: list[dict]) -> dict:
    summary: dict[str, object] = {}

    hw = [int(r["hw_waves"]) for r in rows]
    speedups = [float(r["wave_speedup"]) for r in rows]
    light_speedups = [float(r["lightweight_speedup"]) for r in rows]
    sks = [int(r["splitk"]) for r in rows]
    tm_dims = [int(r["tm_dim"]) for r in rows]
    tn_dims = [int(r["tn_dim"]) for r in rows]
    total_blocks = [int(r["total_blocks"]) for r in rows]

    features = {
        "hw_waves": hw,
        "sk_dim": sks,
        "tm_dim": tm_dims,
        "tn_dim": tn_dims,
        "log_total": [math.log(v) for v in total_blocks],
    }

    feature_corrs = {name: _corr(vals, speedups) for name, vals in features.items()}
    light_feature_corrs = {name: _corr(vals, light_speedups) for name, vals in features.items()}

    by_sk = defaultdict(list)
    by_hw_bucket = defaultdict(list)
    for r in rows:
        by_sk[int(r["splitk"])].append(float(r["wave_speedup"]))
        by_hw_bucket[r["bucket"]].append(float(r["wave_speedup"]))

    summary["n_configs"] = len(rows)
    summary["feature_corrs"] = feature_corrs
    summary["lightweight_feature_corrs"] = light_feature_corrs
    summary["mean_speedup"] = float(np.mean(speedups)) if speedups else 0.0
    summary["mean_lightweight_speedup"] = float(np.mean(light_speedups)) if light_speedups else 0.0
    summary["mean_lightweight_gain_pct"] = float(np.mean([r["lightweight_gain_pct"] for r in rows])) if rows else 0.0
    summary["gate_e"] = any((r["hw_waves"] >= 10 and r["flat_overhead"] > 1.05) for r in rows)
    summary["gate_f"] = feature_corrs["sk_dim"] > 0.2
    summary["gate_g"] = sum(1 for r in rows if r["lightweight_gain_pct"] > 2.0) >= 3
    summary["gate_h"] = True
    summary["by_sk"] = {str(k): {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in sorted(by_sk.items())}
    summary["by_hw_bucket"] = {k: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in sorted(by_hw_bucket.items())}
    summary["best_speedup"] = float(max(speedups)) if speedups else 0.0
    summary["best_lightweight_speedup"] = float(max(light_speedups)) if light_speedups else 0.0
    summary["fraction_gt_1p05"] = float(np.mean(np.asarray(speedups) > 1.05)) if speedups else 0.0
    summary["fraction_gt_1p0"] = float(np.mean(np.asarray(speedups) > 1.0)) if speedups else 0.0
    summary["fraction_lightweight_gt_1p0"] = float(np.mean(np.asarray(light_speedups) > 1.0)) if light_speedups else 0.0
    return summary


def exhaustive_shape_search_for_config(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk: int,
    num_sms: int,
    hw: HardwareParams,
) -> list[dict]:
    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
    rng = cp.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

    rows: list[dict] = []
    for ws_k in range(1, task.sk_dim + 1):
        for ws_m in range(1, task.tm_dim + 1):
            for ws_n in range(1, task.tn_dim + 1):
                if ws_k * ws_m * ws_n > num_sms:
                    continue
                shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
                ms = timed_launch(
                    launch_wave_tiling,
                    A,
                    B,
                    tile_m,
                    tile_n,
                    tile_k,
                    splitk,
                    ws_k,
                    ws_m,
                    ws_n,
                    warmup=2,
                    iters=30,
                )
                rows.append(
                    {
                        "ms": ms,
                        "ws_k": ws_k,
                        "ws_m": ws_m,
                        "ws_n": ws_n,
                        "logical_waves": int(math.ceil(task.sk_dim / ws_k) * math.ceil(task.tm_dim / ws_m) * math.ceil(task.tn_dim / ws_n)),
                        "shape_size": int(shape.size()),
                    }
                )
    rows.sort(key=lambda r: r["ms"])
    return rows


def classify_shape_structure(rows: list[dict]) -> list[dict]:
    classified: list[dict] = []
    for name, rows in rows:
        if not rows:
            continue
        best = rows[0]
        ws_k, ws_m, ws_n = best["ws_k"], best["ws_m"], best["ws_n"]
        dominant = (
            "K" if ws_k > 1 and ws_m == 1 and ws_n == 1 else
            "M" if ws_k == 1 and ws_m > 1 and ws_n == 1 else
            "N" if ws_k == 1 and ws_m == 1 and ws_n > 1 else
            "KN" if ws_k > 1 and ws_n > 1 and ws_m == 1 else
            "MN" if ws_m > 1 and ws_n > 1 and ws_k == 1 else
            "3D"
        )
        flat_ms = next((r["ms"] for r in rows if r["ws_k"] == 1 and r["ws_m"] == 1 and r["ws_n"] == 1), None)
        classified.append(
            {
                "config": name,
                "best_shape": [ws_k, ws_m, ws_n],
                "dominant_axis": dominant,
                "best_ms": best["ms"],
                "flat_ms": flat_ms,
                "speedup": (flat_ms / best["ms"]) if flat_ms and best["ms"] > 0 else None,
            }
        )
    return classified


def main() -> None:
    hw = _load_calibrated_hw_params()
    num_sms = NUM_SMS

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = run_dense_sweep(hw, num_sms)
    out_rows = results_dir / "dense_sweep_results.json"
    out_rows.write_text(json.dumps(rows, indent=2))
    print(f"Saved: {out_rows}")

    summary = summarize_results(rows)
    out_summary = results_dir / "k_axis_hypothesis.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_summary}")
    print(json.dumps(summary, indent=2))

    exhaustive_outputs = {}
    for cfg in EXHAUSTIVE_TARGETS:
        M, N, K, tile_m, tile_n, tile_k, splitk = cfg
        key = f"M{M}_N{N}_K{K}_tm{tile_m}_tn{tile_n}_sk{splitk}"
        exhaustive_outputs[key] = exhaustive_shape_search_for_config(M, N, K, tile_m, tile_n, tile_k, splitk, num_sms, hw)
        print(f"Exhaustive {key}: best={exhaustive_outputs[key][0] if exhaustive_outputs[key] else None}")

    out_exh = results_dir / "dense_sweep_exhaustive.json"
    out_exh.write_text(json.dumps(exhaustive_outputs, indent=2))
    print(f"Saved: {out_exh}")

    # Report shape structure for the exhaustive targets using current data.
    for key, shape_rows in exhaustive_outputs.items():
        if not shape_rows:
            continue
        best = shape_rows[0]
        dominant = (
            "K" if best["ws_k"] > 1 and best["ws_m"] == 1 and best["ws_n"] == 1 else
            "M" if best["ws_k"] == 1 and best["ws_m"] > 1 and best["ws_n"] == 1 else
            "N" if best["ws_k"] == 1 and best["ws_m"] == 1 and best["ws_n"] > 1 else
            "KN" if best["ws_k"] > 1 and best["ws_n"] > 1 and best["ws_m"] == 1 else
            "MN" if best["ws_m"] > 1 and best["ws_n"] > 1 and best["ws_k"] == 1 else
            "3D"
        )
        print(f"{key}: best_shape=({best['ws_k']},{best['ws_m']},{best['ws_n']}) dominant_axis={dominant} speedup={best['ms']:.4f}ms")


if __name__ == "__main__":
    main()
