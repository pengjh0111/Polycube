"""Regime-correct fair benchmark restricted to true multi-wave configurations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import CuboidWaveShape, HardwareParams, TaskSpace3D, compute_wave_count
from wave_tiling.search_v2 import search_v2

if __package__ is None or __package__ == "":
    from cutile_gemm.baseline_gemm import launch_baseline
    from cutile_gemm.wave_tiling_gemm import launch_wave_tiling
else:
    from .baseline_gemm import launch_baseline
    from .wave_tiling_gemm import launch_wave_tiling


NUM_SMS = 170

# All entries are intended to be hw_waves >= 5 for 170 SMs.
MULTIWAVE_CONFIGS = [
    (16, 4096, 8192, 16, 64, 32, 16),
    (16, 4096, 8192, 16, 64, 32, 32),
    (32, 4096, 8192, 16, 64, 32, 8),
    (32, 4096, 8192, 16, 64, 32, 16),
    (64, 4096, 8192, 16, 64, 32, 8),
    (256, 4096, 8192, 16, 64, 32, 1),
    (256, 4096, 8192, 32, 128, 32, 1),
    (512, 4096, 8192, 16, 64, 32, 1),
    (512, 4096, 8192, 32, 128, 32, 1),
    (1024, 4096, 8192, 16, 64, 32, 1),
    (128, 4096, 8192, 16, 64, 32, 4),
    (256, 4096, 8192, 16, 64, 32, 4),
]


def timed_launch(fn, *args, warmup: int = 3, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(*args)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


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


def run_multiwave_benchmark(hw: HardwareParams, num_sms: int) -> list[dict]:
    results: list[dict] = []

    for M, N, K, tile_m, tile_n, tile_k, splitk in MULTIWAVE_CONFIGS:
        tm = math.ceil(M / tile_m)
        tn = math.ceil(N / tile_n)
        total_blocks = tm * tn * splitk
        hw_waves_flat = math.ceil(total_blocks / num_sms)

        if hw_waves_flat < 5:
            print(f"SKIP (only {hw_waves_flat} hw waves): M={M} N={N} K={K} sk={splitk}")
            continue

        rng = cp.random.default_rng(0)
        A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

        ms_base = timed_launch(launch_baseline, A, B, tile_m, tile_n, tile_k, splitk, warmup=3, iters=50)
        ms_flat = timed_launch(launch_wave_tiling, A, B, tile_m, tile_n, tile_k, splitk, 1, 1, 1, warmup=3, iters=50)
        flat_overhead = ms_flat / ms_base if ms_base > 0 else float("inf")

        task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
        candidates = search_v2(task=task, num_sms=num_sms, hw=hw, allow_partial=True, max_shapes=300)

        best_ms = float("inf")
        best_shape: CuboidWaveShape | None = None
        best_wc: int | None = None
        best_benefit: float | None = None

        for wc, benefit, shape in candidates[:10]:
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
                warmup=3,
                iters=50,
            )
            if ms < best_ms:
                best_ms = ms
                best_shape = shape
                best_wc = wc
                best_benefit = benefit

        if best_shape is None or best_wc is None:
            continue

        shape_size = best_shape.size()
        padded_blocks = best_wc * shape_size
        hw_waves_wave = math.ceil(padded_blocks / num_sms)
        hw_waste = (padded_blocks - total_blocks) / total_blocks
        speedup = ms_base / best_ms if best_ms > 0 else 0.0

        entry = {
            "M": M,
            "N": N,
            "K": K,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "splitk": splitk,
            "tm_dim": tm,
            "tn_dim": tn,
            "sk_dim": splitk,
            "total_blocks": total_blocks,
            "hw_waves_flat": hw_waves_flat,
            "hw_waves_wave": hw_waves_wave,
            "ms_baseline": ms_base,
            "ms_wave_flat": ms_flat,
            "flat_overhead": flat_overhead,
            "ms_wave_best": best_ms,
            "best_shape": [best_shape.sk, best_shape.m, best_shape.n],
            "logical_waves": best_wc,
            "shape_size": shape_size,
            "predicted_benefit": best_benefit,
            "hw_waste_pct": hw_waste,
            "wave_speedup": speedup,
        }
        results.append(entry)

        print(
            f"M={M:5d} N={N:5d} K={K:6d} sk={splitk:2d} | "
            f"blocks={total_blocks:5d} hw_waves={hw_waves_flat:3d} | "
            f"base={ms_base:.4f}ms flat={ms_flat:.4f}ms({flat_overhead:.2f}x) "
            f"best={best_ms:.4f}ms({speedup:.4f}x) "
            f"shape=({best_shape.sk},{best_shape.m},{best_shape.n})"
        )

    return results


def exhaustive_shape_search(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk: int,
    num_sms: int,
    A,
    B,
) -> list[dict]:
    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
    all_rows: list[dict] = []

    for ws_k in range(1, task.sk_dim + 1):
        for ws_m in range(1, task.tm_dim + 1):
            for ws_n in range(1, task.tn_dim + 1):
                if ws_k * ws_m * ws_n > num_sms:
                    continue
                shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
                wc = compute_wave_count(task, shape)
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
                all_rows.append(
                    {
                        "ms": ms,
                        "ws_k": ws_k,
                        "ws_m": ws_m,
                        "ws_n": ws_n,
                        "logical_waves": wc,
                        "shape_size": shape.size(),
                    }
                )

    all_rows.sort(key=lambda r: r["ms"])
    print(f"\nTop-10 shapes by latency (M={M}, N={N}, K={K}, tm={tile_m}, tn={tile_n}, sk={splitk}):")
    for row in all_rows[:10]:
        print(
            f"  ({row['ws_k']},{row['ws_m']},{row['ws_n']}) "
            f"wc={row['logical_waves']} ms={row['ms']:.4f}"
        )

    flat_ms = next((r["ms"] for r in all_rows if r["ws_k"] == 1 and r["ws_m"] == 1 and r["ws_n"] == 1), None)
    best_ms = all_rows[0]["ms"] if all_rows else None
    if flat_ms is not None and best_ms is not None and best_ms > 0:
        print(f"Flat: {flat_ms:.4f}ms | Best: {best_ms:.4f}ms | Speedup: {flat_ms / best_ms:.4f}x")

    return all_rows


def analyze_by_hw_waves(results: list[dict]) -> dict:
    buckets: dict[str, list[float]] = {"1-4": [], "5-10": [], "11-25": [], "26+": []}
    for r in results:
        hw = int(r["hw_waves_flat"])
        if hw < 5:
            key = "1-4"
        elif hw < 11:
            key = "5-10"
        elif hw < 26:
            key = "11-25"
        else:
            key = "26+"
        buckets[key].append(float(r["wave_speedup"]))

    print("=== Speedup by Hardware Wave Count ===")
    out: dict[str, dict] = {}
    for key in ["1-4", "5-10", "11-25", "26+"]:
        values = buckets[key]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        print(
            f"  hw_waves {key:5s}: n={len(values):3d} "
            f"mean={arr.mean():.4f}x max={arr.max():.4f}x min={arr.min():.4f}x"
        )
        out[key] = {"n": len(values), "mean": float(arr.mean()), "max": float(arr.max()), "min": float(arr.min())}

    all_waves = np.asarray([float(r["hw_waves_flat"]) for r in results], dtype=np.float64)
    all_speedups = np.asarray([float(r["wave_speedup"]) for r in results], dtype=np.float64)
    corr = 0.0
    if all_waves.size > 1 and all_waves.std() > 1e-12 and all_speedups.std() > 1e-12:
        corr = float(np.corrcoef(all_waves, all_speedups)[0, 1])
        print(f"\nr(hw_waves, speedup) = {corr:.3f}")

    out["hw_waves_speedup_corr"] = corr
    return out


def l2_analysis(M: int, N: int, K: int, tile_m: int, tile_n: int, splitk: int, l2_bytes: int, num_sms: int) -> str:
    A_bytes = M * K * 2
    B_bytes = K * N * 2

    tm = math.ceil(M / tile_m)
    tn = math.ceil(N / tile_n)
    total_blocks = tm * tn * splitk

    per_sm_l2 = l2_bytes / num_sms

    ws_m_max = min(tm, num_sms)
    ws_n_max = min(tn, max(1, num_sms // max(ws_m_max, 1)))

    lines = []
    lines.append(f"L2 Analysis: M={M} N={N} K={K} tm={tile_m} tn={tile_n} sk={splitk}")
    lines.append(f"  total_blocks={total_blocks}, hw_waves={math.ceil(total_blocks / num_sms)}")
    lines.append(
        f"  A matrix: {A_bytes/1e6:.1f} MB | {'FITS' if A_bytes < l2_bytes else 'SPILLS'} in L2 ({l2_bytes/1e6:.1f} MB)"
    )
    lines.append(
        f"  B matrix: {B_bytes/1e6:.1f} MB | {'FITS' if B_bytes < l2_bytes else 'SPILLS'} in L2"
    )
    lines.append(
        f"  A per SM: {A_bytes/num_sms/1e3:.1f} KB | B per SM: {B_bytes/num_sms/1e3:.1f} KB | L2 per SM: {per_sm_l2/1e3:.1f} KB"
    )
    lines.append(f"  With wave shape (1,{ws_m_max},1):")
    lines.append(
        f"    A unique per wave: {ws_m_max * tile_m * K * 2 / 1e6:.2f} MB | A reuse factor approx {ws_n_max:.1f}x"
    )
    lines.append(f"    B unique per wave: {ws_n_max * tile_n * K * 2 / 1e6:.2f} MB")
    return "\n".join(lines)


def _dominant_axis(shape: list[int]) -> str:
    ws_k, ws_m, ws_n = shape
    if ws_k > 1 and ws_m == 1 and ws_n == 1:
        return "K"
    if ws_m > 1 and ws_k == 1 and ws_n == 1:
        return "M"
    if ws_n > 1 and ws_k == 1 and ws_m == 1:
        return "N"
    return "mixed"


def write_definitive_report(
    multiwave_results: list[dict],
    exhaustive_results: dict[str, list[dict]],
    hw_wave_analysis: dict,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Definitive Multi-Wave Report")
    lines.append("")
    lines.append("## H1: Logical Waves vs Hardware Waves")
    lines.append("- Hardware waves are ceil(total_blocks / num_sms).")
    lines.append("- Logical waves are shape-dependent schedule groups; they are not the same metric.")
    lines.append("- Multi-wave-only benchmark avoids the single-wave no-effect regime.")
    lines.append("")

    mw = [r for r in multiwave_results if r["hw_waves_flat"] >= 5]
    speedups = [float(r["wave_speedup"]) for r in mw]
    lines.append("## H3: Latency Effect in Multi-Wave Regime")
    if speedups:
        arr = np.asarray(speedups, dtype=np.float64)
        lines.append(f"- N configs: {len(arr)}")
        lines.append(f"- Mean speedup: {arr.mean():.4f}x")
        lines.append(f"- Max speedup: {arr.max():.4f}x")
        lines.append(f"- % > 1.02x: {np.mean(arr > 1.02):.0%}")
        lines.append(f"- % > 1.05x: {np.mean(arr > 1.05):.0%}")
        lines.append(f"- % > 1.10x: {np.mean(arr > 1.10):.0%}")
    else:
        lines.append("- No valid multi-wave measurements.")

    lines.append("")
    lines.append("## H2: Predictiveness in Multi-Wave Regime")
    pred = np.asarray([float(r["predicted_benefit"]) for r in mw], dtype=np.float64)
    spd = np.asarray([float(r["wave_speedup"]) for r in mw], dtype=np.float64)
    corr_benefit_speedup = 0.0
    if pred.size > 1 and pred.std() > 1e-12 and spd.std() > 1e-12:
        corr_benefit_speedup = float(np.corrcoef(pred, spd)[0, 1])
    lines.append(f"- r(benefit_score, speedup) on hw_waves>=5: {corr_benefit_speedup:.3f}")

    lines.append("")
    lines.append("## H4: Optimal Shape Structure (Exhaustive)")
    for key, rows in exhaustive_results.items():
        if not rows:
            lines.append(f"- {key}: no rows")
            continue
        best = rows[0]
        shape = [best["ws_k"], best["ws_m"], best["ws_n"]]
        lines.append(
            f"- {key}: best_shape={tuple(shape)}, best_ms={best['ms']:.4f}, dominant_axis={_dominant_axis(shape)}"
        )

    lines.append("")
    lines.append("## Gate Results")
    gate_a = len(mw) >= 8
    gate_b = float(hw_wave_analysis.get("hw_waves_speedup_corr", 0.0)) > 0.0
    gate_c = any((r["hw_waves_flat"] >= 10 and float(r["wave_speedup"]) > 1.05) for r in mw)
    gate_d = len(exhaustive_results) >= 2
    lines.append(f"- Gate A (>=8 configs with hw_waves>=5): {'PASS' if gate_a else 'FAIL'}")
    lines.append(f"- Gate B (r(hw_waves, speedup) > 0): {'PASS' if gate_b else 'FAIL'}")
    lines.append(f"- Gate C (exists hw_waves>=10 with speedup>1.05): {'PASS' if gate_c else 'FAIL'}")
    lines.append(f"- Gate D (exhaustive on >=2 configs): {'PASS' if gate_d else 'FAIL'}")

    if not gate_c:
        lines.append("")
        lines.append("## Gate C Failure Follow-up")
        lines.append("- Possible explanation 1: kernel remains compute-bound, so reordering has weak impact.")
        lines.append("- Possible explanation 2: runtime scheduling may weaken user-level block ordering effects.")
        lines.append("- Recommended next step: contrast a memory-bound variant (smaller tile_k) against compute-heavy variant.")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    hw = _load_calibrated_hw_params()
    num_sms = NUM_SMS

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    multiwave_rows = run_multiwave_benchmark(hw, num_sms)
    out_multi = results_dir / "multiwave_benchmark.json"
    out_multi.write_text(json.dumps(multiwave_rows, indent=2))
    print(f"Saved: {out_multi}")

    # Exhaustive search on two key configs.
    exhaustive_targets = [
        (1024, 4096, 8192, 16, 64, 32, 1),
        (32, 4096, 8192, 16, 64, 32, 16),
    ]
    exhaustive_out: dict[str, list[dict]] = {}
    for M, N, K, tile_m, tile_n, tile_k, splitk in exhaustive_targets:
        rng = cp.random.default_rng(0)
        A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
        key = f"M{M}_N{N}_K{K}_tm{tile_m}_tn{tile_n}_sk{splitk}"
        exhaustive_out[key] = exhaustive_shape_search(M, N, K, tile_m, tile_n, tile_k, splitk, num_sms, A, B)

    out_exh = results_dir / "exhaustive_shape_search.json"
    out_exh.write_text(json.dumps(exhaustive_out, indent=2))
    print(f"Saved: {out_exh}")

    hw_wave_analysis = analyze_by_hw_waves(multiwave_rows)

    l2_lines: list[str] = []
    l2_lines.append(f"GPU L2 bytes: {hw.l2_cache_bytes}")
    l2_lines.append(f"Assumed num_sms: {num_sms}")
    l2_lines.append("")
    for cfg in MULTIWAVE_CONFIGS:
        M, N, K, tile_m, tile_n, _tile_k, splitk = cfg
        l2_lines.append(l2_analysis(M, N, K, tile_m, tile_n, splitk, hw.l2_cache_bytes, num_sms))
        l2_lines.append("")
    out_l2 = results_dir / "l2_analysis.txt"
    out_l2.write_text("\n".join(l2_lines))
    print(f"Saved: {out_l2}")

    out_report = results_dir / "definitive_report.md"
    write_definitive_report(multiwave_rows, exhaustive_out, hw_wave_analysis, out_report)
    print(f"Saved: {out_report}")


if __name__ == "__main__":
    main()
