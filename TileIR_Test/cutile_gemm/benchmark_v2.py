"""Fair benchmark: isolate block-ordering effect between baseline and wave-tiling."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cupy as cp
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from diagnosis.shape_classification import FIXED_CONFIGS, find_promising_shapes
from wave_tiling import CuboidWaveShape, HardwareParams, TaskSpace3D, compute_benefit, compute_wave_count
from wave_tiling.search_v2 import search_v2

if __package__ is None or __package__ == "":
    from cutile_gemm.baseline_gemm import launch_baseline
    from cutile_gemm.benchmark import timed_launch
    from cutile_gemm.wave_tiling_gemm import launch_wave_tiling
else:
    from .baseline_gemm import launch_baseline
    from .benchmark import timed_launch
    from .wave_tiling_gemm import launch_wave_tiling


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


def analyze_best_achievable(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    splitk: int,
    num_sms: int,
    hw: HardwareParams,
) -> list[tuple[int, float, CuboidWaveShape]]:
    """Per-axis potential analysis for fixed split-k task space."""

    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
    lb = task.lower_bound_waves(num_sms)

    print(
        f"\nShape analysis: M={M} N={N} K={K} tm={tile_m} tn={tile_n} sk={splitk}\n"
        f"  Task space: ({task.sk_dim}, {task.tm_dim}, {task.tn_dim}) = {task.total_blocks()} blocks, lb={lb} waves"
    )

    axis_shapes = [
        ("flat   ", 1, 1, 1),
        ("k-heavy", min(task.sk_dim, num_sms), 1, 1),
        ("m-heavy", 1, min(task.tm_dim, num_sms), 1),
        ("n-heavy", 1, 1, min(task.tn_dim, num_sms)),
    ]

    for name, ws_k, ws_m, ws_n in axis_shapes:
        shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
        if shape.size() > num_sms:
            continue
        wc = compute_wave_count(task, shape)
        bf = compute_benefit(shape, hw)
        print(f"  {name}: ({ws_k},{ws_m},{ws_n}) size={shape.size()} waves={wc} benefit={bf:.3f}")

    candidates = search_v2(task=task, num_sms=num_sms, hw=hw, allow_partial=True, max_shapes=200)
    if candidates:
        wc, bf, shape = candidates[0]
        print(f"  best   : ({shape.sk},{shape.m},{shape.n}) waves={wc} benefit={bf:.3f}")
    return candidates


def _collect_analysis_for_promising(promising_configs: list[dict], num_sms: int, hw: HardwareParams) -> None:
    for cfg in promising_configs:
        analyze_best_achievable(
            M=cfg["M"],
            N=cfg["N"],
            K=cfg["K"],
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n"],
            splitk=cfg["splitk"],
            num_sms=num_sms,
            hw=hw,
        )


def run_fair_benchmark(hw_calibrated: HardwareParams, num_sms: int, target_keys: set[tuple[int, int, int, int, int, int]]) -> list[dict]:
    """Run fixed-splitk baseline vs wave-tiling where only mapping changes."""

    results = []

    for M, N, K, tile_m, tile_n, splitk in sorted(target_keys):
        tile_k = 32

        rng = cp.random.default_rng(0)
        A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
        B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

        ms_base = timed_launch(
            launch_baseline,
            A,
            B,
            tile_m,
            tile_n,
            tile_k,
            splitk,
            warmup=3,
            iters=50,
        )

        task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
        candidates = search_v2(
            task=task,
            num_sms=num_sms,
            hw=hw_calibrated,
            allow_partial=True,
            max_shapes=100,
        )

        best_ms_wt = float("inf")
        best_shape: CuboidWaveShape | None = None
        best_wc: int | None = None
        best_benefit: float | None = None

        for wc, benefit, shape in candidates[:5]:
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
                warmup=3,
                iters=50,
            )
            if ms_wt < best_ms_wt:
                best_ms_wt = ms_wt
                best_shape = shape
                best_wc = wc
                best_benefit = benefit

        ms_flat = timed_launch(
            launch_wave_tiling,
            A,
            B,
            tile_m,
            tile_n,
            tile_k,
            splitk,
            1,
            1,
            1,
            warmup=3,
            iters=50,
        )

        flat_overhead = ms_flat / ms_base if ms_base > 0 else float("inf")
        lower_bound = task.lower_bound_waves(num_sms)
        flat_wc = compute_wave_count(task, CuboidWaveShape(sk=1, m=1, n=1))

        row = {
            "M": M,
            "N": N,
            "K": K,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "splitk": splitk,
            "ms_baseline": ms_base,
            "ms_wave_flat": ms_flat,
            "flat_overhead": flat_overhead,
            "ms_wave_best": best_ms_wt if best_shape is not None else None,
            "best_shape": [best_shape.sk, best_shape.m, best_shape.n] if best_shape else None,
            "wave_speedup": (ms_base / best_ms_wt) if best_shape is not None and best_ms_wt > 0 else None,
            "predicted_waves": int(best_wc) if best_wc is not None else None,
            "predicted_benefit": float(best_benefit) if best_benefit is not None else None,
            "lower_bound_waves": int(lower_bound),
            "flat_waves": int(flat_wc),
            "wave_reduction_vs_flat": ((flat_wc - best_wc) / flat_wc) if best_wc is not None and flat_wc > 0 else None,
        }
        results.append(row)

        print(
            f"M={M:4d} N={N:5d} K={K:6d} tm={tile_m} tn={tile_n} sk={splitk}: "
            f"base={ms_base:.4f}ms "
            f"flat={ms_flat:.4f}ms({flat_overhead:.2f}x) "
            f"best={best_ms_wt:.4f}ms({row['wave_speedup']:.2f}x) "
            f"shape={best_shape.sk if best_shape else '?'}"
            f",{best_shape.m if best_shape else '?'}"
            f",{best_shape.n if best_shape else '?'}"
        )

    return results


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.std() < 1e-12 or y_arr.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def generate_final_report(fair_results: list[dict], promising_configs: list[dict], report_path: Path) -> dict:
    """Generate honest summary with gate checks and promising-only analysis."""

    lines: list[str] = []
    lines.append("# Fair Benchmark Honest Report")
    lines.append("")
    lines.append("## Section 1: Sanity Checks")

    sanity_failures = []
    for r in fair_results:
        overhead = r["flat_overhead"]
        if not (0.85 <= overhead <= 1.20):
            sanity_failures.append(r)
            lines.append(
                f"- WARN: M={r['M']} N={r['N']} K={r['K']} tm={r['tile_m']} tn={r['tile_n']} sk={r['splitk']} flat_overhead={overhead:.3f}"
            )

    if not sanity_failures:
        lines.append("- All sanity checks passed (flat wave approx baseline).")
    else:
        lines.append(f"- Sanity failures: {len(sanity_failures)}")

    lines.append("")
    lines.append("## Section 2: Wave-Tiling Effect on Promising Shapes")
    promising_keys = {
        (p["M"], p["N"], p["K"], p["tile_m"], p["tile_n"], p["splitk"])
        for p in promising_configs
    }
    promising_results = [
        r
        for r in fair_results
        if (r["M"], r["N"], r["K"], r["tile_m"], r["tile_n"], r["splitk"]) in promising_keys
    ]

    promising_speedups = [r["wave_speedup"] for r in promising_results if r["wave_speedup"] is not None]
    if not promising_speedups:
        lines.append("- No promising configs were benchmarked.")
    else:
        lines.append(f"- Count: {len(promising_speedups)}")
        lines.append(f"- Mean speedup: {sum(promising_speedups) / len(promising_speedups):.3f}x")
        lines.append(f"- Max speedup: {max(promising_speedups):.3f}x")
        lines.append(
            f"- % > 1.05x: {sum(s > 1.05 for s in promising_speedups) / len(promising_speedups):.0%}"
        )
        lines.append(
            f"- % > 1.10x: {sum(s > 1.10 for s in promising_speedups) / len(promising_speedups):.0%}"
        )

    lines.append("")
    lines.append("## Section 3: Conclusions")
    all_speedups = [r["wave_speedup"] for r in fair_results if r["wave_speedup"] is not None]
    mean_all = (sum(all_speedups) / len(all_speedups)) if all_speedups else 0.0
    best_all = max(all_speedups) if all_speedups else 0.0
    frac_gt_1 = (sum(s > 1.0 for s in all_speedups) / len(all_speedups)) if all_speedups else 0.0

    lines.append(f"- Overall mean speedup (all configs): {mean_all:.3f}x")
    lines.append(f"- Overall best speedup (all configs): {best_all:.3f}x")
    lines.append(f"- % configs > 1.0x: {frac_gt_1:.0%}")

    corr_promising = _corr(
        [float(r["predicted_waves"]) for r in promising_results if r["predicted_waves"] is not None],
        [float(r["ms_wave_best"]) for r in promising_results if r["ms_wave_best"] is not None],
    )
    lines.append(f"- Pearson r(predicted_wave_count, latency) on promising: {corr_promising:.3f}")

    gate1 = len(sanity_failures) == 0
    gate2 = len(promising_configs) >= 5
    gate3 = False
    if promising_speedups:
        gate3 = (sum(s > 1.0 for s in promising_speedups) / len(promising_speedups)) >= 0.30
    gate4 = corr_promising < -0.3

    lines.append("")
    lines.append("## Gate Status")
    lines.append(f"- Gate 1 (sanity): {'PASS' if gate1 else 'FAIL'}")
    lines.append(f"- Gate 2 (>=5 promising): {'PASS' if gate2 else 'FAIL'}")
    lines.append(f"- Gate 3 (>=30% promising configs >1.0x): {'PASS' if gate3 else 'FAIL'}")
    lines.append(f"- Gate 4 (r < -0.3 on promising): {'PASS' if gate4 else 'FAIL'}")

    report_path.write_text("\n".join(lines) + "\n")

    return {
        "gate1_sanity": gate1,
        "gate2_promising": gate2,
        "gate3_speedup": gate3,
        "gate4_corr": gate4,
        "num_total": len(fair_results),
        "num_promising": len(promising_configs),
        "promising_corr_predwaves_latency": corr_promising,
        "overall_mean_speedup": mean_all,
        "overall_best_speedup": best_all,
        "overall_fraction_gt1": frac_gt_1,
    }


def main() -> None:
    hw = _load_calibrated_hw_params()
    num_sms = cp.cuda.Device(0).attributes["MultiProcessorCount"]

    classification = find_promising_shapes(num_sms, hw)
    promising = classification["promising"]

    if promising:
        _collect_analysis_for_promising(promising, num_sms, hw)
    else:
        print("No promising configs found; benchmarking full fixed grid for completeness.")

    if promising:
        target_keys = {
            (p["M"], p["N"], p["K"], p["tile_m"], p["tile_n"], p["splitk"])
            for p in promising
        }
    else:
        target_keys = {
            (M, N, K, tile_m, tile_n, splitk)
            for M, N, K in classification["used_shapes"]
            for tile_m, tile_n, _tile_k, splitk in FIXED_CONFIGS
        }

    fair_rows = run_fair_benchmark(hw, num_sms, target_keys)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    class_path = results_dir / "shape_classification.json"
    class_path.write_text(json.dumps(classification, indent=2))

    fair_path = results_dir / "fair_benchmark_results.json"
    fair_path.write_text(json.dumps(fair_rows, indent=2))

    report_path = results_dir / "final_honest_report.md"
    summary = generate_final_report(fair_rows, promising, report_path)

    print(f"Saved: {class_path}")
    print(f"Saved: {fair_path}")
    print(f"Saved: {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
