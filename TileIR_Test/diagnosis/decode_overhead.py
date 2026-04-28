"""Measure decode-only overhead for flat vs wave-tiling block scheduling."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import cupy as cp
import cuda.tile as ct

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import CuboidWaveShape, HardwareParams, TaskSpace3D
from wave_tiling.search_v2 import search_v2


NUM_SMS = 170

# Representative configs spanning the profitable and crossover regime.
DECODE_CONFIGS = [
    (48, 4096, 8192, 16, 64, 32, 4),
    (96, 4096, 8192, 16, 64, 32, 2),
    (128, 4096, 8192, 16, 64, 32, 2),
    (160, 4096, 8192, 16, 64, 32, 2),
    (192, 4096, 8192, 16, 64, 32, 2),
    (224, 4096, 8192, 16, 64, 32, 2),
    (240, 4096, 8192, 16, 64, 32, 2),
    (1024, 4096, 8192, 16, 64, 32, 1),
]


@ct.kernel
def flat_decode_only(
    C,
    M,
    N,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    splitk_factor: ct.Constant[int],
):
    bid = ct.bid(0)
    num_blocks_m = ct.cdiv(M, tile_m)
    num_blocks_n = ct.cdiv(N, tile_n)
    pid_m = bid // (num_blocks_n * splitk_factor)
    pid_n = (bid // splitk_factor) % num_blocks_n
    pid_sk = bid % splitk_factor
    if pid_m >= num_blocks_m:
        return
    ct.atomic_add(C, (pid_m, pid_n), ct.full((), 0.0, dtype=ct.float32), check_bounds=True)


@ct.kernel
def wave_decode_only(
    C,
    M,
    N,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    splitk_factor: ct.Constant[int],
    ws_k: ct.Constant[int],
    ws_m: ct.Constant[int],
    ws_n: ct.Constant[int],
):
    bid = ct.bid(0)
    num_blocks_m = ct.cdiv(M, tile_m)
    num_blocks_n = ct.cdiv(N, tile_n)
    wave_size = ws_k * ws_m * ws_n
    wave_idx = bid // wave_size
    local_idx = bid % wave_size
    local_sk = local_idx // (ws_m * ws_n)
    local_m = (local_idx % (ws_m * ws_n)) // ws_n
    local_n = local_idx % ws_n
    waves_n = ct.cdiv(num_blocks_n, ws_n)
    waves_m = ct.cdiv(num_blocks_m, ws_m)
    wave_sk = wave_idx // (waves_m * waves_n)
    wave_m = (wave_idx % (waves_m * waves_n)) // waves_n
    wave_n = wave_idx % waves_n
    pid_sk = wave_sk * ws_k + local_sk
    pid_m = wave_m * ws_m + local_m
    pid_n = wave_n * ws_n + local_n
    if pid_m >= num_blocks_m or pid_n >= num_blocks_n or pid_sk >= splitk_factor:
        return
    ct.atomic_add(C, (pid_m, pid_n), ct.full((), 0.0, dtype=ct.float32), check_bounds=True)


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


def timed_launch(fn, warmup: int = 5, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def main() -> None:
    hw = _load_calibrated_hw_params()
    C = cp.zeros((512, 512), dtype=cp.float32)
    rows: list[dict] = []

    print(f"{'blocks':>8} {'hw_waves':>9} {'flat_us':>10} {'wave_us':>10} {'overhead%':>10}")

    for M, N, K, tile_m, tile_n, tile_k, splitk in DECODE_CONFIGS:
        task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
        total_blocks = task.total_blocks()
        hw_waves = math.ceil(total_blocks / NUM_SMS)
        if hw_waves < 5:
            continue

        candidates = search_v2(task=task, num_sms=NUM_SMS, hw=hw, allow_partial=True, max_shapes=50)
        if not candidates:
            continue
        _, _, best_shape = candidates[0]

        flat_launch = lambda: ct.launch(
            cp.cuda.get_current_stream(),
            (total_blocks, 1, 1),
            flat_decode_only,
            (C, M, N, tile_m, tile_n, splitk),
        )
        wave_launch = lambda: ct.launch(
            cp.cuda.get_current_stream(),
            (total_blocks, 1, 1),
            wave_decode_only,
            (C, M, N, tile_m, tile_n, splitk, best_shape.sk, best_shape.m, best_shape.n),
        )

        flat_us = timed_launch(flat_launch)
        wave_us = timed_launch(wave_launch)
        overhead_pct = (wave_us - flat_us) / flat_us * 100.0 if flat_us > 0 else 0.0

        row = {
            "M": M,
            "N": N,
            "K": K,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "splitk": splitk,
            "total_blocks": total_blocks,
            "hw_waves": hw_waves,
            "best_shape": [best_shape.sk, best_shape.m, best_shape.n],
            "flat_us": flat_us,
            "wave_us": wave_us,
            "overhead_pct": overhead_pct,
        }
        rows.append(row)

        print(f"{total_blocks:8d} {hw_waves:9d} {flat_us:10.2f} {wave_us:10.2f} {overhead_pct:10.1f}")

    out_path = Path("results/decode_overhead.json")
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved: {out_path}")

    if rows:
        overheads = np.asarray([r["overhead_pct"] for r in rows], dtype=np.float64)
        buckets = {
            "5-10": [r["overhead_pct"] for r in rows if 5 <= r["hw_waves"] <= 10],
            "11+": [r["overhead_pct"] for r in rows if r["hw_waves"] >= 11],
        }
        print("=== Decode Overhead Summary ===")
        print(f"mean={float(np.mean(overheads)):.1f}% max={float(np.max(overheads)):.1f}%")
        for name, values in buckets.items():
            if values:
                print(f"bucket {name}: mean={float(np.mean(values)):.1f}% n={len(values)}")


if __name__ == "__main__":
    import numpy as np

    main()
