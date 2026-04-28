"""Software proxy for cache reuse under wave schedules."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import CuboidWaveShape, TaskSpace3D


def estimate_unique_cachelines_per_wave(
    task: TaskSpace3D,
    shape: CuboidWaveShape,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    K: int,
    N: int,
    cacheline_bytes: int = 128,
    elem_bytes: int = 2,
) -> dict[str, float]:
    """Estimate unique A/B cachelines touched by one representative wave."""

    k_per_sk = math.ceil(K / max(task.sk_dim, 1))
    k_iters = math.ceil(k_per_sk / tile_k)

    a_lines: set[int] = set()
    b_lines: set[int] = set()

    a_tile_bytes = tile_m * tile_k * elem_bytes
    b_tile_bytes = tile_k * tile_n * elem_bytes
    a_lines_per_tile = max(1, math.ceil(a_tile_bytes / cacheline_bytes))
    b_lines_per_tile = max(1, math.ceil(b_tile_bytes / cacheline_bytes))

    for sk in range(shape.sk):
        k_base = sk * k_per_sk
        for m in range(shape.m):
            for n in range(shape.n):
                for ki in range(k_iters):
                    k = k_base + ki * tile_k
                    a_base_line = ((m * tile_m * K + k) * elem_bytes) // cacheline_bytes
                    b_base_line = ((k * N + n * tile_n) * elem_bytes) // cacheline_bytes
                    for offset in range(a_lines_per_tile):
                        a_lines.add(a_base_line + offset)
                    for offset in range(b_lines_per_tile):
                        b_lines.add(b_base_line + offset)

    total_unique = len(a_lines) + len(b_lines)
    total_accesses = shape.sk * shape.m * shape.n * k_iters * (a_lines_per_tile + b_lines_per_tile)
    reuse_ratio = 1.0 - total_unique / max(total_accesses, 1)

    return {
        "unique_A_lines": float(len(a_lines)),
        "unique_B_lines": float(len(b_lines)),
        "total_unique": float(total_unique),
        "total_accesses": float(total_accesses),
        "reuse_ratio": float(reuse_ratio),
    }


def proxy_correlation(results_path: str | Path = "benchmark_results.json") -> dict[str, float]:
    rows = json.loads(Path(results_path).read_text())
    reuse = []
    effective_reuse = []
    latency = []
    for row in rows:
        task = TaskSpace3D.from_problem(
            row["M"],
            row["N"],
            row["K"],
            row["tile_m"],
            row["tile_n"],
            row["splitk"],
        )
        sk, m, n = row["wave_shape"]
        shape = CuboidWaveShape(sk=sk, m=m, n=n)
        proxy = estimate_unique_cachelines_per_wave(
            task=task,
            shape=shape,
            tile_m=row["tile_m"],
            tile_n=row["tile_n"],
            tile_k=row.get("tile_k", 32),
            K=row["K"],
            N=row["N"],
        )
        reuse.append(proxy["reuse_ratio"])
        effective_reuse.append(proxy["reuse_ratio"] / max(row.get("predicted_waves", 1), 1))
        latency.append(row["ms_wave_tiling"])

    reuse_arr = np.asarray(reuse, dtype=np.float64)
    effective_arr = np.asarray(effective_reuse, dtype=np.float64)
    lat_arr = np.asarray(latency, dtype=np.float64)
    corr = float(np.corrcoef(reuse_arr, lat_arr)[0, 1]) if len(reuse_arr) > 1 else 0.0
    effective_corr = float(np.corrcoef(effective_arr, lat_arr)[0, 1]) if len(effective_arr) > 1 else 0.0
    print(f"r(proxy_reuse, latency) = {corr:.4f}")
    print(f"r(effective_proxy_reuse, latency) = {effective_corr:.4f}")
    return {"proxy_latency_corr": corr, "effective_proxy_latency_corr": effective_corr}


if __name__ == "__main__":
    proxy_correlation()
