"""Layer 2 split-k selector for unified three-layer search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class SplitKConfig:
    splitk: int
    hw_waves: int
    eta: float
    rho_wave: float
    total_blocks: int
    cost_compute: float
    cost_reduce: float


def select_splitk_candidates(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    M: int,
    N: int,
    K: int,
    num_sms: int,
    c_atomic_normalized: float = 0.05,
    eta_min: float = 0.85,
    splitk_options: tuple = (1, 2, 4, 8, 16, 32, 64),
) -> List[SplitKConfig]:
    T_m = math.ceil(M / tile_m)
    T_n = math.ceil(N / tile_n)

    raw_candidates: List[SplitKConfig] = []
    for s in splitk_options:
        if s * tile_k > K:
            continue
        k_per_partition = K // s
        if k_per_partition < tile_k:
            continue

        total = T_m * T_n * s
        hw_w = math.ceil(total / num_sms)
        eta = total / (hw_w * num_sms)
        rho_w = 1.0 - eta

        cost_compute = float(hw_w)
        cost_reduce = (s - 1) * c_atomic_normalized

        raw_candidates.append(
            SplitKConfig(
                splitk=s,
                hw_waves=hw_w,
                eta=eta,
                rho_wave=rho_w,
                total_blocks=total,
                cost_compute=cost_compute,
                cost_reduce=cost_reduce,
            )
        )

    if not raw_candidates:
        return []

    candidates = [c for c in raw_candidates if c.splitk == 1 or c.eta >= eta_min]

    # For small-M decode shapes, strict eta can remove all useful splitk>1 options.
    if len([c for c in candidates if c.splitk > 1]) == 0:
        extra = sorted(
            [c for c in raw_candidates if c.splitk > 1],
            key=lambda c: (-c.eta, c.cost_reduce),
        )[:3]
        candidates.extend(extra)

    candidates.sort(key=lambda c: (c.hw_waves, c.rho_wave, c.cost_reduce))
    return candidates


def find_optimal_splitk(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    M: int,
    N: int,
    K: int,
    num_sms: int,
    c_atomic_normalized: float = 0.05,
) -> int:
    candidates = select_splitk_candidates(
        tile_m,
        tile_n,
        tile_k,
        M,
        N,
        K,
        num_sms,
        c_atomic_normalized=c_atomic_normalized,
        eta_min=0.0,
    )
    if not candidates:
        return 1
    best = min(candidates, key=lambda c: c.cost_compute + c.cost_reduce)
    return best.splitk


def splitk_utilization_table(tile_m: int, tile_n: int, M: int, N: int, K: int, num_sms: int) -> None:
    T_m = math.ceil(M / tile_m)
    T_n = math.ceil(N / tile_n)
    print(
        f"\n=== SplitK Utilization: M={M} N={N} K={K} T_m={T_m} T_n={T_n} num_sms={num_sms} ==="
    )
    print(f"{'s':>4} | {'blocks':>8} | {'hw_waves':>9} | {'eta':>7} | {'rho_wave':>8}")
    print("-" * 54)
    for s in (1, 2, 4, 8, 16, 32, 64, 128):
        if s * 32 > K:
            break
        total = T_m * T_n * s
        hw_w = math.ceil(total / num_sms)
        eta = total / (hw_w * num_sms)
        marker = " <- near-full" if abs(eta - 1.0) < 0.05 else ""
        print(f"{s:>4} | {total:>8} | {hw_w:>9} | {eta:>7.1%} | {1-eta:>8.1%}{marker}")
