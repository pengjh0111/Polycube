"""Layer 1 tile-size filter for unified three-layer search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class TileConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    rho_tile_m: float
    rho_tile_n: float
    rho_tile: float
    ai_estimate: float

    def is_valid(self, smem_bytes: int, dtype_bytes: int = 2) -> bool:
        acc_bytes = self.tile_m * self.tile_n * 4
        a_bytes = self.tile_m * self.tile_k * dtype_bytes
        b_bytes = self.tile_k * self.tile_n * dtype_bytes
        return (acc_bytes + a_bytes + b_bytes) <= smem_bytes


def tile_quantization_loss(dim: int, tile: int) -> float:
    """Fraction of tile compute wasted on padding.

    Case A (dim >= tile): padding appears at the tail boundary.
    Case B (dim < tile): one tile still runs, and (tile-dim)/tile work is wasted.
    """
    if dim >= tile:
        padded = math.ceil(dim / tile) * tile
        return 1.0 - dim / padded
    return (tile - dim) / tile


def arithmetic_intensity(tile_m: int, tile_n: int, tile_k: int, dtype_bytes: int = 2) -> float:
    flops = 2 * tile_m * tile_n * tile_k
    bytes_loaded = (tile_m * tile_k + tile_k * tile_n) * dtype_bytes
    return flops / bytes_loaded


def filter_tile_candidates(
    M: int,
    N: int,
    K: int,
    smem_bytes: int,
    tc_alignment: int = 16,
    rho_threshold: float = 0.15,
    dtype_bytes: int = 2,
    tile_m_options: tuple = (16, 32, 64, 128),
    tile_n_options: tuple = (32, 64, 128, 256),
    tile_k_options: tuple = (16, 32, 64),
) -> List[TileConfig]:
    candidates: List[TileConfig] = []
    all_computed: List[TileConfig] = []

    for t_m in tile_m_options:
        if t_m % tc_alignment != 0:
            continue
        if t_m > max(M * 4, tc_alignment):
            continue

        for t_n in tile_n_options:
            if t_n % tc_alignment != 0:
                continue

            for t_k in tile_k_options:
                if t_k % tc_alignment != 0:
                    continue
                if t_k > K:
                    continue

                rho_m = tile_quantization_loss(M, t_m)
                rho_n = tile_quantization_loss(N, t_n)
                rho = 1 - (1 - rho_m) * (1 - rho_n)

                ai = arithmetic_intensity(t_m, t_n, t_k, dtype_bytes)
                cfg = TileConfig(
                    tile_m=t_m,
                    tile_n=t_n,
                    tile_k=t_k,
                    rho_tile_m=rho_m,
                    rho_tile_n=rho_n,
                    rho_tile=rho,
                    ai_estimate=ai,
                )
                if not cfg.is_valid(smem_bytes, dtype_bytes):
                    continue
                all_computed.append(cfg)

                if rho > rho_threshold and rho > 0.35:
                    continue

                candidates.append(cfg)

    if not candidates and all_computed:
        # Small-M decode shapes can exceed strict rho thresholds; keep best options.
        candidates = sorted(all_computed, key=lambda c: (c.rho_tile, -c.ai_estimate))[:5]

    candidates.sort(key=lambda c: (c.rho_tile, -c.ai_estimate))
    return candidates


def explain_tile_filter(M: int, N: int, K: int, smem_bytes: int) -> None:
    print(f"\n=== Tile Filter Analysis: M={M} N={N} K={K} ===")
    print(
        f"{'t_m':>5} {'t_n':>5} {'t_k':>5} | {'rho_m':>7} {'rho_n':>7} {'rho':>7} | "
        f"{'AI':>8} | {'SMEM':>6} | {'status':>10}"
    )
    print("-" * 78)

    for t_m in (16, 32, 64):
        for t_n in (32, 64, 128, 256):
            for t_k in (16, 32, 64):
                if t_k > K:
                    continue
                rho_m = tile_quantization_loss(M, t_m)
                rho_n = tile_quantization_loss(N, t_n)
                rho = 1 - (1 - rho_m) * (1 - rho_n)
                ai = arithmetic_intensity(t_m, t_n, t_k)
                smem_ok = TileConfig(t_m, t_n, t_k, rho_m, rho_n, rho, ai).is_valid(smem_bytes)
                status = "PASS" if (rho <= 0.15 and smem_ok) else ("rho>15%" if rho > 0.15 else "SMEM")
                print(
                    f"{t_m:>5} {t_n:>5} {t_k:>5} | {rho_m:>7.1%} {rho_n:>7.1%} {rho:>7.1%} | "
                    f"{ai:>8.1f} | {'OK' if smem_ok else 'FAIL':>6} | {status:>10}"
                )
