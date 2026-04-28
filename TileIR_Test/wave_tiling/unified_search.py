"""Three-layer unified search engine over tile, split-k, and wave shape."""

from __future__ import annotations

from typing import List

from .benefit_model import HardwareParams
from .search_v2 import search_v2
from .splitk_selector import select_splitk_candidates
from .task_space import TaskSpace3D
from .tile_filter import filter_tile_candidates
from .unified_score import UnifiedConfig, UnifiedWeights, compute_unified_score


def unified_search(
    M: int,
    N: int,
    K: int,
    num_sms: int,
    smem_bytes: int,
    hw: HardwareParams,
    weights: UnifiedWeights,
    rho_tile_threshold: float = 0.15,
    tile_m_options: tuple = (16, 32, 64, 128),
    tile_n_options: tuple = (32, 64, 128, 256),
    tile_k_options: tuple = (16, 32, 64),
    eta_min: float = 0.80,
    splitk_options: tuple = (1, 2, 4, 8, 16, 32, 64),
    min_hw_waves: int = 3,
    max_hw_waves: int = 15,
    max_wave_shapes: int = 100,
    top_k_per_splitk: int = 3,
    top_k_final: int = 20,
) -> List[UnifiedConfig]:
    tile_candidates = filter_tile_candidates(
        M,
        N,
        K,
        smem_bytes=smem_bytes,
        rho_threshold=rho_tile_threshold,
        tile_m_options=tile_m_options,
        tile_n_options=tile_n_options,
        tile_k_options=tile_k_options,
    )

    if not tile_candidates:
        tile_candidates = filter_tile_candidates(
            M,
            N,
            K,
            smem_bytes=smem_bytes,
            rho_threshold=0.35,
            tile_m_options=tile_m_options,
            tile_n_options=tile_n_options,
            tile_k_options=tile_k_options,
        )

    all_configs: List[UnifiedConfig] = []
    for tile in tile_candidates:
        splitk_candidates = select_splitk_candidates(
            tile.tile_m,
            tile.tile_n,
            tile.tile_k,
            M,
            N,
            K,
            num_sms=num_sms,
            c_atomic_normalized=weights.c_atomic,
            eta_min=eta_min,
            splitk_options=splitk_options,
        )

        in_regime = [c for c in splitk_candidates if min_hw_waves <= c.hw_waves <= max_hw_waves]
        if in_regime:
            splitk_candidates = in_regime

        for sk_cfg in splitk_candidates:
            s = sk_cfg.splitk
            task = TaskSpace3D.from_problem(M, N, K, tile.tile_m, tile.tile_n, s)
            wave_candidates = search_v2(
                task=task,
                num_sms=num_sms,
                hw=hw,
                allow_partial=True,
                max_shapes=max_wave_shapes,
            )

            for wc, benefit, shape in wave_candidates[:top_k_per_splitk]:
                cfg = UnifiedConfig(
                    M=M,
                    N=N,
                    K=K,
                    tile_m=tile.tile_m,
                    tile_n=tile.tile_n,
                    tile_k=tile.tile_k,
                    splitk=s,
                    ws_k=shape.sk,
                    ws_m=shape.m,
                    ws_n=shape.n,
                    rho_tile=tile.rho_tile,
                    rho_wave=sk_cfg.rho_wave,
                    benefit=benefit,
                    hw_waves=sk_cfg.hw_waves,
                    total_blocks=sk_cfg.total_blocks,
                    ai_estimate=tile.ai_estimate,
                )
                cfg.score = compute_unified_score(cfg, weights)
                all_configs.append(cfg)

    all_configs.sort(key=lambda c: -c.score)
    return all_configs[:top_k_final]


def explain_unified_search(
    M: int,
    N: int,
    K: int,
    num_sms: int,
    smem_bytes: int,
    hw: HardwareParams,
    weights: UnifiedWeights,
):
    results = unified_search(
        M,
        N,
        K,
        num_sms,
        smem_bytes,
        hw,
        weights,
        top_k_final=10,
    )

    print("\n" + "=" * 96)
    print(f"Unified Search: M={M} N={N} K={K}")
    print("=" * 96)
    print(
        f"{'rank':>4} {'tm':>4} {'tn':>4} {'tk':>4} {'sk':>4} | {'ws':>13} | "
        f"{'rho_tile':>8} {'rho_wave':>8} {'benefit':>9} | {'hw_w':>5} {'score':>9}"
    )
    print("-" * 96)

    for i, cfg in enumerate(results, start=1):
        ws = f"({cfg.ws_k},{cfg.ws_m},{cfg.ws_n})"
        print(
            f"{i:>4} {cfg.tile_m:>4} {cfg.tile_n:>4} {cfg.tile_k:>4} {cfg.splitk:>4} | "
            f"{ws:>13} | {cfg.rho_tile:>8.1%} {cfg.rho_wave:>8.1%} {cfg.benefit:>9.4f} | "
            f"{cfg.hw_waves:>5} {cfg.score:>9.4f}"
        )

    return results
