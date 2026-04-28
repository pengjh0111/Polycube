"""JSON report helpers for wave-tiling sweeps."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

from .benefit_model import HardwareParams
from .search import TilingResult, search_optimal_wave_shape


@dataclass(frozen=True)
class CaseReport:
    M: int
    N: int
    K: int
    frontier: list[dict]
    best_config: dict | None
    theoretical_wave_count: int | None = None
    actual_wave_count: int | None = None
    latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    speedup: float | None = None


def generate_sweep_report(
    cases: list[tuple[int, int, int]],
    hw: HardwareParams,
    tile_m_candidates: list[int],
    tile_n_candidates: list[int],
    splitk_candidates: list[int],
    benchmark_fn: Callable[[int, int, int, TilingResult, int, int], float] | None = None,
    baseline_fn: Callable[[int, int, int, int, int], float] | None = None,
    shape_type: str = "cuboid",
    max_shapes: int = 500,
) -> dict:
    """Build a JSON-serializable report for a problem sweep."""

    reports: list[dict] = []
    for M, N, K in cases:
        frontier = search_optimal_wave_shape(
            M=M,
            N=N,
            K=K,
            tile_m_candidates=tile_m_candidates,
            tile_n_candidates=tile_n_candidates,
            splitk_candidates=splitk_candidates,
            hw=hw,
            shape_type=shape_type,
            max_shapes=max_shapes,
        )
        best = frontier[0] if frontier else None
        latency_ms = None
        baseline_latency_ms = None
        speedup = None
        actual_wave_count = None
        if best is not None and benchmark_fn is not None and baseline_fn is not None:
            latency_ms = benchmark_fn(M, N, K, best, 10, 100)
            baseline_latency_ms = baseline_fn(M, N, K, 10, 100)
            speedup = baseline_latency_ms / latency_ms if latency_ms > 0 else None
            actual_wave_count = best.wave_count
        reports.append(
            asdict(
                CaseReport(
                    M=M,
                    N=N,
                    K=K,
                    frontier=[item.to_dict() for item in frontier],
                    best_config=best.to_dict() if best is not None else None,
                    theoretical_wave_count=best.wave_count if best is not None else None,
                    actual_wave_count=actual_wave_count,
                    latency_ms=latency_ms,
                    baseline_latency_ms=baseline_latency_ms,
                    speedup=speedup,
                )
            )
        )
    return {
        "hardware": asdict(hw),
        "cases": reports,
    }