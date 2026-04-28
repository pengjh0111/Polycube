"""Validation and lightweight benchmarking harness for wave tiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ir_emitter import wave_shape_to_block_mapping
from .search import TilingResult, compute_wave_count
from .task_space import TaskSpace3D
from .wave_shape import CuboidWaveShape


@dataclass(frozen=True)
class BenchmarkResult:
    latency_ms: float
    wave_count: int
    baseline_latency_ms: float
    speedup: float
    l2_hit_rate: float | None = None


def verify_coverage(task: TaskSpace3D, shape: CuboidWaveShape) -> bool:
    """Check that the translated wave lattice covers the entire task space."""

    covered = set()
    total_launch = compute_wave_count(task, shape) * shape.size()
    for block_id in range(total_launch):
        sk, m, n = wave_shape_to_block_mapping(task, shape, block_id)
        if 0 <= sk < task.sk_dim and 0 <= m < task.tm_dim and 0 <= n < task.tn_dim:
            covered.add((sk, m, n))
    expected = {
        (sk, m, n)
        for sk in range(task.sk_dim)
        for m in range(task.tm_dim)
        for n in range(task.tn_dim)
    }
    return covered == expected


def benchmark_config(
    M: int,
    N: int,
    K: int,
    result: TilingResult,
    num_warmup: int = 10,
    num_iters: int = 100,
    benchmark_fn: Callable[[int, int, int, TilingResult, int, int], float] | None = None,
    baseline_fn: Callable[[int, int, int, int, int], float] | None = None,
) -> BenchmarkResult:
    """Measure latency for a TilingResult using caller-provided backends."""

    if benchmark_fn is None or baseline_fn is None:
        raise RuntimeError("benchmark_config requires benchmark_fn and baseline_fn backends")

    latency_ms = benchmark_fn(M, N, K, result, num_warmup, num_iters)
    baseline_ms = baseline_fn(M, N, K, num_warmup, num_iters)
    speedup = baseline_ms / latency_ms if latency_ms > 0 else float("inf")
    return BenchmarkResult(
        latency_ms=latency_ms,
        wave_count=result.wave_count,
        baseline_latency_ms=baseline_ms,
        speedup=speedup,
    )


def hypothesis_test(
    M: int,
    N: int,
    K: int,
    hw,
    benchmark_fn: Callable[[int, int, int, TilingResult, int, int], float] | None = None,
    baseline_fn: Callable[[int, int, int, int, int], float] | None = None,
) -> dict[str, float]:
    """Compare sk-only, swizzle-only, and joint search strategies."""

    if benchmark_fn is None or baseline_fn is None:
        raise RuntimeError("hypothesis_test requires benchmark_fn and baseline_fn backends")

    from .benefit_model import HardwareParams
    from .search import search_optimal_wave_shape

    if not isinstance(hw, HardwareParams):
        raise TypeError("hw must be a HardwareParams instance")

    flat_shape = CuboidWaveShape(sk=1, m=1, n=hw.sm_count)
    flat_result = TilingResult(
        tile_m=32,
        tile_n=32,
        splitk_factor=1,
        wave_shape=flat_shape,
        wave_count=1,
        benefit=0.0,
    )
    sk_only = benchmark_fn(M, N, K, flat_result, 10, 100)
    sw_only = benchmark_fn(M, N, K, flat_result, 10, 100)
    frontier = search_optimal_wave_shape(
        M=M,
        N=N,
        K=K,
        tile_m_candidates=[32],
        tile_n_candidates=[32],
        splitk_candidates=[1],
        hw=hw,
    )
    joint = min(benchmark_fn(M, N, K, item, 10, 100) for item in frontier)
    return {
        "sk_only_ms": sk_only,
        "swizzle_only_ms": sw_only,
        "joint_ms": joint,
        "joint_over_best_single": max(sk_only, sw_only) / joint if joint > 0 else float("inf"),
    }