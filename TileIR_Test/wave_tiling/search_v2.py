"""Improved cuboid search prioritizing perfect-fit wave shapes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .benefit_model import HardwareParams, compute_benefit, pareto_frontier
from .search import TilingResult, compute_wave_count
from .task_space import TaskSpace3D
from .wave_shape import CuboidWaveShape


def _candidate_shapes(task: TaskSpace3D, num_sms: int) -> Iterable[CuboidWaveShape]:
    # ws_k is constrained to task.sk_dim; larger k-extent only creates empty slots.
    for sk in range(1, task.sk_dim + 1):
        for m in range(1, task.tm_dim + 1):
            for n in range(1, task.tn_dim + 1):
                size = sk * m * n
                if size <= 0 or size > num_sms:
                    continue
                yield CuboidWaveShape(sk=sk, m=m, n=n)


def search_v2(
    task: TaskSpace3D,
    num_sms: int,
    hw: HardwareParams,
    allow_partial: bool = True,
    wave_penalty: float = 0.0,
    max_shapes: int = 500,
) -> list[tuple[int, float, CuboidWaveShape]]:
    """Search cuboids with perfect-fit-first ordering.

    Perfect-fit = task dims divisible by shape dims and shape size divides SM count.
    """

    perfect: list[tuple[int, float, CuboidWaveShape, int]] = []
    near: list[tuple[int, float, CuboidWaveShape, int]] = []

    for shape in _candidate_shapes(task, num_sms):
        size = shape.size()
        if not allow_partial and num_sms % size != 0:
            continue
        wc = compute_wave_count(task, shape)
        bf = compute_benefit(shape, hw) - wave_penalty * wc
        pad = wc * size - task.total_blocks()
        entry = (wc, bf, shape, pad)
        if (
            task.sk_dim % shape.sk == 0
            and task.tm_dim % shape.m == 0
            and task.tn_dim % shape.n == 0
            and num_sms % size == 0
        ):
            perfect.append(entry)
        else:
            near.append(entry)

    perfect.sort(key=lambda item: (item[0], -item[1]))
    near.sort(key=lambda item: (item[0], item[3], -item[1]))
    combined = perfect + near
    return [(wc, bf, shape) for wc, bf, shape, _ in combined[:max_shapes]]


def search_v2_joint(
    M: int,
    N: int,
    K: int,
    tile_m_candidates: list[int],
    tile_n_candidates: list[int],
    splitk_candidates: list[int],
    hw: HardwareParams,
    num_sms: int,
    max_shapes: int = 500,
    allow_partial: bool = True,
    wave_penalty: float = 0.0,
) -> list[TilingResult]:
    """Joint search over tiling config and perfect-fit-first cuboid shapes."""

    all_results: list[TilingResult] = []
    for tile_m in tile_m_candidates:
        for tile_n in tile_n_candidates:
            for splitk in splitk_candidates:
                task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
                for waves, benefit, shape in search_v2(
                    task,
                    num_sms=num_sms,
                    hw=hw,
                    allow_partial=allow_partial,
                    wave_penalty=wave_penalty,
                    max_shapes=max_shapes,
                ):
                    all_results.append(
                        TilingResult(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            splitk_factor=splitk,
                            wave_shape=shape,
                            wave_count=waves,
                            benefit=benefit,
                        )
                    )

    frontier = pareto_frontier([(r.wave_count, r.benefit, r) for r in all_results])
    return [item[2] for item in sorted(frontier, key=lambda item: (item[0], -item[1]))]


def explain_wave_excess(task: TaskSpace3D, shape: CuboidWaveShape, num_sms: int) -> None:
    """Decompose sources of wave excess for a specific task/shape pair."""

    wc_k = math.ceil(task.sk_dim / shape.sk)
    wc_m = math.ceil(task.tm_dim / shape.m)
    wc_n = math.ceil(task.tn_dim / shape.n)
    total_wc = wc_k * wc_m * wc_n
    lb = task.lower_bound_waves(num_sms)

    pad_k = (wc_k * shape.sk - task.sk_dim) / max(task.sk_dim, 1)
    pad_m = (wc_m * shape.m - task.tm_dim) / max(task.tm_dim, 1)
    pad_n = (wc_n * shape.n - task.tn_dim) / max(task.tn_dim, 1)

    print(f"Task: ({task.sk_dim}, {task.tm_dim}, {task.tn_dim})")
    print(f"Shape: ({shape.sk}, {shape.m}, {shape.n}) size={shape.size()}")
    print(
        f"Waves: k={wc_k}(pad={pad_k:.0%}) "
        f"m={wc_m}(pad={pad_m:.0%}) n={wc_n}(pad={pad_n:.0%})"
    )
    print(f"Total waves={total_wc} vs lb={lb} excess={(total_wc - lb) / lb:.1%}")
    print(f"SM utilization per wave: {task.total_blocks() / (total_wc * num_sms):.1%}")
