"""Benefit model for wave-shape ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .wave_shape import CuboidWaveShape, PolycubeWaveShape


@dataclass(frozen=True)
class HardwareParams:
    """Calibrate these per GPU target."""

    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 2.0
    delta: float = 0.0
    l2_cache_bytes: int = 50_331_648
    sm_count: int = 132
    shared_mem_bytes: int = 232_448


def compute_benefit(
    shape: CuboidWaveShape | PolycubeWaveShape,
    hw: HardwareParams,
    wave_size_override: int | None = None,
) -> float:
    """Higher is better."""

    proj = shape.projections()
    wave_size = wave_size_override if wave_size_override is not None else shape.size()
    return (
        hw.alpha * proj["k"]
        + hw.beta * (wave_size / proj["m"])
        + hw.gamma * (wave_size / proj["n"])
    )


def pareto_frontier(
    candidates: list[tuple[int, float, Any]],
) -> list[tuple[int, float, Any]]:
    """Return the Pareto-optimal set minimizing wave_count and maximizing benefit."""

    frontier: list[tuple[int, float, Any]] = []
    best_benefit = float("-inf")
    for waves, benefit, payload in sorted(candidates, key=lambda item: (item[0], -item[1])):
        if benefit <= best_benefit:
            continue
        frontier.append((waves, benefit, payload))
        best_benefit = benefit
    return frontier