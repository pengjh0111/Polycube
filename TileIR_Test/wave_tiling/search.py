"""Wave-shape search and coverage utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from .benefit_model import HardwareParams, compute_benefit, pareto_frontier
from .task_space import TaskSpace3D
from .wave_shape import CuboidWaveShape, PolycubeWaveShape, enumerate_cuboid_shapes, enumerate_polycubes


WAVE_COUNT_FAST_PATH_THRESHOLD = 0


@dataclass(frozen=True)
class TilingResult:
    tile_m: int
    tile_n: int
    splitk_factor: int
    wave_shape: CuboidWaveShape | PolycubeWaveShape
    wave_count: int
    benefit: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        shape = self.wave_shape
        if isinstance(shape, CuboidWaveShape):
            out["wave_shape"] = {"type": "cuboid", **asdict(shape)}
        else:
            out["wave_shape"] = {"type": "polycube", "points": sorted(shape.points)}
        return out


def compute_wave_count(task: TaskSpace3D, shape: CuboidWaveShape) -> int:
    """Exact cuboid wave count via ceiling division."""

    waves_k = math.ceil(task.sk_dim / shape.sk)
    waves_m = math.ceil(task.tm_dim / shape.m)
    waves_n = math.ceil(task.tn_dim / shape.n)
    return waves_k * waves_m * waves_n


def _translated_points(
    shape: PolycubeWaveShape,
    translation: tuple[int, int, int],
) -> set[tuple[int, int, int]]:
    tk, tm, tn = translation
    return {(k + tk, m + tm, n + tn) for k, m, n in shape.points}


def _in_bounds(task: TaskSpace3D, point: tuple[int, int, int]) -> bool:
    return (
        0 <= point[0] < task.sk_dim
        and 0 <= point[1] < task.tm_dim
        and 0 <= point[2] < task.tn_dim
    )


def wave_composite_coords(
    unit_coords: list[tuple[int, int, int]],
    task: TaskSpace3D,
    num_sm: int,
) -> set[tuple[int, int, int]]:
    """
    Return the union of coordinates covered by (num_sm // f) copies of a
    polycube unit tiled through task space in (sk, m, n) row-major order,
    wrapping via modulo on each dimension boundary.

    This matches the block layout convention in ir_emitter.py exactly.
    """

    f = len(unit_coords)
    copies = num_sm // f

    composite: set[tuple[int, int, int]] = set()
    total_mn = task.tm_dim * task.tn_dim

    for c in range(copies):
        # Decode flat offset c*f into (sk, m, n) using row-major layout.
        base = c * f
        sk_off = base // total_mn
        m_off = (base % total_mn) // task.tn_dim
        n_off = base % task.tn_dim

        for sk_u, tm_u, tn_u in unit_coords:
            composite.add(
                (
                    (sk_u + sk_off) % task.sk_dim,
                    (tm_u + m_off) % task.tm_dim,
                    (tn_u + n_off) % task.tn_dim,
                )
            )

    return composite


def scan_order_first_wave(
    task: TaskSpace3D,
    shape: PolycubeWaveShape,
    num_sm: int,
) -> set[tuple[int, int, int]]:
    """
    Return the composite coordinates of the first hardware wave under
    scan-order polycube placement.

    Iterates task space in (sk, m, n) row-major order. Each time an
    uncovered point is found, places the polycube unit anchored at that
    point (first sorted point = anchor), clipping out-of-bounds members.
    Stops after placing (num_sm // f) units.
    """

    unit_points = sorted(shape.points)
    anchor = unit_points[0]
    composite: set[tuple[int, int, int]] = set()
    emitted = 0
    total_mn = task.tm_dim * task.tn_dim

    for c in range(num_sm):
        sk_off = c // total_mn
        m_off = (c % total_mn) // task.tn_dim
        n_off = c % task.tn_dim

        for pk, pm, pn in unit_points:
            q = (
                pk - anchor[0] + sk_off,
                pm - anchor[1] + m_off,
                pn - anchor[2] + n_off,
            )
            if (
                0 <= q[0] < task.sk_dim
                and 0 <= q[1] < task.tm_dim
                and 0 <= q[2] < task.tn_dim
                and q not in composite
            ):
                composite.add(q)
                emitted += 1
                if emitted >= num_sm:
                    return composite

    return composite


def scan_order_full_cover(
    task: TaskSpace3D,
    shape: PolycubeWaveShape,
) -> list[tuple[int, int, int]]:
    """
    Cover the full task space using scan-order polycube placement.
    Returns ordered (sk, m, n) coordinates for execution LUT.
    """

    return list(_scan_order_emit(task, shape))


def _scan_order_emit(
    task: TaskSpace3D,
    shape: PolycubeWaveShape,
) -> list[tuple[int, int, int]]:
    """Emit ordered unique task points using scan-order anchored placement."""

    unit_points = sorted(shape.points)
    anchor = unit_points[0]

    covered: set[tuple[int, int, int]] = set()
    ordered: list[tuple[int, int, int]] = []

    for sk in range(task.sk_dim):
        for m in range(task.tm_dim):
            for n in range(task.tn_dim):
                if (sk, m, n) in covered:
                    continue

                tk = sk - anchor[0]
                tm_off = m - anchor[1]
                tn_off = n - anchor[2]

                for pk, pm, pn in unit_points:
                    q = (pk + tk, pm + tm_off, pn + tn_off)
                    if (
                        0 <= q[0] < task.sk_dim
                        and 0 <= q[1] < task.tm_dim
                        and 0 <= q[2] < task.tn_dim
                    ):
                        if q not in covered:
                            covered.add(q)
                            ordered.append(q)

    # Safety net: append any uncovered boundary fragments in row-major order.
    for sk in range(task.sk_dim):
        for m in range(task.tm_dim):
            for n in range(task.tn_dim):
                if (sk, m, n) not in covered:
                    ordered.append((sk, m, n))

    return ordered


def _candidate_translations(task: TaskSpace3D, shape: PolycubeWaveShape) -> set[tuple[int, int, int]]:
    translations: set[tuple[int, int, int]] = set()
    task_points = {
        (k, m, n)
        for k in range(task.sk_dim)
        for m in range(task.tm_dim)
        for n in range(task.tn_dim)
    }
    for task_point in task_points:
        for shape_point in shape.points:
            translation = (
                task_point[0] - shape_point[0],
                task_point[1] - shape_point[1],
                task_point[2] - shape_point[2],
            )
            translated = _translated_points(shape, translation)
            if all(_in_bounds(task, point) for point in translated):
                translations.add(translation)
    return translations


def compute_wave_count_polycube(
    task: TaskSpace3D,
    shape: PolycubeWaveShape,
    strategy: str = "greedy",
) -> tuple[int, list[tuple[int, int, int]]]:
    """Find wave count for a polycube wave shape.

    For large task spaces (total_blocks > WAVE_COUNT_FAST_PATH_THRESHOLD),
    returns an analytical estimate immediately to avoid O(total^2) behavior.
    For small task spaces, runs exact greedy or ILP covering.
    """

    if not shape.points:
        return 0, []

    total_blocks = task.total_blocks()
    shape_size = shape.size()

    # Fast path: analytical estimate for large task spaces.
    if total_blocks > WAVE_COUNT_FAST_PATH_THRESHOLD:
        lb = math.ceil(total_blocks / shape_size)
        bb_k, bb_m, bb_n = shape.bounding_box()
        wc_k = math.ceil(task.sk_dim / max(1, bb_k))
        wc_m = math.ceil(task.tm_dim / max(1, bb_m))
        wc_n = math.ceil(task.tn_dim / max(1, bb_n))
        wave_count = max(lb, wc_k * wc_m * wc_n)
        return wave_count, []

    if strategy not in {"greedy", "ilp"}:
        raise ValueError("strategy must be 'greedy' or 'ilp'")

    task_points = {
        (k, m, n)
        for k in range(task.sk_dim)
        for m in range(task.tm_dim)
        for n in range(task.tn_dim)
    }
    translations = sorted(_candidate_translations(task, shape))
    if not translations:
        return 0, []

    if strategy == "ilp":
        try:
            import pulp
        except Exception:
            strategy = "greedy"
        else:
            problem = pulp.LpProblem("wave_cover", pulp.LpMinimize)
            vars_by_translation = {
                translation: pulp.LpVariable(f"x_{idx}", cat="Binary")
                for idx, translation in enumerate(translations)
            }
            for task_point in task_points:
                covering = [
                    vars_by_translation[translation]
                    for translation in translations
                    if task_point in _translated_points(shape, translation)
                ]
                if covering:
                    problem += pulp.lpSum(covering) >= 1
            problem += pulp.lpSum(vars_by_translation.values())
            problem.solve(pulp.PULP_CBC_CMD(msg=False))
            chosen = [translation for translation, var in vars_by_translation.items() if var.value() and var.value() > 0.5]
            return len(chosen), chosen

    uncovered = set(task_points)
    chosen: list[tuple[int, int, int]] = []
    while uncovered:
        best_translation = None
        best_cover: set[tuple[int, int, int]] = set()
        for translation in translations:
            covered = _translated_points(shape, translation) & uncovered
            if len(covered) > len(best_cover):
                best_cover = covered
                best_translation = translation
        if best_translation is None or not best_cover:
            break
        chosen.append(best_translation)
        uncovered -= best_cover
    return len(chosen), chosen


@lru_cache(maxsize=64)
def _cached_shape_candidates_for_search(
    num_sms: int,
    shape_type: str,
    max_shapes: int,
) -> tuple[CuboidWaveShape | PolycubeWaveShape, ...]:
    if shape_type == "cuboid":
        return tuple(enumerate_cuboid_shapes(num_sms)[:max_shapes])
    if shape_type == "polycube":
        shapes: list[CuboidWaveShape | PolycubeWaveShape] = []
        divisors = [size for size in range(1, min(num_sms, 12) + 1) if num_sms % size == 0]
        for size in divisors:
            remaining = max_shapes - len(shapes)
            if remaining <= 0:
                break
            shapes.extend(enumerate_polycubes(size, max_count=remaining))
            if len(shapes) >= max_shapes:
                break
        return tuple(shapes[:max_shapes])
    raise ValueError("shape_type must be 'cuboid' or 'polycube'")


def _shape_candidates_for_search(num_sms: int, shape_type: str, max_shapes: int) -> list[CuboidWaveShape | PolycubeWaveShape]:
    return list(_cached_shape_candidates_for_search(num_sms, shape_type, max_shapes))


def _evaluate_shape_candidate(
    payload: tuple[TaskSpace3D, CuboidWaveShape | PolycubeWaveShape, HardwareParams, str, int, bool],
) -> tuple[int, float]:
    task, shape, hw, polycube_strategy, num_sm, use_wave_composite = payload
    if isinstance(shape, CuboidWaveShape):
        wave_count = compute_wave_count(task, shape)
        benefit = compute_benefit(shape, hw)
    else:
        del polycube_strategy
        wave_count = math.ceil(task.total_blocks() / shape.size())
        if use_wave_composite:
            composite = scan_order_first_wave(task, shape, num_sm)
            composite_shape = PolycubeWaveShape(frozenset(composite))
            benefit = compute_benefit(composite_shape, hw, wave_size_override=hw.sm_count)
        else:
            benefit = compute_benefit(shape, hw)
    return wave_count, benefit


def search_optimal_wave_shape(
    M: int,
    N: int,
    K: int,
    tile_m_candidates: list[int],
    tile_n_candidates: list[int],
    splitk_candidates: list[int],
    hw: HardwareParams,
    shape_type: str = "cuboid",
    max_shapes: int = 500,
    log_path: str | Path | None = None,
    num_workers: int = 1,
    polycube_strategy: str = "greedy",
    use_wave_composite: bool = True,
) -> list[TilingResult]:
    """Joint search over tile sizes, split-k factor, and wave shape."""

    results: list[TilingResult] = []
    search_log: list[dict[str, Any]] = []
    shapes = _shape_candidates_for_search(hw.sm_count, shape_type, max_shapes)

    executor = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    try:
        for tile_m in tile_m_candidates:
            for tile_n in tile_n_candidates:
                for splitk in splitk_candidates:
                    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)

                    if executor is None:
                        evaluated = [
                            _evaluate_shape_candidate((task, shape, hw, polycube_strategy, hw.sm_count, use_wave_composite))
                            for shape in shapes
                        ]
                    else:
                        payloads = (
                            (task, shape, hw, polycube_strategy, hw.sm_count, use_wave_composite)
                            for shape in shapes
                        )
                        evaluated = list(executor.map(_evaluate_shape_candidate, payloads, chunksize=16))

                    for shape, (wave_count, benefit) in zip(shapes, evaluated):
                        result = TilingResult(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            splitk_factor=splitk,
                            wave_shape=shape,
                            wave_count=wave_count,
                            benefit=benefit,
                        )
                        results.append(result)
                        search_log.append({
                            "M": M,
                            "N": N,
                            "K": K,
                            "tile_m": tile_m,
                            "tile_n": tile_n,
                            "splitk_factor": splitk,
                            "task": {"sk_dim": task.sk_dim, "tm_dim": task.tm_dim, "tn_dim": task.tn_dim},
                            "wave_shape": result.to_dict()["wave_shape"],
                            "wave_count": wave_count,
                            "benefit": benefit,
                        })
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if log_path is not None:
        Path(log_path).write_text(json.dumps({"candidates": search_log}, indent=2))

    frontier = pareto_frontier([(r.wave_count, r.benefit, r) for r in results])
    return [item[2] for item in sorted(frontier, key=lambda item: (item[0], -item[1]))]