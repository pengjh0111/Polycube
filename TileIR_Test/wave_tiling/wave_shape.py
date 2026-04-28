"""Wave-shape abstractions and enumeration helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable


def _normalize_points(points: Iterable[tuple[int, int, int]]) -> frozenset[tuple[int, int, int]]:
    pts = list(points)
    if not pts:
        return frozenset()
    min_k = min(point[0] for point in pts)
    min_m = min(point[1] for point in pts)
    min_n = min(point[2] for point in pts)
    return frozenset((k - min_k, m - min_m, n - min_n) for k, m, n in pts)


def _rotation_matrices() -> list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
    mats = []
    seen = set()
    axes = (0, 1, 2)
    for perm in permutations(axes):
        for signs in product((-1, 1), repeat=3):
            matrix = [[0, 0, 0] for _ in range(3)]
            for row, axis in enumerate(perm):
                matrix[row][axis] = signs[row]
            det = (
                matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
            )
            if det != 1:
                continue
            key = tuple(tuple(row) for row in matrix)
            if key not in seen:
                seen.add(key)
                mats.append(key)
    return mats


_ROTATIONS = _rotation_matrices()


def _apply_rotation(
    point: tuple[int, int, int],
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> tuple[int, int, int]:
    return tuple(sum(matrix[row][col] * point[col] for col in range(3)) for row in range(3))


def _canonical_points(points: Iterable[tuple[int, int, int]]) -> frozenset[tuple[int, int, int]]:
    pts = list(points)
    if not pts:
        return frozenset()
    candidates = []
    for matrix in _ROTATIONS:
        rotated = [_apply_rotation(point, matrix) for point in pts]
        candidates.append(tuple(sorted(_normalize_points(rotated))))
    best = min(candidates)
    return frozenset(best)


def _neighbors(point: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    k, m, n = point
    return {
        (k + 1, m, n),
        (k - 1, m, n),
        (k, m + 1, n),
        (k, m - 1, n),
        (k, m, n + 1),
        (k, m, n - 1),
    }


@dataclass(frozen=True)
class CuboidWaveShape:
    """A rectangular box of SMs in (sk, m, n) space."""

    sk: int
    m: int
    n: int

    def size(self) -> int:
        return self.sk * self.m * self.n

    def point_set(self) -> set[tuple[int, int, int]]:
        return {
            (k, i, j)
            for k in range(self.sk)
            for i in range(self.m)
            for j in range(self.n)
        }

    def projections(self) -> dict[str, int]:
        return {"k": self.sk, "m": self.m, "n": self.n}


def enumerate_cuboid_shapes(num_sms: int) -> list[CuboidWaveShape]:
    """Return all exact cuboids whose volume is ``num_sms``."""

    shapes: list[CuboidWaveShape] = []
    for sk in range(1, num_sms + 1):
        if num_sms % sk != 0:
            continue
        remainder = num_sms // sk
        for m in range(1, remainder + 1):
            if remainder % m != 0:
                continue
            n = remainder // m
            shapes.append(CuboidWaveShape(sk=sk, m=m, n=n))
    return shapes


@dataclass(frozen=True)
class PolycubeWaveShape:
    """An arbitrary W-point subset of Z^3 (a connected polycube)."""

    points: frozenset[tuple[int, int, int]]

    def size(self) -> int:
        return len(self.points)

    def bounding_box(self) -> tuple[int, int, int]:
        if not self.points:
            return (0, 0, 0)
        ks = [point[0] for point in self.points]
        ms = [point[1] for point in self.points]
        ns = [point[2] for point in self.points]
        return (max(ks) - min(ks) + 1, max(ms) - min(ms) + 1, max(ns) - min(ns) + 1)

    def projections(self) -> dict[str, int]:
        return {
            "k": len({point[0] for point in self.points}),
            "m": len({point[1] for point in self.points}),
            "n": len({point[2] for point in self.points}),
        }

    @classmethod
    def from_cuboid(cls, c: CuboidWaveShape) -> "PolycubeWaveShape":
        return cls(frozenset(c.point_set()))


def enumerate_polycubes(W: int, max_count: int = 1000) -> list[PolycubeWaveShape]:
    """Enumerate fixed polycubes of size W in Z^3.

    For W > 12, this switches to randomized connected sampling.
    """

    if W <= 0:
        return []
    if W == 1:
        return [PolycubeWaveShape(frozenset({(0, 0, 0)}))]

    if W > 4:
        rng = random.Random(W)
        shapes: list[PolycubeWaveShape] = []
        seen: set[frozenset[tuple[int, int, int]]] = set()
        attempts = 0
        while len(shapes) < max_count and attempts < max_count * 50:
            attempts += 1
            points = {(0, 0, 0)}
            frontier = set(_neighbors((0, 0, 0)))
            while len(points) < W and frontier:
                choice = rng.choice(tuple(frontier))
                frontier.remove(choice)
                if choice in points:
                    continue
                points.add(choice)
                frontier |= _neighbors(choice)
                frontier -= points
            if len(points) != W:
                continue
            canonical = _canonical_points(points)
            if canonical in seen:
                continue
            seen.add(canonical)
            shapes.append(PolycubeWaveShape(canonical))
        return shapes

    shapes: list[PolycubeWaveShape] = []
    seen: set[frozenset[tuple[int, int, int]]] = set()

    def recurse(points: set[tuple[int, int, int]], frontier: set[tuple[int, int, int]]) -> None:
        if len(shapes) >= max_count:
            return
        if len(points) == W:
            canonical = _canonical_points(points)
            if canonical not in seen:
                seen.add(canonical)
                shapes.append(PolycubeWaveShape(canonical))
            return
        for cell in sorted(frontier):
            next_points = set(points)
            next_points.add(cell)
            next_frontier = set(frontier)
            next_frontier.remove(cell)
            next_frontier |= _neighbors(cell)
            next_frontier -= next_points
            recurse(next_points, next_frontier)

    origin = (0, 0, 0)
    recurse({origin}, _neighbors(origin))
    return shapes