from __future__ import annotations

import os
import sys

import cupy as cp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import TaskSpace3D


SHAPES = [
    (16, 4096, 8192),
    (32, 4096, 8192),
    (64, 4096, 8192),
    (32, 8192, 8192),
    (128, 4096, 4096),
    (32, 4096, 16384),
]
TILES = [(16, 64), (32, 128)]
SPLITKS = [1, 2, 4, 8]


def main() -> None:
    sm_count = cp.cuda.Device(0).attributes["MultiProcessorCount"]
    print(f"SM_COUNT={sm_count}")
    for M, N, K in SHAPES:
        for tm, tn in TILES:
            for sk in SPLITKS:
                task = TaskSpace3D.from_problem(M, N, K, tm, tn, sk)
                total = task.total_blocks()
                lb = task.lower_bound_waves(sm_count)
                good_shapes = [
                    (a, b, c)
                    for a in range(1, task.sk_dim + 1)
                    if task.sk_dim % a == 0
                    for b in range(1, task.tm_dim + 1)
                    if task.tm_dim % b == 0
                    for c in range(1, task.tn_dim + 1)
                    if task.tn_dim % c == 0 and a * b * c <= sm_count
                ]
                print(
                    f"M={M} N={N} K={K} tm={tm} tn={tn} sk={sk}: "
                    f"space=({task.sk_dim},{task.tm_dim},{task.tn_dim}) "
                    f"total={total} lb={lb} perfect_fit_shapes={len(good_shapes)}"
                )


if __name__ == "__main__":
    main()
