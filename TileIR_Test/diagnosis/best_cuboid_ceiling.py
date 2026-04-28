from __future__ import annotations

import os
import sys

import cupy as cp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import TaskSpace3D, compute_wave_count, enumerate_cuboid_shapes


CASES = [(32, 4096, 8192), (64, 4096, 8192)]
TILES = [(16, 64), (32, 128)]
SPLITKS = [1, 2, 4, 8]


def main() -> None:
    sm_count = cp.cuda.Device(0).attributes["MultiProcessorCount"]
    shapes = enumerate_cuboid_shapes(sm_count)
    for M, N, K in CASES:
        for tm, tn in TILES:
            for sk in SPLITKS:
                task = TaskSpace3D.from_problem(M, N, K, tm, tn, sk)
                lb = task.lower_bound_waves(sm_count)
                best_wc = min(compute_wave_count(task, s) for s in shapes)
                print(
                    f"M={M} N={N} K={K} tm={tm} tn={tn} sk={sk}: "
                    f"lb={lb} best_cuboid_wc={best_wc} "
                    f"best_cuboid_excess={(best_wc - lb) / lb:.1%}"
                )


if __name__ == "__main__":
    main()
