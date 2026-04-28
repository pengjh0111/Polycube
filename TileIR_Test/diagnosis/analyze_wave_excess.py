from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def main() -> None:
    results = json.loads(open("benchmark_results.json", "r", encoding="utf-8").read())

    print("=== Wave Excess Distribution ===")
    excesses = []
    for r in results:
        lb = r["lower_bound_waves"]
        act = r["predicted_waves"]
        excess = (act - lb) / lb if lb > 0 else 0.0
        excesses.append(excess)
        if excess > 1.5:
            print(
                f"  SEVERE: M={r['M']} N={r['N']} K={r['K']} "
                f"sk={r['splitk']} shape={tuple(r['wave_shape'])} "
                f"waves={act} lb={lb} excess={excess:.1%}"
            )

    arr = np.asarray(excesses, dtype=np.float64)
    print(f"\nMean excess: {arr.mean():.2%}")
    print(f"Median excess: {np.median(arr):.2%}")
    print(f"% cases excess > 50%: {np.mean(arr > 0.5):.1%}")
    print(f"% cases excess > 100%: {np.mean(arr > 1.0):.1%}")

    by_shape: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for r, e in zip(results, excesses):
        by_shape[tuple(r["wave_shape"])].append(e)

    print("\n=== Excess by wave_shape (sk, m, n) ===")
    for shape, es in sorted(by_shape.items(), key=lambda item: np.mean(item[1])):
        es_arr = np.asarray(es)
        print(
            f"  {shape}: mean={es_arr.mean():.1%} "
            f"min={es_arr.min():.1%} max={es_arr.max():.1%}"
        )


if __name__ == "__main__":
    main()
