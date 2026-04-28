"""Find candidate configs that run in the true multi-wave regime."""

from __future__ import annotations

import json
import math
from pathlib import Path


NUM_SMS = 170
TARGET_HW_WAVES = 5

SHAPES_EXTENDED = [
    (16, 4096, 8192),
    (32, 4096, 8192),
    (64, 4096, 8192),
    (128, 4096, 8192),
    (32, 8192, 8192),
    (128, 4096, 4096),
    (32, 4096, 16384),
    (256, 4096, 8192),
    (512, 4096, 8192),
    (1024, 4096, 8192),
    (256, 8192, 8192),
]

TILE_CONFIGS = [(16, 64), (32, 128)]
SPLITKS = [1, 2, 4, 8, 16, 32]


def _bucket(hw_waves: int) -> str:
    if hw_waves < 5:
        return "1-4"
    if hw_waves < 11:
        return "5-10"
    if hw_waves < 21:
        return "11-20"
    if hw_waves < 51:
        return "21-50"
    return "50+"


def main() -> None:
    print(f"{'M':>5} {'N':>5} {'K':>6} {'tm':>4} {'tn':>4} {'sk':>4} | {'blocks':>7} {'hw_waves':>9} {'regime':>12}")
    print("-" * 72)

    multiwave_configs: list[dict] = []
    buckets: dict[str, int] = {"5-10": 0, "11-20": 0, "21-50": 0, "50+": 0}

    for M, N, K in SHAPES_EXTENDED:
        for tile_m, tile_n in TILE_CONFIGS:
            for splitk in SPLITKS:
                tm = math.ceil(M / tile_m)
                tn = math.ceil(N / tile_n)
                total_blocks = tm * tn * splitk
                hw_waves = math.ceil(total_blocks / NUM_SMS)

                regime = "MULTI" if hw_waves >= TARGET_HW_WAVES else ("DUAL" if hw_waves >= 2 else "SINGLE")
                if regime != "MULTI":
                    continue

                bucket = _bucket(hw_waves)
                row = {
                    "M": M,
                    "N": N,
                    "K": K,
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": 32,
                    "splitk": splitk,
                    "total_blocks": total_blocks,
                    "hw_waves": hw_waves,
                    "bucket": bucket,
                }
                multiwave_configs.append(row)
                if bucket in buckets:
                    buckets[bucket] += 1
                print(
                    f"{M:>5} {N:>5} {K:>6} {tile_m:>4} {tile_n:>4} {splitk:>4} | "
                    f"{total_blocks:>7} {hw_waves:>9} {regime:>12}"
                )

    print(f"\nTotal multi-wave configs: {len(multiwave_configs)}")
    print("Buckets:", buckets)

    out_path = Path("results/multiwave_candidates.json")
    out_path.write_text(json.dumps({"num_sms": NUM_SMS, "target_hw_waves": TARGET_HW_WAVES, "buckets": buckets, "configs": multiwave_configs}, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
