"""Classify fixed-splitk benchmark configs by wave-count reduction potential."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import cupy as cp

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import CuboidWaveShape, HardwareParams, TaskSpace3D, compute_wave_count
from wave_tiling.search_v2 import search_v2

PROMISING_THRESHOLD = 0.2

FIXED_CONFIGS = [
    # (tile_m, tile_n, tile_k, splitk)
    (16, 64, 32, 1),
    (16, 64, 32, 2),
    (16, 64, 32, 4),
    (16, 64, 32, 8),
    (32, 128, 32, 1),
    (32, 128, 32, 2),
    (32, 128, 32, 4),
    (32, 128, 32, 8),
]

BASE_SHAPES = [
    (16, 4096, 8192),
    (32, 4096, 8192),
    (64, 4096, 8192),
    (32, 8192, 8192),
    (128, 4096, 4096),
    (32, 4096, 16384),
]

EXTRA_SHAPES = [
    (256, 4096, 8192),
    (512, 4096, 8192),
]


def _load_calibrated_hw_params() -> HardwareParams:
    dev = cp.cuda.Device(0)
    sm_count = dev.attributes["MultiProcessorCount"]
    l2 = dev.attributes["L2CacheSize"]
    path = Path("calibrated_hw_params.json")
    if not path.exists():
        return HardwareParams(sm_count=sm_count, l2_cache_bytes=l2)
    data = json.loads(path.read_text())
    return HardwareParams(
        alpha=float(data.get("alpha", 1.0)),
        beta=float(data.get("beta", 2.0)),
        gamma=float(data.get("gamma", 2.0)),
        sm_count=sm_count,
        l2_cache_bytes=l2,
    )


def classify_configs(shapes: list[tuple[int, int, int]], num_sms: int, hw: HardwareParams) -> dict:
    hopeless: list[dict] = []
    marginal: list[dict] = []
    promising: list[dict] = []

    for M, N, K in shapes:
        for tile_m, tile_n, tile_k, splitk in FIXED_CONFIGS:
            task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
            flat_wc = compute_wave_count(task, CuboidWaveShape(sk=1, m=1, n=1))

            base_row = {
                "M": M,
                "N": N,
                "K": K,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "splitk": splitk,
                "flat_wc": flat_wc,
            }

            if flat_wc <= 2:
                hopeless.append(base_row | {"reason": "flat_wc<=2"})
                continue

            candidates = search_v2(task=task, num_sms=num_sms, hw=hw, allow_partial=True, max_shapes=200)
            if not candidates:
                marginal.append(base_row | {"best_wc": flat_wc, "reduction": 0.0, "reason": "no_candidates"})
                continue

            best_wc, best_benefit, best_shape = candidates[0]
            reduction = (flat_wc - best_wc) / flat_wc
            row = base_row | {
                "best_wc": int(best_wc),
                "reduction": float(reduction),
                "best_shape": [best_shape.sk, best_shape.m, best_shape.n],
                "best_benefit": float(best_benefit),
            }

            if reduction >= PROMISING_THRESHOLD:
                promising.append(row)
            else:
                marginal.append(row)

    return {
        "threshold": PROMISING_THRESHOLD,
        "num_sms": num_sms,
        "hopeless": hopeless,
        "marginal": marginal,
        "promising": promising,
    }


def find_promising_shapes(num_sms: int, hw: HardwareParams) -> dict:
    first_pass = classify_configs(BASE_SHAPES, num_sms, hw)
    if len(first_pass["promising"]) >= 5:
        return first_pass | {"used_shapes": BASE_SHAPES, "expanded_with_large_m": False}

    shapes = BASE_SHAPES + EXTRA_SHAPES
    second_pass = classify_configs(shapes, num_sms, hw)
    return second_pass | {"used_shapes": shapes, "expanded_with_large_m": True}


def main() -> None:
    hw = _load_calibrated_hw_params()
    num_sms = cp.cuda.Device(0).attributes["MultiProcessorCount"]
    out = find_promising_shapes(num_sms, hw)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "shape_classification.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=== Shape Classification ===")
    print(f"Hopeless:  {len(out['hopeless'])}")
    print(f"Marginal:  {len(out['marginal'])}")
    print(f"Promising: {len(out['promising'])}")
    print(f"Expanded with larger M shapes: {out['expanded_with_large_m']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()