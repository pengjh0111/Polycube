"""Unified score function and calibration for three-layer search."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .benefit_model import HardwareParams, compute_benefit
from .splitk_selector import select_splitk_candidates
from .tile_filter import arithmetic_intensity, tile_quantization_loss
from .wave_shape import CuboidWaveShape


@dataclass
class UnifiedConfig:
    M: int
    N: int
    K: int
    tile_m: int
    tile_n: int
    tile_k: int
    splitk: int
    ws_k: int
    ws_m: int
    ws_n: int
    rho_tile: float
    rho_wave: float
    benefit: float
    hw_waves: int
    total_blocks: int
    ai_estimate: float
    score: float = 0.0


@dataclass
class UnifiedWeights:
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    lambda4: float = 1.0
    c_atomic: float = 0.05


def compute_unified_score(cfg: UnifiedConfig, w: UnifiedWeights) -> float:
    return (
        -w.lambda1 * cfg.rho_tile
        -w.lambda2 * cfg.rho_wave
        +w.lambda3 * cfg.benefit
        -w.lambda4 * (cfg.splitk - 1) * w.c_atomic
    )


def calibrate_unified_weights(
    benchmark_results_path: str = "results/fair_benchmark_results.json",
    hw_params_path: str = "calibrated_hw_params.json",
) -> UnifiedWeights:
    dataset_paths = [
        benchmark_results_path,
        "results/multiwave_benchmark.json",
        "results/dense_sweep_results.json",
    ]
    results_by_path = {}
    for path in dataset_paths:
        try:
            results_by_path[path] = json.loads(open(path).read())
        except Exception:
            continue

    if not results_by_path:
        raise FileNotFoundError("No benchmark datasets found for unified weight calibration")

    hw_raw = json.loads(open(hw_params_path).read())
    hw = HardwareParams(
        alpha=float(hw_raw.get("alpha", 1.0)),
        beta=float(hw_raw.get("beta", 2.0)),
        gamma=float(hw_raw.get("gamma", 2.0)),
        delta=float(hw_raw.get("delta", 0.0)),
        sm_count=int(hw_raw.get("sm_count", 170)) if "sm_count" in hw_raw else 170,
    )

    def _extract_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        latencies = []
        for r in rows:
            if not r.get("ms_wave_best"):
                continue
            if r.get("best_shape"):
                ws_k, ws_m, ws_n = r["best_shape"]
                benefit = float(r.get("best_benefit", 0.0))
            elif r.get("wave_shape"):
                ws_k, ws_m, ws_n = r["wave_shape"]
                benefit = float(r.get("predicted_benefit", r.get("benefit", 0.0)))
            else:
                continue

            M, N = int(r["M"]), int(r["N"])
            tm, tn = int(r["tile_m"]), int(r["tile_n"])
            sk = int(r.get("splitk", r.get("wave_splitk", 1)))

            rho_m = tile_quantization_loss(M, tm)
            rho_n = tile_quantization_loss(N, tn)
            rho_tile = 1 - (1 - rho_m) * (1 - rho_n)

            total = math.ceil(M / tm) * math.ceil(N / tn) * sk
            hw_waves = math.ceil(total / 170)
            eta = total / (hw_waves * 170)
            rho_wave = 1.0 - eta

            if benefit == 0.0:
                shape = CuboidWaveShape(sk=int(ws_k), m=int(ws_m), n=int(ws_n))
                benefit = compute_benefit(shape, hw)

            feats.append([rho_tile, rho_wave, benefit, float(sk - 1)])
            latencies.append(float(r["ms_wave_best"]))

        return np.asarray(feats, dtype=np.float64), np.asarray(latencies, dtype=np.float64)

    best_fit = None
    for path, rows in results_by_path.items():
        X, y = _extract_features(rows)
        if X.shape[0] < 4:
            continue

        F = np.column_stack([-X[:, 0], -X[:, 1], X[:, 2], -X[:, 3]])
        w_ls, *_ = np.linalg.lstsq(F, -y, rcond=None)
        w_abs = np.abs(w_ls)
        scores = F @ w_abs
        if scores.std() < 1e-12 or y.std() < 1e-12:
            continue
        corr = float(np.corrcoef(scores, y)[0, 1])
        entry = (abs(corr), corr, w_abs, path)
        if best_fit is None or entry[0] > best_fit[0]:
            best_fit = entry

    if best_fit is None:
        weights = UnifiedWeights()
        json.dump(
            {
                "lambda1": weights.lambda1,
                "lambda2": weights.lambda2,
                "lambda3": weights.lambda3,
                "lambda4": weights.lambda4,
                "c_atomic": weights.c_atomic,
                "achieved_r": 0.0,
                "calibration_source": None,
            },
            open("unified_weights.json", "w"),
            indent=2,
        )
        return weights

    _, achieved_r, best_w, source_path = best_fit
    lambda1, lambda2, lambda3, lambda4 = [float(v) for v in best_w]

    print("Calibrated weights:")
    print(f"  lambda1={lambda1:.4f} (tile quantization)")
    print(f"  lambda2={lambda2:.4f} (wave quantization)")
    print(f"  lambda3={lambda3:.4f} (L2 reuse benefit)")
    print(f"  lambda4={lambda4:.4f} (reduction cost)")
    print(f"  Source dataset: {source_path}")
    print(f"  Achieved r(score, latency) = {achieved_r:.3f}")

    weights = UnifiedWeights(
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        lambda3=float(lambda3),
        lambda4=float(lambda4),
        c_atomic=0.05,
    )

    json.dump(
        {
            "lambda1": weights.lambda1,
            "lambda2": weights.lambda2,
            "lambda3": weights.lambda3,
            "lambda4": weights.lambda4,
            "c_atomic": weights.c_atomic,
            "achieved_r": achieved_r,
            "calibration_source": source_path,
        },
        open("unified_weights.json", "w"),
        indent=2,
    )
    return weights
