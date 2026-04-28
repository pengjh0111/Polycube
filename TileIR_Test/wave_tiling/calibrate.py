"""Calibrate benefit-model weights against benchmark measurements."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _load_results(path: str | Path = "benchmark_results.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def _build_normalized(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["M"], row["N"], row["K"], row["splitk"])
        groups[key].append(row)

    feats = []
    target = []
    wave_count = []
    for grp in groups.values():
        min_lat = min(item["ms_wave_tiling"] for item in grp)
        for item in grp:
            sk, m, n = item["wave_shape"]
            wave_size = sk * m * n
            feats.append([float(sk), float(wave_size / m), float(wave_size / n)])
            target.append(float(item["ms_wave_tiling"] / min_lat))
            wave_count.append(float(item["predicted_waves"]))
    return (
        np.asarray(feats, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(wave_count, dtype=np.float64),
    )


def _corr(weights: np.ndarray, X: np.ndarray, y: np.ndarray, waves: np.ndarray) -> float:
    core = weights[:3]
    delta = weights[3] if weights.shape[0] > 3 else 0.0
    score = X @ core - delta * waves
    if score.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(score, y)[0, 1])


def _fit_least_squares(X: np.ndarray, y: np.ndarray, waves: np.ndarray) -> np.ndarray:
    # Solve X_aug * w ~= -y so score has a strong negative relationship with latency.
    X_aug = np.column_stack([X, waves])
    w, *_ = np.linalg.lstsq(X_aug, -y, rcond=None)
    alpha, beta, gamma, wave_coeff = w
    delta = -wave_coeff
    return np.asarray([alpha, beta, gamma, delta], dtype=np.float64)


def calibrate(
    results_path: str | Path = "benchmark_results.json",
    out_path: str | Path = "calibrated_hw_params.json",
) -> dict:
    rows = _load_results(results_path)
    X, y, waves = _build_normalized(rows)

    start = np.asarray([1.0, 2.0, 2.0, 0.1], dtype=np.float64)
    old_r = _corr(start, X, y, waves)

    best_w = _fit_least_squares(X, y, waves)
    best_r = _corr(best_w, X, y, waves)

    try:
        from scipy.optimize import minimize
    except Exception:
        # Fallback: random local search when scipy isn't available.
        rng = np.random.default_rng(0)
        for _ in range(20000):
            cand = best_w + rng.normal(scale=0.15, size=4)
            r = _corr(cand, X, y, waves)
            if r < best_r:
                best_r = r
                best_w = cand
    else:
        result = minimize(
            lambda w: _corr(w, X, y, waves),
            x0=best_w,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4},
        )
        nm_w = result.x
        nm_r = _corr(nm_w, X, y, waves)
        if nm_r < best_r:
            best_w = nm_w
            best_r = nm_r

    params = {
        "alpha": float(best_w[0]),
        "beta": float(best_w[1]),
        "gamma": float(best_w[2]),
        "delta": float(best_w[3]),
        "achieved_r": float(best_r),
        "before_r": float(old_r),
    }
    Path(out_path).write_text(json.dumps(params, indent=2))
    print(
        f"Calibrated: alpha={params['alpha']:.4f} beta={params['beta']:.4f} "
        f"gamma={params['gamma']:.4f} delta={params['delta']:.4f}"
    )
    print(f"Correlation before calibration: {old_r:.4f}")
    print(f"Correlation after calibration:  {best_r:.4f}")
    return params


if __name__ == "__main__":
    calibrate()
