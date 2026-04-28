"""Autotuner wrapper for wave-tiling cuTile GEMM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cupy as cp

from wave_tiling import HardwareParams, search_optimal_wave_shape

from .benchmark import timed_launch
from .wave_tiling_gemm import launch_wave_tiling

CACHE_PATH = Path("autotune_cache.json")


@dataclass
class GemmConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    splitk_factor: int
    ws_k: int
    ws_m: int
    ws_n: int
    predicted_waves: int
    measured_ms: float | None = None

    def launch(self, A, B):
        return launch_wave_tiling(
            A,
            B,
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.splitk_factor,
            self.ws_k,
            self.ws_m,
            self.ws_n,
        )


class GemmAutotuner:
    def __init__(
        self,
        hw: HardwareParams,
        tile_m_options=(16, 32),
        tile_n_options=(64, 128),
        tile_k_options=(32,),
        splitk_options=(1, 2, 4, 8),
        top_k_benchmark=5,
    ):
        self.hw = hw
        self.tile_m_options = tuple(tile_m_options)
        self.tile_n_options = tuple(tile_n_options)
        self.tile_k_options = tuple(tile_k_options)
        self.splitk_options = tuple(splitk_options)
        self.top_k = top_k_benchmark
        self._cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    def _key(self, M, N, K):
        return hashlib.md5(f"{M}:{N}:{K}".encode(), usedforsecurity=False).hexdigest()[:12]

    def best_config(self, M, N, K, A=None, B=None) -> GemmConfig:
        key = self._key(M, N, K)
        if key in self._cache:
            return GemmConfig(**self._cache[key])

        pareto = search_optimal_wave_shape(
            M,
            N,
            K,
            tile_m_candidates=list(self.tile_m_options),
            tile_n_candidates=list(self.tile_n_options),
            splitk_candidates=list(self.splitk_options),
            hw=self.hw,
        )

        if A is None:
            A = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
            B = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

        best_cfg = None
        best_ms = float("inf")

        for pr in pareto[: self.top_k]:
            shape = pr.wave_shape
            for tile_k in self.tile_k_options:
                ms = timed_launch(
                    launch_wave_tiling,
                    A,
                    B,
                    pr.tile_m,
                    pr.tile_n,
                    tile_k,
                    pr.splitk_factor,
                    shape.sk,
                    shape.m,
                    shape.n,
                    warmup=5,
                    iters=30,
                )
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = GemmConfig(
                        tile_m=pr.tile_m,
                        tile_n=pr.tile_n,
                        tile_k=tile_k,
                        splitk_factor=pr.splitk_factor,
                        ws_k=shape.sk,
                        ws_m=shape.m,
                        ws_n=shape.n,
                        predicted_waves=pr.wave_count,
                        measured_ms=ms,
                    )

        if best_cfg is None:
            raise RuntimeError("No valid candidate found during autotuning")

        self._cache[key] = best_cfg.__dict__
        CACHE_PATH.write_text(json.dumps(self._cache, indent=2))
        return best_cfg
