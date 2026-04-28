from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

cp = pytest.importorskip("cupy")
ct = pytest.importorskip("cuda.tile")

from wave_tiling import CuboidWaveShape, TaskSpace3D, compute_wave_count, verify_coverage

from cutile_gemm.baseline_gemm import launch_baseline
from cutile_gemm.wave_tiling_gemm import launch_wave_tiling, wave_tiling_gemm_kernel


SHAPES = [(32, 512, 256), (16, 1024, 512), (64, 2048, 1024)]


def _ref_gemm(A, B):
    return cp.asarray(np.dot(A.get().astype(np.float32), B.get().astype(np.float32)))


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_baseline_correctness(M, N, K):
    rng = cp.random.default_rng(7)
    A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
    C_ref = _ref_gemm(A, B)
    C_out = launch_baseline(A, B, tile_m=16, tile_n=64, tile_k=32, splitk_factor=4)
    err = float(cp.max(cp.abs(C_out - C_ref)))
    assert err < 5e-2


@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("ws_k,ws_m,ws_n", [(1, 1, 1), (1, 2, 4), (2, 2, 2), (4, 1, 2)])
def test_wave_tiling_matches_baseline(M, N, K, ws_k, ws_m, ws_n):
    rng = cp.random.default_rng(11)
    A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

    tile_m, tile_n, tile_k, splitk = 16, 64, 32, 4
    C_base = launch_baseline(A, B, tile_m, tile_n, tile_k, splitk)
    C_wt = launch_wave_tiling(A, B, tile_m, tile_n, tile_k, splitk, ws_k, ws_m, ws_n)

    err = float(cp.max(cp.abs(C_wt - C_base)))
    assert err < 2e-2


def test_coverage_all_shapes():
    for M, N, K in SHAPES:
        task = TaskSpace3D.from_problem(M, N, K, 16, 64, 4)
        for ws_k, ws_m, ws_n in [(1, 2, 4), (2, 2, 2), (4, 1, 2)]:
            shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
            assert verify_coverage(task, shape)


def _launch_wave_into_buffer(A, B, C, tile_m, tile_n, tile_k, splitk, ws_k, ws_m, ws_n):
    M, K = A.shape
    _, N = B.shape
    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
    shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
    total_blocks = compute_wave_count(task, shape) * shape.size()
    ct.launch(
        cp.cuda.get_current_stream(),
        (total_blocks, 1, 1),
        wave_tiling_gemm_kernel,
        (A, B, C, M, N, K, tile_m, tile_n, tile_k, splitk, ws_k, ws_m, ws_n),
    )
    cp.cuda.get_current_stream().synchronize()


def test_no_output_corruption():
    M, N, K = 32, 512, 256
    rng = cp.random.default_rng(19)
    A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

    # Reuse the same output buffer object across runs/shapes.
    C_buf = cp.zeros((M, N), dtype=cp.float32)
    C_buf.fill(0)
    _launch_wave_into_buffer(A, B, C_buf, 16, 64, 32, 4, 1, 2, 4)
    out1 = C_buf.copy()

    C_buf.fill(0)
    _launch_wave_into_buffer(A, B, C_buf, 16, 64, 32, 4, 2, 2, 2)
    out2 = C_buf.copy()

    err = float(cp.max(cp.abs(out1 - out2)))
    assert err < 2e-2
