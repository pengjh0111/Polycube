"""Baseline Split-K GEMM implemented with cuTile Python.

This implementation uses elementwise ``ct.atomic_add`` to accumulate fp32 tiles
into ``C``. A two-pass Split-K fallback is not needed for this environment
because ``ct.atomic_add`` is available in the installed cuda.tile package.
"""

from __future__ import annotations

import os
import sys

import cupy as cp
import cuda.tile as ct
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@ct.kernel
def splitk_gemm_baseline(
    A,
    B,
    C,
    M,
    N,
    K,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    tile_k: ct.Constant[int],
    splitk_factor: ct.Constant[int],
):
    """Flat row-major block_id -> (pid_m, pid_n, pid_sk) mapping."""

    bid = ct.bid(0)

    num_blocks_m = ct.cdiv(M, tile_m)
    num_blocks_n = ct.cdiv(N, tile_n)
    num_blocks_sk = splitk_factor

    pid_m = bid // (num_blocks_n * num_blocks_sk)
    pid_n = (bid // num_blocks_sk) % num_blocks_n
    pid_sk = bid % num_blocks_sk

    if pid_m >= num_blocks_m:
        return

    k_slice = ct.cdiv(K, splitk_factor)
    k_start = pid_sk * k_slice
    k_end = ct.minimum(k_start + k_slice, K)

    # Slice K explicitly so tail chunks stay inside this split-K partition.
    A_slice = A.slice(axis=1, start=k_start, stop=k_end)
    B_slice = B.slice(axis=0, start=k_start, stop=k_end)

    acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)

    k_local = 0
    k_local_end = k_end - k_start
    while k_local < k_local_end:
        tile_k_idx = k_local // tile_k
        a_tile = ct.load(
            A_slice,
            index=(pid_m, tile_k_idx),
            shape=(tile_m, tile_k),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            B_slice,
            index=(tile_k_idx, pid_n),
            shape=(tile_k, tile_n),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.matmul(a_tile, b_tile) + acc
        k_local += tile_k

    row_base = pid_m * tile_m
    col_base = pid_n * tile_n
    rows = ct.arange(tile_m, dtype=ct.int32) + row_base
    cols = ct.arange(tile_n, dtype=ct.int32) + col_base
    rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
    cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
    ct.atomic_add(C, (rows2d, cols2d), acc, check_bounds=True)


def launch_baseline(A, B, tile_m, tile_n, tile_k, splitk_factor):
    """Launch baseline split-k kernel and return fp32 output array."""

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = cp.zeros((M, N), dtype=cp.float32)

    num_blocks_m = (M + tile_m - 1) // tile_m
    num_blocks_n = (N + tile_n - 1) // tile_n
    total_blocks = num_blocks_m * num_blocks_n * splitk_factor

    ct.launch(
        cp.cuda.get_current_stream(),
        (total_blocks, 1, 1),
        splitk_gemm_baseline,
        (A, B, C, M, N, K, tile_m, tile_n, tile_k, splitk_factor),
    )
    cp.cuda.get_current_stream().synchronize()
    return C


def verify_against_reference(M=32, N=512, K=256, atol=5e-2):
    """Quick correctness check against NumPy fp32 GEMM."""

    rng = cp.random.default_rng(42)
    A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)

    C_ref = cp.asarray(np.dot(A.get().astype(np.float32), B.get().astype(np.float32)))
    C_out = launch_baseline(
        A,
        B,
        tile_m=16,
        tile_n=64,
        tile_k=32,
        splitk_factor=4,
    )

    max_err = float(cp.max(cp.abs(C_out - C_ref)))
    print(f"Baseline max error: {max_err:.5f} (tol={atol})")
    if max_err >= atol:
        raise AssertionError("Baseline correctness FAILED")
    print("Baseline correctness OK")


if __name__ == "__main__":
    verify_against_reference()
