"""StreamK GEMM kernel for improved SM utilization on small-M/N large-K workloads."""

from __future__ import annotations

import os
import sys

import cupy as cp
import cuda.tile as ct
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@ct.kernel
def streamk_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    tile_k: ct.Constant[int],
    total_iters,
    iters_per_sm,
    num_sms,
):
    sm_id = ct.bid(0)

    remainder = total_iters % num_sms
    iter_start = sm_id * iters_per_sm + ct.minimum(sm_id, remainder)
    iter_end = iter_start + iters_per_sm + (1 if sm_id < remainder else 0)

    if iter_start >= iter_end:
        return

    num_tiles_n = ct.cdiv(N, tile_n)
    num_k_iters = ct.cdiv(K, tile_k)

    iter_idx = iter_start
    while iter_idx < iter_end:
        tile_id = iter_idx // num_k_iters
        tile_iter_base = tile_id * num_k_iters

        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n

        k_iter_start = iter_idx - tile_iter_base
        k_iter_end = ct.minimum(iter_end - tile_iter_base, num_k_iters)

        acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
        ki = k_iter_start
        while ki < k_iter_end:
            a_tile = ct.load(
                A,
                index=(pid_m, ki),
                shape=(tile_m, tile_k),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_tile = ct.load(
                B,
                index=(ki, pid_n),
                shape=(tile_k, tile_n),
                padding_mode=ct.PaddingMode.ZERO,
            )
            acc = ct.matmul(a_tile, b_tile) + acc
            ki += 1

        owns_full_tile = (k_iter_start == 0 and k_iter_end == num_k_iters)
        if owns_full_tile:
            ct.store(C, index=(pid_m, pid_n), tile=acc)
        else:
            row_base = pid_m * tile_m
            col_base = pid_n * tile_n
            rows = ct.arange(tile_m, dtype=ct.int32) + row_base
            cols = ct.arange(tile_n, dtype=ct.int32) + col_base
            rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
            cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
            ct.atomic_add(C, (rows2d, cols2d), acc, check_bounds=True)

        iter_idx = tile_iter_base + k_iter_end


def launch_streamk(A, B, tile_m=16, tile_n=64, tile_k=32, num_sms=None):
    """Launch StreamK GEMM with one block per SM."""

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    if num_sms is None:
        num_sms = cp.cuda.Device(0).attributes["MultiProcessorCount"]

    num_tiles_m = (M + tile_m - 1) // tile_m
    num_tiles_n = (N + tile_n - 1) // tile_n
    num_tiles = num_tiles_m * num_tiles_n
    num_k_iters = (K + tile_k - 1) // tile_k
    total_iters = num_tiles * num_k_iters
    iters_per_sm = total_iters // num_sms

    C = cp.zeros((M, N), dtype=cp.float32)
    ct.launch(
        cp.cuda.get_current_stream(),
        (num_sms, 1, 1),
        streamk_gemm_kernel,
        (A, B, C, M, N, K, tile_m, tile_n, tile_k, total_iters, iters_per_sm, num_sms),
    )
    cp.cuda.get_current_stream().synchronize()
    return C


def verify_streamk(M=32, N=512, K=256, atol=5e-2):
    """Validate StreamK output against NumPy fp32 GEMM reference."""

    rng = cp.random.default_rng(42)
    A = rng.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = rng.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
    C_ref = cp.asarray(np.dot(A.get().astype(np.float32), B.get().astype(np.float32)))
    C_out = launch_streamk(A, B, tile_m=16, tile_n=64, tile_k=32)
    err = float(cp.max(cp.abs(C_out - C_ref)))
    print(f"StreamK max_err: {err:.5f}")
    if err >= atol:
        raise AssertionError("StreamK correctness FAILED")


if __name__ == "__main__":
    verify_streamk()
