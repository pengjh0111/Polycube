"""Wave-tiling GEMM variant with lightweight decode constants."""

from __future__ import annotations

import os
import sys

import cupy as cp
import cuda.tile as ct

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wave_tiling import CuboidWaveShape, TaskSpace3D, compute_wave_count


@ct.kernel
def wave_tiling_lightweight_kernel(
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
    ws_k: ct.Constant[int],
    ws_m: ct.Constant[int],
    ws_n: ct.Constant[int],
    waves_m: ct.Constant[int],
    waves_n: ct.Constant[int],
):
    bid = ct.bid(0)

    num_blocks_m = ct.cdiv(M, tile_m)
    num_blocks_n = ct.cdiv(N, tile_n)

    wave_size = ws_k * ws_m * ws_n
    wave_idx = bid // wave_size
    local_idx = bid % wave_size

    local_sk = local_idx // (ws_m * ws_n)
    local_m = (local_idx % (ws_m * ws_n)) // ws_n
    local_n = local_idx % ws_n

    wave_sk = wave_idx // (waves_m * waves_n)
    wave_m = (wave_idx % (waves_m * waves_n)) // waves_n
    wave_n = wave_idx % waves_n

    pid_sk = wave_sk * ws_k + local_sk
    pid_m = wave_m * ws_m + local_m
    pid_n = wave_n * ws_n + local_n

    if pid_m >= num_blocks_m or pid_n >= num_blocks_n or pid_sk >= splitk_factor:
        return

    k_slice = ct.cdiv(K, splitk_factor)
    k_start = pid_sk * k_slice
    k_end = ct.minimum(k_start + k_slice, K)

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


def launch_wave_tiling_lightweight(A, B, tile_m, tile_n, tile_k, splitk_factor, ws_k, ws_m, ws_n):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = cp.zeros((M, N), dtype=cp.float32)
    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
    waves = compute_wave_count(task, shape)
    total_blocks = waves * shape.size()
    waves_m = (task.tm_dim + ws_m - 1) // ws_m
    waves_n = (task.tn_dim + ws_n - 1) // ws_n

    ct.launch(
        cp.cuda.get_current_stream(),
        (total_blocks, 1, 1),
        wave_tiling_lightweight_kernel,
        (
            A,
            B,
            C,
            M,
            N,
            K,
            tile_m,
            tile_n,
            tile_k,
            splitk_factor,
            ws_k,
            ws_m,
            ws_n,
            waves_m,
            waves_n,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return C
