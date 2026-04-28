"""Optimized cuTile GEMM baseline used by scheduling experiments."""

from __future__ import annotations

import inspect

import cuda.tile as ct
import cupy as cp

ConstInt = ct.Constant[int]
_HAS_MMA = hasattr(ct, "mma")
_NUM_CTAS = ct.ByTarget(sm_100=2, default=2)


@ct.function
def swizzle_2d_from_linear(
    linear_idx: int,
    num_blocks_m: int,
    num_blocks_n: int,
    group_size_m: ConstInt,
):
    """Map linear MN tile index to swizzled (m, n) for better L2 reuse."""

    blocks_per_group = group_size_m * num_blocks_n
    group_id = linear_idx // blocks_per_group
    first_block_m = group_id * group_size_m

    actual_group_m = ct.minimum(num_blocks_m - first_block_m, group_size_m)

    local_idx = linear_idx % blocks_per_group
    local_n = local_idx // actual_group_m
    local_m = local_idx % actual_group_m

    bid_m = first_block_m + local_m
    bid_n = local_n
    return bid_m, bid_n


if _HAS_MMA:

    @ct.function
    def _accumulate(a, b, acc):
        return ct.mma(a, b, acc)

else:

    @ct.function
    def _accumulate(a, b, acc):
        return ct.matmul(a, b) + acc


@ct.kernel(num_ctas=_NUM_CTAS)
def optimized_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    tile_m: ConstInt,
    tile_n: ConstInt,
    tile_k: ConstInt,
    splitk_factor: ConstInt,
    ws_k: ConstInt,
    ws_m: ConstInt,
    ws_n: ConstInt,
    group_size_m: ConstInt,
):
    num_blocks_m = ct.cdiv(M, tile_m)
    num_blocks_n = ct.cdiv(N, tile_n)

    wave_size = ws_k * ws_m * ws_n
    wave_idx = ct.bid(0) // wave_size
    local_idx = ct.bid(0) % wave_size

    local_sk = local_idx // (ws_m * ws_n)
    local_m = (local_idx % (ws_m * ws_n)) // ws_n
    local_n = local_idx % ws_n

    waves_n = ct.cdiv(num_blocks_n, ws_n)
    waves_m = ct.cdiv(num_blocks_m, ws_m)

    wave_sk = wave_idx // (waves_m * waves_n)
    wave_m = (wave_idx % (waves_m * waves_n)) // waves_n
    wave_n = wave_idx % waves_n

    pid_sk = wave_sk * ws_k + local_sk
    pid_m_raw = wave_m * ws_m + local_m
    pid_n_raw = wave_n * ws_n + local_n

    # Apply swizzle over logical (m, n) coordinates.
    linear_mn = pid_m_raw * num_blocks_n + pid_n_raw
    pid_m, pid_n = swizzle_2d_from_linear(linear_mn, num_blocks_m, num_blocks_n, group_size_m)

    if pid_m >= num_blocks_m or pid_n >= num_blocks_n or pid_sk >= splitk_factor:
        return

    k_slice = ct.cdiv(K, splitk_factor)
    k_start = pid_sk * k_slice
    k_end = ct.minimum(k_start + k_slice, K)

    A_slice = A.slice(axis=1, start=k_start, stop=k_end)
    B_slice = B.slice(axis=0, start=k_start, stop=k_end)

    acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    num_k_tiles = ct.num_tiles(A_slice, axis=1, shape=(tile_m, tile_k))
    for k_tile in range(num_k_tiles):
        a_tile = ct.load(
            A_slice,
            index=(pid_m, k_tile),
            shape=(tile_m, tile_k),
            padding_mode=zero_pad,
            latency=8,
        ).astype(dtype)
        b_tile = ct.load(
            B_slice,
            index=(k_tile, pid_n),
            shape=(tile_k, tile_n),
            padding_mode=zero_pad,
            latency=8,
        ).astype(dtype)
        acc = _accumulate(a_tile, b_tile, acc)

    acc_out = ct.astype(acc, C.dtype)
    if splitk_factor == 1:
        ct.store(C, index=(pid_m, pid_n), tile=acc_out)
    else:
        rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
        cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
        rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
        cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
        ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)


def get_ct_mma_signature() -> str:
    if not _HAS_MMA:
        return "ct.mma not available"
    try:
        return str(inspect.signature(ct.mma))
    except Exception:
        return "ct.mma available (signature introspection unavailable)"


def launch_optimized(
    A,
    B,
    tile_m,
    tile_n,
    tile_k,
    splitk_factor,
    ws_k=1,
    ws_m=1,
    ws_n=1,
    group_size_m=8,
):
    from wave_tiling import CuboidWaveShape, TaskSpace3D, compute_wave_count

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    out_dtype = cp.float16 if splitk_factor == 1 else cp.float32
    C = cp.zeros((M, N), dtype=out_dtype)

    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    shape = CuboidWaveShape(sk=ws_k, m=ws_m, n=ws_n)
    waves = compute_wave_count(task, shape)
    total_blocks = waves * shape.size()

    ct.launch(
        cp.cuda.get_current_stream(),
        (total_blocks, 1, 1),
        optimized_gemm_kernel,
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
            group_size_m,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return C
