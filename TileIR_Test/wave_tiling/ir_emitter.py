"""Wave-shape mapping helpers for kernel code generation."""

from __future__ import annotations

import math

import cuda.tile as ct

from .task_space import TaskSpace3D
from .wave_shape import CuboidWaveShape, PolycubeWaveShape


STATIC_ITER_LIMIT = 64
SMALL_TM_LIMIT = 16
SMALL_TM_ITER_CAP = 512


def wave_shape_to_block_mapping(
    task: TaskSpace3D,
    shape: CuboidWaveShape,
    block_id: int,
) -> tuple[int, int, int]:
    """Map a linear block id to (sk_idx, tm_idx, tn_idx)."""

    wave_size = shape.size()
    wave_idx = block_id // wave_size
    local_idx = block_id % wave_size

    local_sk = local_idx // (shape.m * shape.n)
    local_m = (local_idx % (shape.m * shape.n)) // shape.n
    local_n = local_idx % shape.n

    waves_m = math.ceil(task.tm_dim / shape.m)
    waves_n = math.ceil(task.tn_dim / shape.n)

    wave_sk = wave_idx // (waves_m * waves_n)
    wave_m = (wave_idx % (waves_m * waves_n)) // waves_n
    wave_n = wave_idx % waves_n

    pid_sk = wave_sk * shape.sk + local_sk
    pid_m = wave_m * shape.m + local_m
    pid_n = wave_n * shape.n + local_n
    return pid_sk, pid_m, pid_n


def emit_wave_shape_decode(shape: CuboidWaveShape, task: TaskSpace3D) -> str:
    """Emit a CUDA C++ prologue for the 3-axis decode."""

    return f"""
    // Wave-shape decode: ({shape.sk}, {shape.m}, {shape.n}) in (K, M, N) axes
    const int _wave_size = {shape.sk * shape.m * shape.n};
    const int _wave_idx  = blockIdx.x / _wave_size;
    const int _local_idx = blockIdx.x % _wave_size;

    const int _waves_n   = ({task.tn_dim} + {shape.n} - 1) / {shape.n};
    const int _waves_m   = ({task.tm_dim} + {shape.m} - 1) / {shape.m};

    const int _local_sk  = _local_idx / ({shape.m * shape.n});
    const int _local_m   = (_local_idx % {shape.m * shape.n}) / {shape.n};
    const int _local_n   = _local_idx % {shape.n};

    const int _wave_sk   = _wave_idx / (_waves_m * _waves_n);
    const int _wave_m    = (_wave_idx % (_waves_m * _waves_n)) / _waves_n;
    const int _wave_n    = _wave_idx % _waves_n;

    const int pid_sk = _wave_sk * {shape.sk} + _local_sk;
    const int pid_m  = _wave_m  * {shape.m}  + _local_m;
    const int pid_n  = _wave_n  * {shape.n}  + _local_n;

    if (pid_m >= {task.tm_dim} || pid_n >= {task.tn_dim} || pid_sk >= {task.sk_dim})
        return;  // tail wave guard
    """.strip()


def make_polycube_kernel_from_order(
    ordered_points: list[tuple[int, int, int]],
    task: TaskSpace3D,
):
    """
    Return a @ct.kernel with block order baked in from ordered_points.

    ordered_points: complete list of (sk, m, n) in execution order.
    Uses exact path (static_iter over ordered_points) when
    len(ordered_points) <= STATIC_ITER_LIMIT, otherwise raises
    NotImplementedError — caller must ensure total_blocks is small enough.

    This is the single implementation used by both polycube and row-major
    compiled plans, ensuring B and D use identical kernel code.
    """

    total_blocks = len(ordered_points)
    if total_blocks > STATIC_ITER_LIMIT:
        raise NotImplementedError(
            f"total_blocks={total_blocks} exceeds STATIC_ITER_LIMIT={STATIC_ITER_LIMIT}. "
            f"Use make_polycube_kernel for large task spaces (fast path)."
        )
    sk_dim = task.sk_dim
    tm_dim = task.tm_dim
    tn_dim = task.tn_dim

    _HAS_MMA = hasattr(ct, "mma")
    if _HAS_MMA:

        @ct.function
        def _acc(a, b, acc):
            return ct.mma(a, b, acc)

    else:

        @ct.function
        def _acc(a, b, acc):
            return ct.matmul(a, b) + acc

    @ct.kernel
    def compiled_kernel(
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
        bid = ct.bid(0)
        pid_sk = 0
        pid_m = 0
        pid_n = 0
        for i, (pk, pm, pn) in ct.static_iter(enumerate(ordered_points_tuple)):
            if bid == i:
                pid_sk = pk
                pid_m = pm
                pid_n = pn

        if pid_m >= tm_dim or pid_n >= tn_dim or pid_sk >= sk_dim:
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
            acc = _acc(a_tile, b_tile, acc)
        acc_out = ct.astype(acc, C.dtype)
        if splitk_factor == 1:
            ct.store(C, index=(pid_m, pid_n), tile=acc_out)
        else:
            rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
            cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
            rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
            cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
            ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)

    return compiled_kernel


def make_rowmajor_kernel(task: TaskSpace3D):
    """
    Return a @ct.kernel with row-major block decode compiled in.
    O(1) pure arithmetic — no if-chain, no LUT, no static_iter.

    Decode:
        pid_sk = bid // (tm_dim * tn_dim)
        pid_m  = (bid % (tm_dim * tn_dim)) // tn_dim
        pid_n  = bid % tn_dim

    This is the B-variant baseline kernel for block-order ablation.
    Paired with make_polycube_kernel fast path (also O(f) arithmetic)
    for a clean D/B comparison with symmetric decode overhead.
    """

    sk_dim = task.sk_dim
    tm_dim = task.tm_dim
    tn_dim = task.tn_dim
    total_mn = tm_dim * tn_dim

    _HAS_MMA = hasattr(ct, "mma")
    if _HAS_MMA:

        @ct.function
        def _acc(a, b, acc):
            return ct.mma(a, b, acc)

    else:

        @ct.function
        def _acc(a, b, acc):
            return ct.matmul(a, b) + acc

    @ct.kernel
    def rowmajor_kernel(
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
        bid = ct.bid(0)

        # O(1) arithmetic decode — row-major (sk, m, n) order
        pid_sk = bid // total_mn
        pid_m = (bid % total_mn) // tn_dim
        pid_n = bid % tn_dim

        if pid_m >= tm_dim or pid_n >= tn_dim or pid_sk >= sk_dim:
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
            acc = _acc(a_tile, b_tile, acc)
        acc_out = ct.astype(acc, C.dtype)
        if splitk_factor == 1:
            ct.store(C, index=(pid_m, pid_n), tile=acc_out)
        else:
            rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
            cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
            rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
            cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
            ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)

    return rowmajor_kernel


def make_optimized_rowmajor_kernel(task: TaskSpace3D):
    """
    Optimized kernel with swizzle block order.
    Used for C variant (heuristic tile) and E variant (framework tile).
    Combines: MMA, num_ctas=2, latency=8, swizzle_2d_from_linear.
    O(1) arithmetic decode — no size limit.
    """

    tm_dim = task.tm_dim
    tn_dim = task.tn_dim
    sk_dim = task.sk_dim
    total_mn = tm_dim * tn_dim
    group_size_m = 8

    _NUM_CTAS = ct.ByTarget(sm_100=2, default=2)
    _HAS_MMA = hasattr(ct, "mma")

    if _HAS_MMA:

        @ct.function
        def _acc(a, b, acc):
            return ct.mma(a, b, acc)

    else:

        @ct.function
        def _acc(a, b, acc):
            return ct.matmul(a, b) + acc

    @ct.function
    def _swizzle(linear_idx, num_blocks_m, num_blocks_n):
        blocks_per_group = group_size_m * num_blocks_n
        group_id = linear_idx // blocks_per_group
        first_block_m = group_id * group_size_m
        actual_group_m = ct.minimum(num_blocks_m - first_block_m, group_size_m)
        local_idx = linear_idx % blocks_per_group
        local_n = local_idx // actual_group_m
        local_m = local_idx % actual_group_m
        return first_block_m + local_m, local_n

    @ct.kernel(num_ctas=_NUM_CTAS)
    def optimized_rowmajor_kernel(
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
        bid = ct.bid(0)
        pid_sk = bid // total_mn
        linear = bid % total_mn
        pid_m, pid_n = _swizzle(linear, tm_dim, tn_dim)

        if pid_m >= tm_dim or pid_n >= tn_dim or pid_sk >= sk_dim:
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
            acc = _acc(a_tile, b_tile, acc)
        acc_out = ct.astype(acc, C.dtype)
        if splitk_factor == 1:
            ct.store(C, index=(pid_m, pid_n), tile=acc_out)
        else:
            rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
            cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
            rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
            cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
            ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)

    return optimized_rowmajor_kernel


def make_optimized_polycube_kernel(
    shape: "PolycubeWaveShape",
    task: TaskSpace3D,
):
    """
    Optimized kernel with polycube block order.
    Used for D variant: framework tile/splitk + polycube ordering.
    Combines: MMA, num_ctas=2, latency=8, scan-order polycube decode.
    swizzle_2d_from_linear is NOT applied — polycube replaces it.
    """

    unit_points_sorted = sorted(shape.points)
    f = len(unit_points_sorted)
    sk_dim = task.sk_dim
    tm_dim = task.tm_dim
    tn_dim = task.tn_dim
    total_mn = tm_dim * tn_dim
    total_blocks = task.total_blocks()

    assert f <= STATIC_ITER_LIMIT, f"Polycube unit too large for static_iter: f={f}"

    _NUM_CTAS = ct.ByTarget(sm_100=2, default=2)
    _HAS_MMA = hasattr(ct, "mma")

    if _HAS_MMA:

        @ct.function
        def _acc(a, b, acc):
            return ct.mma(a, b, acc)

    else:

        @ct.function
        def _acc(a, b, acc):
            return ct.matmul(a, b) + acc

    use_exact = (
        total_blocks <= STATIC_ITER_LIMIT
        or (tm_dim <= SMALL_TM_LIMIT and total_blocks <= SMALL_TM_ITER_CAP)
    )

    if use_exact:
        from .search import scan_order_full_cover

        ordered_points_tuple = tuple(scan_order_full_cover(task, shape))

        @ct.kernel(num_ctas=_NUM_CTAS)
        def optimized_polycube_kernel(
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
            bid = ct.bid(0)
            pid_sk = 0
            pid_m = 0
            pid_n = 0
            for i, (pk, pm, pn) in ct.static_iter(enumerate(ordered_points_tuple)):
                if bid == i:
                    pid_sk = pk
                    pid_m = pm
                    pid_n = pn

            if pid_m >= tm_dim or pid_n >= tn_dim or pid_sk >= sk_dim:
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
                acc = _acc(a_tile, b_tile, acc)
            acc_out = ct.astype(acc, C.dtype)
            if splitk_factor == 1:
                ct.store(C, index=(pid_m, pid_n), tile=acc_out)
            else:
                rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
                cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
                rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
                cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
                ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)

    else:

        @ct.kernel(num_ctas=_NUM_CTAS)
        def optimized_polycube_kernel(
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
            bid = ct.bid(0)
            local = bid % f
            base = (bid // f) * f
            sk_off = base // total_mn
            m_off = (base % total_mn) // tn_dim
            n_off = base % tn_dim
            pid_sk = 0
            pid_m = 0
            pid_n = 0
            for i, (pk, pm, pn) in ct.static_iter(enumerate(unit_points_sorted)):
                if local == i:
                    pid_sk = (pk + sk_off) % sk_dim
                    pid_m = (pm + m_off) % tm_dim
                    pid_n = (pn + n_off) % tn_dim

            if pid_m >= tm_dim or pid_n >= tn_dim or pid_sk >= sk_dim:
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
                acc = _acc(a_tile, b_tile, acc)
            acc_out = ct.astype(acc, C.dtype)
            if splitk_factor == 1:
                ct.store(C, index=(pid_m, pid_n), tile=acc_out)
            else:
                rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
                cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
                rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
                cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
                ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)

    return optimized_polycube_kernel


def make_polycube_kernel(
    shape: PolycubeWaveShape,
    task: TaskSpace3D,
):
    unit_points_sorted = sorted(shape.points)
    f = len(unit_points_sorted)
    sk_dim = task.sk_dim
    tm_dim = task.tm_dim
    tn_dim = task.tn_dim
    total_mn = tm_dim * tn_dim
    total_blocks = task.total_blocks()

    assert f <= STATIC_ITER_LIMIT, f"Polycube unit too large for static_iter: f={f}"

    use_exact = (
        total_blocks <= STATIC_ITER_LIMIT
        or (tm_dim <= SMALL_TM_LIMIT and total_blocks <= SMALL_TM_ITER_CAP)
    )

    if use_exact:
        from .search import scan_order_full_cover

        ordered_points = scan_order_full_cover(task, shape)
        return make_polycube_kernel_from_order(ordered_points, task)

    else:

        _HAS_MMA = hasattr(ct, "mma")
        if _HAS_MMA:

            @ct.function
            def _acc(a, b, acc):
                return ct.mma(a, b, acc)

        else:

            @ct.function
            def _acc(a, b, acc):
                return ct.matmul(a, b) + acc

        @ct.kernel
        def polycube_compiled_kernel(
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
            bid = ct.bid(0)
            local = bid % f
            base = (bid // f) * f
            sk_off = base // total_mn
            m_off = (base % total_mn) // tn_dim
            n_off = base % tn_dim
            pid_sk = 0
            pid_m = 0
            pid_n = 0
            for i, (pk, pm, pn) in ct.static_iter(enumerate(unit_points_sorted)):
                if local == i:
                    pid_sk = (pk + sk_off) % sk_dim
                    pid_m = (pm + m_off) % tm_dim
                    pid_n = (pn + n_off) % tn_dim

            if pid_m >= tm_dim or pid_n >= tn_dim or pid_sk >= sk_dim:
                return

            # --- GEMM computation: identical to polycube_runtime_kernel ---
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
                acc = _acc(a_tile, b_tile, acc)

            acc_out = ct.astype(acc, C.dtype)
            if splitk_factor == 1:
                ct.store(C, index=(pid_m, pid_n), tile=acc_out)
            else:
                rows = ct.arange(tile_m, dtype=ct.int32) + pid_m * tile_m
                cols = ct.arange(tile_n, dtype=ct.int32) + pid_n * tile_n
                rows2d = ct.broadcast_to(ct.expand_dims(rows, axis=1), (tile_m, tile_n))
                cols2d = ct.broadcast_to(ct.expand_dims(cols, axis=0), (tile_m, tile_n))
                ct.atomic_add(C, (rows2d, cols2d), acc_out, check_bounds=True)

    return polycube_compiled_kernel