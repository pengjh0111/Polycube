"""Native polycube runtime mapping for cuTile GEMM execution.

This module executes block order exactly from a polycube point-set cover by
materializing per-block lookup tables (pid_sk, pid_m, pid_n) on device.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import cuda.tile as ct
import numpy as np

from wave_tiling.ir_emitter import make_polycube_kernel
from wave_tiling.search import scan_order_full_cover
from wave_tiling.task_space import TaskSpace3D
from wave_tiling.wave_shape import PolycubeWaveShape


_HAS_MMA = hasattr(ct, "mma")


if _HAS_MMA:

    @ct.function
    def _accumulate(a, b, acc):
        return ct.mma(a, b, acc)

else:

    @ct.function
    def _accumulate(a, b, acc):
        return ct.matmul(a, b) + acc


@dataclass
class PolycubeRuntimePlan:
    tile_m: int
    tile_n: int
    tile_k: int
    splitk_factor: int
    M: int
    N: int
    K: int
    pid_sk_lut: cp.ndarray
    pid_m_lut: cp.ndarray
    pid_n_lut: cp.ndarray


@dataclass
class PolycubeCompiledPlan:
    """Plan using compile-time decoded kernel — no LUT arrays."""

    tile_m: int
    tile_n: int
    tile_k: int
    splitk_factor: int
    M: int
    N: int
    K: int
    total_blocks: int
    kernel: object


def _prewarm_plan(plan: PolycubeCompiledPlan) -> None:
    """
    Force JIT compilation by executing one dummy launch.
    Must be called at end of every build function so timing
    loops never see the compilation cost.
    """

    M, N, K = plan.M, plan.N, plan.K
    A_dummy = cp.zeros((M, K), dtype=cp.float16)
    B_dummy = cp.zeros((K, N), dtype=cp.float16)
    launch_polycube_compiled(A_dummy, B_dummy, plan)
    cp.cuda.get_current_stream().synchronize()


@dataclass
class OptimizedCompiledPlan:
    """Plan carrying an optimized compiled kernel (C, E, or D variant)."""

    tile_m: int
    tile_n: int
    tile_k: int
    splitk_factor: int
    M: int
    N: int
    K: int
    total_blocks: int
    kernel: object


def _prewarm_optimized(plan: "OptimizedCompiledPlan") -> None:
    """Force JIT compilation during plan build, not during timing."""

    A_d = cp.zeros((plan.M, plan.K), dtype=cp.float16)
    B_d = cp.zeros((plan.K, plan.N), dtype=cp.float16)
    launch_optimized_compiled(A_d, B_d, plan)
    cp.cuda.get_current_stream().synchronize()


def launch_optimized_compiled(A, B, plan: "OptimizedCompiledPlan"):
    """Launch an optimized compiled kernel plan."""

    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M == plan.M and N == plan.N and K == plan.K
    out_dtype = cp.float16 if plan.splitk_factor == 1 else cp.float32
    C = cp.zeros((M, N), dtype=out_dtype)
    ct.launch(
        cp.cuda.get_current_stream(),
        (plan.total_blocks, 1, 1),
        plan.kernel,
        (
            A,
            B,
            C,
            M,
            N,
            K,
            plan.tile_m,
            plan.tile_n,
            plan.tile_k,
            plan.splitk_factor,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return C


def build_optimized_rowmajor_plan(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk_factor: int,
) -> OptimizedCompiledPlan:
    """
    Build C or E variant plan:
    optimized kernel (MMA+num_ctas+latency+swizzle) + swizzle block order.
    C uses heuristic tile/splitk; E uses framework tile/splitk.
    Same function — caller decides which tile/splitk to pass.
    """

    from wave_tiling.ir_emitter import make_optimized_rowmajor_kernel

    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    kernel = make_optimized_rowmajor_kernel(task)
    plan = OptimizedCompiledPlan(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        splitk_factor=splitk_factor,
        M=M,
        N=N,
        K=K,
        total_blocks=task.total_blocks(),
        kernel=kernel,
    )
    _prewarm_optimized(plan)
    return plan


def build_optimized_polycube_plan(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk_factor: int,
    shape: "PolycubeWaveShape",
) -> OptimizedCompiledPlan:
    """
    Build D variant plan:
    optimized kernel (MMA+num_ctas+latency) + polycube block order.
    Uses framework tile/splitk and framework-searched polycube shape.
    """

    from wave_tiling.ir_emitter import make_optimized_polycube_kernel

    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    kernel = make_optimized_polycube_kernel(shape, task)
    plan = OptimizedCompiledPlan(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        splitk_factor=splitk_factor,
        M=M,
        N=N,
        K=K,
        total_blocks=task.total_blocks(),
        kernel=kernel,
    )
    _prewarm_optimized(plan)
    return plan


def _build_exact_order(task: TaskSpace3D, shape: PolycubeWaveShape, strategy: str = "greedy") -> list[tuple[int, int, int]]:
    """Build execution order using deterministic scan-order polycube placement."""

    del strategy  # kept for API compatibility
    return scan_order_full_cover(task, shape)


def build_polycube_plan(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk_factor: int,
    shape: PolycubeWaveShape,
    strategy: str = "greedy",
) -> PolycubeRuntimePlan:
    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    ordered = _build_exact_order(task, shape, strategy=strategy)

    pid_sk = cp.asarray(np.asarray([q[0] for q in ordered], dtype=np.int32))
    pid_m = cp.asarray(np.asarray([q[1] for q in ordered], dtype=np.int32))
    pid_n = cp.asarray(np.asarray([q[2] for q in ordered], dtype=np.int32))

    return PolycubeRuntimePlan(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        splitk_factor=splitk_factor,
        M=M,
        N=N,
        K=K,
        pid_sk_lut=pid_sk,
        pid_m_lut=pid_m,
        pid_n_lut=pid_n,
    )


def build_polycube_compiled_plan(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk_factor: int,
    shape: PolycubeWaveShape,
) -> PolycubeCompiledPlan:
    """
    Build a plan that uses compile-time IR decode instead of runtime LUT.
    The kernel is compiled once per (shape, task) combination.
    """

    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    f = len(shape.points)
    try:
        kernel = make_polycube_kernel(shape, task)
    except Exception as exc:  # pragma: no cover - defensive compile-time guard
        raise NotImplementedError(
            f"compile-time polycube kernel is unsupported for shape with f={f} and task={task.as_tuple()}: {exc}"
        ) from exc

    total_blocks = task.total_blocks()
    compiled_plan = PolycubeCompiledPlan(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        splitk_factor=splitk_factor,
        M=M,
        N=N,
        K=K,
        total_blocks=total_blocks,
        kernel=kernel,
    )
    _prewarm_plan(compiled_plan)
    return compiled_plan


def build_rowmajor_compiled_plan(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    splitk_factor: int,
) -> PolycubeCompiledPlan:
    """
    Build a compiled plan with O(1) arithmetic row-major decode.
    No if-chain, no LUT, no static_iter limit.
    Used as B variant in block-order ablation against polycube D variant.
    """
    from wave_tiling.ir_emitter import make_rowmajor_kernel
    task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk_factor)
    kernel = make_rowmajor_kernel(task)

    rowmajor_plan = PolycubeCompiledPlan(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        splitk_factor=splitk_factor,
        M=M,
        N=N,
        K=K,
        total_blocks=task.total_blocks(),
        kernel=kernel,
    )
    _prewarm_plan(rowmajor_plan)
    return rowmajor_plan


@ct.kernel
def polycube_runtime_kernel(
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
    pid_sk_lut,
    pid_m_lut,
    pid_n_lut,
):
    bid = ct.bid(0)

    pid_sk_tile = ct.load(pid_sk_lut, index=(bid,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    pid_m_tile = ct.load(pid_m_lut, index=(bid,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    pid_n_tile = ct.load(pid_n_lut, index=(bid,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)

    pid_sk = ct.extract(pid_sk_tile, index=(0,), shape=())
    pid_m = ct.extract(pid_m_tile, index=(0,), shape=())
    pid_n = ct.extract(pid_n_tile, index=(0,), shape=())

    num_blocks_m = ct.cdiv(M, tile_m)
    num_blocks_n = ct.cdiv(N, tile_n)
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


def launch_polycube_plan(A, B, plan: PolycubeRuntimePlan):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M == plan.M and N == plan.N and K == plan.K

    out_dtype = cp.float16 if plan.splitk_factor == 1 else cp.float32
    C = cp.zeros((M, N), dtype=out_dtype)

    total_blocks = int(plan.pid_sk_lut.shape[0])

    ct.launch(
        cp.cuda.get_current_stream(),
        (total_blocks, 1, 1),
        polycube_runtime_kernel,
        (
            A,
            B,
            C,
            M,
            N,
            K,
            plan.tile_m,
            plan.tile_n,
            plan.tile_k,
            plan.splitk_factor,
            plan.pid_sk_lut,
            plan.pid_m_lut,
            plan.pid_n_lut,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return C


def launch_polycube_compiled(A, B, plan: PolycubeCompiledPlan):
    """Launch the compile-time decoded kernel. No LUT transfers needed."""

    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M == plan.M and N == plan.N and K == plan.K

    out_dtype = cp.float16 if plan.splitk_factor == 1 else cp.float32
    C = cp.zeros((M, N), dtype=out_dtype)

    ct.launch(
        cp.cuda.get_current_stream(),
        (plan.total_blocks, 1, 1),
        plan.kernel,
        (
            A,
            B,
            C,
            M,
            N,
            K,
            plan.tile_m,
            plan.tile_n,
            plan.tile_k,
            plan.splitk_factor,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return C
