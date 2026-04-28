"""Compare ct.mma and ct.matmul paths on the same GEMM workload."""

from __future__ import annotations

import time

import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]
HAS_MMA = hasattr(ct, "mma")


@ct.kernel
def kernel_mma(A, B, C, M, N, K, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid = ct.bid(0)
    num_m = ct.cdiv(M, tm)
    num_n = ct.cdiv(N, tn)
    pid_m = bid // num_n
    pid_n = bid % num_n
    if pid_m >= num_m:
        return

    acc = ct.zeros((tm, tn), dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    num_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    for k in range(num_k):
        a = ct.load(A, index=(pid_m, k), shape=(tm, tk), padding_mode=zero_pad, latency=8)
        b = ct.load(B, index=(k, pid_n), shape=(tk, tn), padding_mode=zero_pad, latency=8)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


@ct.kernel
def kernel_matmul(A, B, C, M, N, K, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid = ct.bid(0)
    num_m = ct.cdiv(M, tm)
    num_n = ct.cdiv(N, tn)
    pid_m = bid // num_n
    pid_n = bid % num_n
    if pid_m >= num_m:
        return

    acc = ct.zeros((tm, tn), dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    num_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    for k in range(num_k):
        a = ct.load(A, index=(pid_m, k), shape=(tm, tk), padding_mode=zero_pad, latency=8)
        b = ct.load(B, index=(k, pid_n), shape=(tk, tn), padding_mode=zero_pad, latency=8)
        acc = ct.matmul(a, b) + acc

    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


def _time_kernel(kernel, A, B, C, M, N, K, tm, tn, tk, iters=40):
    grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
    for _ in range(3):
        ct.launch(cp.cuda.get_current_stream(), grid, kernel, (A, B, C, M, N, K, tm, tn, tk))
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        ct.launch(cp.cuda.get_current_stream(), grid, kernel, (A, B, C, M, N, K, tm, tn, tk))
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def run_comparison():
    M = N = K = 2048
    tm, tn, tk = 64, 64, 32

    A = cp.random.standard_normal((M, K), dtype=cp.float32).astype(cp.float16)
    B = cp.random.standard_normal((K, N), dtype=cp.float32).astype(cp.float16)
    C = cp.zeros((M, N), dtype=cp.float32)

    ms_matmul = _time_kernel(kernel_matmul, A, B, C, M, N, K, tm, tn, tk)
    tflops_matmul = (2 * M * N * K) / ms_matmul / 1e9

    print(f"ct.mma available: {HAS_MMA}")
    print(f"matmul path: {ms_matmul:.3f} ms, {tflops_matmul:.1f} TFLOP/s")

    if HAS_MMA:
        ms_mma = _time_kernel(kernel_mma, A, B, C, M, N, K, tm, tn, tk)
        tflops_mma = (2 * M * N * K) / ms_mma / 1e9
        print(f"mma path:    {ms_mma:.3f} ms, {tflops_mma:.1f} TFLOP/s")
        print(f"mma speedup over matmul path: {ms_matmul / ms_mma:.3f}x")

    print("Run Nsight Compute to verify tensor core pipe usage:")
    print("ncu --metrics sm__inst_executed_pipe_tensor.sum python cutile_gemm/mma_vs_matmul_bench.py")


if __name__ == "__main__":
    run_comparison()
