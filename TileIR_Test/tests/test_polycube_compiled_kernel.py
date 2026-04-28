"""Tests for compile-time IR polycube kernel vs runtime LUT kernel."""
import time

import cupy as cp
import torch

from cutile_gemm.polycube_runtime_gemm import (
    build_polycube_plan,
    launch_polycube_plan,
    build_polycube_compiled_plan,
    launch_polycube_compiled,
)
from wave_tiling.wave_shape import PolycubeWaveShape


SHAPE = PolycubeWaveShape(frozenset([
    (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
    (0, 2, 2), (1, 0, 1), (1, 1, 1), (1, 2, 1), (1, 3, 1),
]))


def _allclose_fp16(a, b, atol=1e-1):
    return float(cp.max(cp.abs(a.astype(cp.float32) - b.astype(cp.float32)))) < atol


class TestCompiledKernelCorrectness:

    def test_compiled_matches_lut_sq1k(self):
        """Compiled kernel must produce same result as LUT kernel."""
        M, N, K = 1024, 1024, 1024
        tile_m, tile_n, tile_k, splitk = 16, 256, 32, 1

        A = cp.random.standard_normal((M, K)).astype(cp.float16)
        B = cp.random.standard_normal((K, N)).astype(cp.float16)

        lut_plan = build_polycube_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)
        comp_plan = build_polycube_compiled_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)

        C_lut = launch_polycube_plan(A, B, lut_plan)
        C_comp = launch_polycube_compiled(A, B, comp_plan)

        assert _allclose_fp16(C_lut, C_comp), "Compiled kernel output differs from LUT kernel"

    def test_compiled_matches_cublas(self):
        """Compiled kernel result must be close to cuBLAS reference."""
        M, N, K = 512, 512, 512
        tile_m, tile_n, tile_k, splitk = 16, 256, 32, 1

        A = cp.random.standard_normal((M, K)).astype(cp.float16)
        B = cp.random.standard_normal((K, N)).astype(cp.float16)

        A_t = torch.as_tensor(A, device="cuda")
        B_t = torch.as_tensor(B, device="cuda")
        C_ref = cp.asarray(torch.mm(A_t, B_t).cpu().numpy())

        comp_plan = build_polycube_compiled_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)
        C_comp = launch_polycube_compiled(A, B, comp_plan)

        assert _allclose_fp16(C_comp, C_ref), "Compiled kernel differs from cuBLAS reference"


class TestCompiledVsLUTPerformance:

    def _time_ms(self, fn, *args, warmup=5, iters=50):
        for _ in range(warmup):
            fn(*args)
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        cp.cuda.Device().synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0

    def test_compiled_not_slower_than_lut(self):
        """
        Compiled kernel must not be slower than LUT kernel by more than 10%.
        (Compile-time decode should be equal or faster due to no LUT reads.)
        """
        M, N, K = 1024, 1024, 1024
        tile_m, tile_n, tile_k, splitk = 16, 256, 32, 1

        A = cp.random.standard_normal((M, K)).astype(cp.float16)
        B = cp.random.standard_normal((K, N)).astype(cp.float16)

        lut_plan = build_polycube_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)
        comp_plan = build_polycube_compiled_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)

        ms_lut = self._time_ms(launch_polycube_plan, A, B, lut_plan)
        ms_comp = self._time_ms(launch_polycube_compiled, A, B, comp_plan)

        ratio = ms_comp / ms_lut
        print(f"\nLUT: {ms_lut:.3f}ms  Compiled: {ms_comp:.3f}ms  ratio: {ratio:.3f}x")
        assert ratio < 1.10, f"Compiled kernel is {ratio:.2f}x slower than LUT — unacceptable regression"

    def test_compiled_faster_on_small_shape(self):
        """
        For small shapes (llm-decode), compiled should be faster than LUT
        because LUT overhead is relatively larger.
        """
        M, N, K = 1, 4096, 4096
        tile_m, tile_n, tile_k, splitk = 16, 256, 32, 16

        A = cp.random.standard_normal((M, K)).astype(cp.float16)
        B = cp.random.standard_normal((K, N)).astype(cp.float16)

        lut_plan = build_polycube_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)
        comp_plan = build_polycube_compiled_plan(M, N, K, tile_m, tile_n, tile_k, splitk, SHAPE)

        ms_lut = self._time_ms(launch_polycube_plan, A, B, lut_plan)
        ms_comp = self._time_ms(launch_polycube_compiled, A, B, comp_plan)

        print(f"\nllm-decode LUT: {ms_lut:.3f}ms  Compiled: {ms_comp:.3f}ms")
        assert ms_comp <= ms_lut * 1.05, f"Compiled not faster on small shape: {ms_comp:.3f}ms vs {ms_lut:.3f}ms"
