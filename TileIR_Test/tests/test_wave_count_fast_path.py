"""Tests for the O(1) fast path in compute_wave_count_polycube."""

import math
import time

from wave_tiling.search import (
    WAVE_COUNT_FAST_PATH_THRESHOLD,
    compute_wave_count_polycube,
)
from wave_tiling.task_space import TaskSpace3D
from wave_tiling.wave_shape import PolycubeWaveShape


def _make_shape(points):
    return PolycubeWaveShape(frozenset(points))


class TestFastPath:

    def test_large_task_uses_fast_path_and_is_instant(self):
        """sq-4k sk=8 must complete in under 1 second (was 20+ hours)."""
        task = TaskSpace3D(sk_dim=8, tm_dim=256, tn_dim=128)  # total=262,144
        shape = _make_shape([(0, 0, 0), (0, 1, 0), (0, 1, 1)])  # f=3
        assert task.total_blocks() > WAVE_COUNT_FAST_PATH_THRESHOLD

        t0 = time.perf_counter()
        wc, translations = compute_wave_count_polycube(task, shape)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"Fast path took {elapsed:.2f}s, expected <1s"
        assert wc >= math.ceil(task.total_blocks() / shape.size()), (
            "wave_count must be >= analytical lower bound"
        )
        assert translations == [], "Fast path returns empty translation list"

    def test_small_task_uses_exact_path(self):
        """Small task space must still run exact greedy."""
        task = TaskSpace3D(sk_dim=2, tm_dim=4, tn_dim=8)  # total=64
        shape = _make_shape([(0, 0, 0), (0, 1, 0)])  # f=2
        assert task.total_blocks() <= WAVE_COUNT_FAST_PATH_THRESHOLD

        wc, translations = compute_wave_count_polycube(task, shape)

        # Exact path returns non-empty translations
        assert wc > 0
        assert len(translations) > 0, "Exact path must return translation list"

    def test_fast_path_lb_is_valid_lower_bound(self):
        """Fast path wave count must be >= flat lower bound ceil(total/f)."""
        task = TaskSpace3D(sk_dim=4, tm_dim=128, tn_dim=64)  # total=32,768
        shape = _make_shape([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)])  # f=4

        wc, _ = compute_wave_count_polycube(task, shape)
        lb = math.ceil(task.total_blocks() / shape.size())
        assert wc >= lb

    def test_sq4k_splitk8_completes(self):
        """Full search for sq-4k sk=8 must complete in under 30 seconds."""
        from wave_tiling.benefit_model import HardwareParams
        from wave_tiling.search import search_optimal_wave_shape

        hw = HardwareParams(sm_count=170)
        t0 = time.perf_counter()
        results = search_optimal_wave_shape(
            M=4096,
            N=4096,
            K=4096,
            tile_m_candidates=[16, 32],
            tile_n_candidates=[32, 64],
            splitk_candidates=[1, 4, 8],
            hw=hw,
            shape_type="polycube",
            max_shapes=30,
        )
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, f"Search took {elapsed:.1f}s, expected <30s"
        assert len(results) > 0

    def test_threshold_boundary(self):
        """total_blocks == threshold uses exact path; threshold+1 uses fast."""
        # threshold=4096 -> sk=1, tm=64, tn=64
        task_exact = TaskSpace3D(sk_dim=1, tm_dim=64, tn_dim=64)
        task_fast = TaskSpace3D(sk_dim=1, tm_dim=64, tn_dim=65)
        shape = _make_shape([(0, 0, 0)])

        assert task_exact.total_blocks() == WAVE_COUNT_FAST_PATH_THRESHOLD
        _, tr_exact = compute_wave_count_polycube(task_exact, shape)
        _, tr_fast = compute_wave_count_polycube(task_fast, shape)

        assert len(tr_exact) > 0, "At threshold, use exact path"
        assert len(tr_fast) == 0, "Above threshold, use fast path"
