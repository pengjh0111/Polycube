"""Consistency checks for scan-order placement in search and execution."""

import time


def test_first_wave_matches_execution_order():
    """
    The first (num_sm // f) blocks in execution order must match
    scan_order_first_wave coordinates.
    """
    from wave_tiling.search import scan_order_first_wave, scan_order_full_cover
    from wave_tiling.task_space import TaskSpace3D
    from wave_tiling.wave_shape import PolycubeWaveShape

    shape = PolycubeWaveShape(frozenset([(0, 0, 0), (0, 1, 0), (0, 1, 1)]))
    task = TaskSpace3D(sk_dim=2, tm_dim=8, tn_dim=8)
    num_sm = 12

    first_wave = scan_order_first_wave(task, shape, num_sm)

    full_order = scan_order_full_cover(task, shape)
    execution_first_wave = set(full_order[:num_sm])

    assert first_wave == execution_first_wave, (
        f"Search composite {first_wave} != execution first wave {execution_first_wave}"
    )


def test_full_cover_covers_all_points():
    """scan_order_full_cover must cover every task point exactly once."""
    from wave_tiling.search import scan_order_full_cover
    from wave_tiling.task_space import TaskSpace3D
    from wave_tiling.wave_shape import PolycubeWaveShape

    shape = PolycubeWaveShape(frozenset([(0, 0, 0), (0, 0, 1), (0, 1, 0)]))
    task = TaskSpace3D(sk_dim=2, tm_dim=6, tn_dim=6)

    ordered = scan_order_full_cover(task, shape)
    all_points = {
        (sk, m, n)
        for sk in range(task.sk_dim)
        for m in range(task.tm_dim)
        for n in range(task.tn_dim)
    }

    assert set(ordered) == all_points, "Must cover all task space points"
    assert len(ordered) == len(all_points), "No duplicates"


def test_search_timing_large_shape():
    """search_optimal_wave_shape must complete sq-4k in under 60s."""
    from wave_tiling.benefit_model import HardwareParams
    from wave_tiling.search import search_optimal_wave_shape

    hw = HardwareParams(sm_count=170)
    t0 = time.perf_counter()
    results = search_optimal_wave_shape(
        M=4096,
        N=4096,
        K=4096,
        tile_m_candidates=[16, 32, 64],
        tile_n_candidates=[32, 64, 128, 256],
        splitk_candidates=[1, 2, 4, 8, 16, 32, 64],
        hw=hw,
        shape_type="polycube",
        max_shapes=50,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0, f"Took {elapsed:.1f}s"
    assert len(results) > 0
