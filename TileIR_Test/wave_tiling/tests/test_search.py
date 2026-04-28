import random

from wave_tiling import (
    HardwareParams,
    TaskSpace3D,
    CuboidWaveShape,
    compute_wave_count,
    search_optimal_wave_shape,
    verify_coverage,
)


def test_compute_wave_count_matches_manual_formula():
    task = TaskSpace3D(2, 8, 8)
    shape = CuboidWaveShape(sk=1, m=4, n=4)
    assert compute_wave_count(task, shape) == 8


def test_verify_coverage_for_random_task_shape_pairs():
    rng = random.Random(0)
    shapes = [CuboidWaveShape(sk=1, m=1, n=1), CuboidWaveShape(sk=1, m=2, n=2), CuboidWaveShape(sk=2, m=1, n=2)]
    for _ in range(10):
        task = TaskSpace3D(
            sk_dim=rng.randint(1, 4),
            tm_dim=rng.randint(1, 5),
            tn_dim=rng.randint(1, 5),
        )
        assert verify_coverage(task, rng.choice(shapes))


def test_search_optimal_wave_shape_returns_pareto_frontier():
    hw = HardwareParams(sm_count=12)
    frontier = search_optimal_wave_shape(
        M=32,
        N=4096,
        K=8192,
        tile_m_candidates=[16, 32],
        tile_n_candidates=[64, 128],
        splitk_candidates=[1, 2, 4],
        hw=hw,
        max_shapes=64,
    )
    assert len(frontier) >= 3
    assert all(item.wave_count > 0 for item in frontier)