from wave_tiling import TaskSpace3D


def test_from_problem_matches_block_counts_for_three_shapes():
    cases = [
        (128, 128, 4096, 64, 64, 1, (1, 2, 2)),
        (256, 128, 8192, 64, 64, 2, (2, 4, 2)),
        (32, 4096, 16384, 16, 256, 4, (4, 2, 16)),
    ]
    for M, N, K, tile_m, tile_n, splitk, expected in cases:
        task = TaskSpace3D.from_problem(M, N, K, tile_m, tile_n, splitk)
        assert task.as_tuple() == expected
        assert task.total_blocks() == expected[0] * expected[1] * expected[2]


def test_lower_bound_waves():
    task = TaskSpace3D(2, 8, 8)
    assert task.lower_bound_waves(16) == 8