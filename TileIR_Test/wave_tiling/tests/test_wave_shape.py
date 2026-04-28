from wave_tiling import CuboidWaveShape, PolycubeWaveShape, enumerate_cuboid_shapes, enumerate_polycubes


def test_enumerate_cuboid_shapes_matches_bruteforce_108():
    expected = {
        (sk, m, n)
        for sk in range(1, 109)
        for m in range(1, 109)
        for n in range(1, 109)
        if sk * m * n == 108
    }
    actual = {(shape.sk, shape.m, shape.n) for shape in enumerate_cuboid_shapes(108)}
    assert actual == expected


def test_compute_benefit_inputs_are_readable():
    from wave_tiling import HardwareParams, compute_benefit

    hw = HardwareParams(alpha=1.0, beta=2.0, gamma=3.0)
    shape = CuboidWaveShape(sk=2, m=4, n=3)
    assert compute_benefit(shape, hw) == 2 + 2.0 * (24 / 4) + 3.0 * (24 / 3)


def test_polycube_canonical_deduplication():
    shapes = enumerate_polycubes(4, max_count=64)
    canonical = {shape.points for shape in shapes}
    assert len(canonical) == len(shapes)
    assert all(isinstance(shape, PolycubeWaveShape) for shape in shapes)