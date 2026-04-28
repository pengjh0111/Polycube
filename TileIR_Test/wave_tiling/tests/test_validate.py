from wave_tiling import CuboidWaveShape, TilingResult, benchmark_config, hypothesis_test


def test_benchmark_config_uses_backends():
    result = TilingResult(
        tile_m=16,
        tile_n=16,
        splitk_factor=1,
        wave_shape=CuboidWaveShape(sk=1, m=1, n=4),
        wave_count=4,
        benefit=1.0,
    )

    def benchmark_fn(M, N, K, result, num_warmup, num_iters):
        return 2.0

    def baseline_fn(M, N, K, num_warmup, num_iters):
        return 6.0

    bench = benchmark_config(32, 32, 32, result, benchmark_fn=benchmark_fn, baseline_fn=baseline_fn)
    assert bench.latency_ms == 2.0
    assert bench.baseline_latency_ms == 6.0
    assert bench.speedup == 3.0


def test_hypothesis_test_uses_backends():
    from wave_tiling import HardwareParams

    hw = HardwareParams(sm_count=12)

    def benchmark_fn(M, N, K, result, num_warmup, num_iters):
        return float(result.wave_count)

    def baseline_fn(M, N, K, num_warmup, num_iters):
        return 10.0

    summary = hypothesis_test(32, 32, 32, hw, benchmark_fn=benchmark_fn, baseline_fn=baseline_fn)
    assert summary["joint_ms"] > 0