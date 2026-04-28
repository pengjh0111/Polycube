"""Wave-tiling helpers for Tile IR GEMM search and validation."""

from .benefit_model import HardwareParams, compute_benefit, pareto_frontier
try:
    from .ir_emitter import emit_wave_shape_decode, wave_shape_to_block_mapping
except ImportError:
    emit_wave_shape_decode = None
    wave_shape_to_block_mapping = None
from .search import (
    TilingResult,
    compute_wave_count,
    compute_wave_count_polycube,
    search_optimal_wave_shape,
)
from .search_v2 import explain_wave_excess, search_v2, search_v2_joint
from .report import generate_sweep_report
from .splitk_selector import find_optimal_splitk, select_splitk_candidates, splitk_utilization_table
from .tile_filter import TileConfig, explain_tile_filter, filter_tile_candidates
from .unified_score import UnifiedConfig, UnifiedWeights, calibrate_unified_weights, compute_unified_score
from .unified_search import explain_unified_search, unified_search
from .task_space import TaskSpace3D
try:
    from .validate import BenchmarkResult, benchmark_config, hypothesis_test, verify_coverage
except ImportError:
    BenchmarkResult = benchmark_config = hypothesis_test = verify_coverage = None
from .wave_shape import (
    CuboidWaveShape,
    PolycubeWaveShape,
    enumerate_cuboid_shapes,
    enumerate_polycubes,
)

__all__ = [
    "BenchmarkResult",
    "CuboidWaveShape",
    "HardwareParams",
    "PolycubeWaveShape",
    "TaskSpace3D",
    "TileConfig",
    "TilingResult",
    "UnifiedConfig",
    "UnifiedWeights",
    "benchmark_config",
    "calibrate_unified_weights",
    "compute_benefit",
    "compute_unified_score",
    "compute_wave_count",
    "compute_wave_count_polycube",
    "emit_wave_shape_decode",
    "explain_tile_filter",
    "explain_unified_search",
    "enumerate_cuboid_shapes",
    "enumerate_polycubes",
    "filter_tile_candidates",
    "find_optimal_splitk",
    "generate_sweep_report",
    "hypothesis_test",
    "pareto_frontier",
    "select_splitk_candidates",
    "search_optimal_wave_shape",
    "search_v2",
    "search_v2_joint",
    "explain_wave_excess",
    "splitk_utilization_table",
    "unified_search",
    "verify_coverage",
    "wave_shape_to_block_mapping",
]