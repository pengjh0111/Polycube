"""Tests for the wave-level composite benefit fix in polycube search."""

import math
import pytest

from wave_tiling.search import wave_composite_coords
from wave_tiling.task_space import TaskSpace3D


class TestWaveCompositeCoords:

    def test_composite_larger_than_unit(self):
        """Composite must cover more coordinates than the single unit."""
        unit = [(0, 0, 0), (0, 1, 0), (0, 1, 1)]   # L-shape, f=3
        task = TaskSpace3D(sk_dim=2, tm_dim=8, tn_dim=8)
        num_sm = 12   # 12/3 = 4 copies

        composite = wave_composite_coords(unit, task, num_sm)

        assert len(composite) > len(unit), (
            f"Composite ({len(composite)} pts) must exceed unit ({len(unit)} pts)"
        )

    def test_composite_projections_gte_unit(self):
        """Each dimensional projection of composite must be >= unit projection."""
        unit = [(0, 0, 0), (0, 1, 0), (0, 1, 1)]
        task = TaskSpace3D(sk_dim=2, tm_dim=8, tn_dim=8)
        num_sm = 12

        unit_set = set(unit)
        composite = wave_composite_coords(unit, task, num_sm)

        for axis, idx in enumerate(["sk", "tm", "tn"]):
            unit_proj = len({p[axis] for p in unit_set})
            comp_proj = len({p[axis] for p in composite})
            assert comp_proj >= unit_proj, (
                f"Axis {idx}: composite projection {comp_proj} < unit {unit_proj}"
            )

    def test_all_coords_in_bounds(self):
        """All composite coordinates must lie within task space bounds."""
        unit = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]   # f=4
        task = TaskSpace3D(sk_dim=3, tm_dim=6, tn_dim=6)
        num_sm = 8

        composite = wave_composite_coords(unit, task, num_sm)

        for (sk, tm, tn) in composite:
            assert 0 <= sk < task.sk_dim, f"sk={sk} out of bounds"
            assert 0 <= tm < task.tm_dim, f"tm={tm} out of bounds"
            assert 0 <= tn < task.tn_dim, f"tn={tn} out of bounds"

    def test_single_point_unit_fills_sm(self):
        """
        A 1x1x1 unit tiled to num_sm copies in a large-enough task space
        must produce exactly num_sm distinct coordinates.
        This is the consistency check with the cuboid path: a 1x1x1 polycube
        is the degenerate cuboid, and the composite should span num_sm blocks.
        """
        unit = [(0, 0, 0)]
        num_sm = 108
        task = TaskSpace3D(sk_dim=4, tm_dim=16, tn_dim=16)   # total=1024 >> 108

        composite = wave_composite_coords(unit, task, num_sm)

        assert len(composite) == num_sm, (
            f"Expected {num_sm} distinct coords for 1-pt unit, got {len(composite)}"
        )

    def test_exact_copies_count(self):
        """num_sm // f copies are placed; composite size <= num_sm."""
        unit = [(0, 0, 0), (0, 0, 1)]   # f=2
        task = TaskSpace3D(sk_dim=2, tm_dim=4, tn_dim=8)
        num_sm = 8   # 8/2 = 4 copies

        composite = wave_composite_coords(unit, task, num_sm)

        # After modulo-wrap and dedup, composite size can be <= num_sm
        # but must be > f (single unit)
        assert len(composite) > len(unit)
        assert len(composite) <= num_sm


def test_composite_benefit_uses_sm_count_not_dedup_size():
    """
    When modulo wrap causes coordinate collisions (dedup reduces composite
    below num_sm), the benefit formula must still use W (hw.sm_count) as
    numerator, not len(composite).
    """
    from wave_tiling.benefit_model import HardwareParams, compute_benefit
    from wave_tiling.wave_shape import PolycubeWaveShape

    # Tiny task space to force wrap collisions
    unit = [(0, 0, 0), (0, 0, 1)]   # f=2
    task = TaskSpace3D(sk_dim=1, tm_dim=2, tn_dim=2)   # only 4 total cells
    num_sm = 8   # copies=4, 4*2=8 placements -> heavy collision -> dedup < 8

    composite = wave_composite_coords(unit, task, num_sm)
    assert len(composite) < num_sm, (
        "Need a collision case; increase num_sm or shrink task space"
    )

    composite_shape = PolycubeWaveShape(frozenset(composite))
    hw = HardwareParams(sm_count=num_sm, alpha=0.0, beta=1.0, gamma=0.0)

    benefit_wrong = compute_benefit(composite_shape, hw)
    benefit_right = compute_benefit(
        composite_shape,
        hw,
        wave_size_override=num_sm,
    )

    proj_m = len({p[1] for p in composite})
    assert abs(benefit_wrong - len(composite) / proj_m) < 1e-9
    assert abs(benefit_right - num_sm / proj_m) < 1e-9
    assert benefit_right > benefit_wrong, (
        "With collision, W-based benefit must exceed dedup-size benefit"
    )


class TestBenefitScoreComparison:
    """
    Confirm that the fix changes benefit scores for realistic GEMM shapes.
    Requires the full search stack.
    """

    @pytest.fixture(autouse=True)
    def _imports(self):
        from wave_tiling.benefit_model import HardwareParams
        from wave_tiling.search import search_optimal_wave_shape
        self.hw = HardwareParams(sm_count=108)   # A100-like
        self.search = search_optimal_wave_shape

    def _run(self, M, N, K, use_wave_composite: bool):
        """Run polycube search returning the top-1 TilingResult."""
        results = self.search(
            M=M, N=N, K=K,
            tile_m_candidates=[128],
            tile_n_candidates=[128],
            splitk_candidates=[1, 2],
            hw=self.hw,
            shape_type="polycube",
            max_shapes=50,
            use_wave_composite=use_wave_composite,
        )
        return results[0] if results else None

    def test_benefit_changes_for_majority_of_shapes(self):
        """
        For at least 2 of 3 GEMM shapes the composite benefit must differ
        from the single-unit benefit, confirming the fix is active.
        """
        cases = [
            (512, 512, 512, "square-small"),
            (2048, 2048, 2048, "square-large"),
            (128, 4096, 8192, "tall-skinny"),
        ]

        header = (f"{'Shape':<14} {'OldBenefit':>12} {'NewBenefit':>12} "
                  f"{'Delta':>10} {'Changed':>8}")
        print("\n" + header)
        print("-" * len(header))

        changed = 0
        for M, N, K, label in cases:
            old = self._run(M, N, K, use_wave_composite=False)
            new = self._run(M, N, K, use_wave_composite=True)
            if old is None or new is None:
                pytest.skip(f"Search returned no results for {label}")
            delta = abs(new.benefit - old.benefit)
            did_change = delta > 1e-6
            if did_change:
                changed += 1
            print(f"{label:<14} {old.benefit:>12.4f} {new.benefit:>12.4f} "
                  f"{delta:>10.4f} {'YES' if did_change else 'no':>8}")

        assert changed >= 2, (
            f"Benefit changed for only {changed}/3 shapes - "
            "fix may not be taking effect"
        )
