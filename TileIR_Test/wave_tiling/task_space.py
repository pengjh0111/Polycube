"""Task-space abstractions for 3D wave tiling."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpace3D:
    """Represents the 3D integer lattice of all blocks to be computed."""

    sk_dim: int
    tm_dim: int
    tn_dim: int

    @classmethod
    def from_problem(
        cls,
        M: int,
        N: int,
        K: int,
        tile_m: int,
        tile_n: int,
        splitk_factor: int,
    ) -> "TaskSpace3D":
        """Construct from GEMM shape and tile configuration.

        This model assumes split-k partitions the reduction axis explicitly.
        If the active kernel path does not materialize split-k blocks yet,
        the corresponding integration code should pass ``splitk_factor=1``.
        """

        if tile_m <= 0 or tile_n <= 0:
            raise ValueError("tile dimensions must be positive")
        if splitk_factor <= 0:
            raise ValueError("splitk_factor must be positive")
        tm_dim = math.ceil(M / tile_m)
        tn_dim = math.ceil(N / tile_n)
        sk_dim = max(1, splitk_factor)
        return cls(sk_dim=sk_dim, tm_dim=tm_dim, tn_dim=tn_dim)

    def total_blocks(self) -> int:
        return self.sk_dim * self.tm_dim * self.tn_dim

    def lower_bound_waves(self, num_sms: int) -> int:
        return math.ceil(self.total_blocks() / num_sms)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.sk_dim, self.tm_dim, self.tn_dim)