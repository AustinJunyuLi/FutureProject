from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Simple tick-based trading cost model."""

    dollars_per_tick: float
    legs: int
    ticks_per_leg_per_side: float

    def cost_per_unit_change(self, unit_change: float) -> float:
        """Cost for changing position by `unit_change` (absolute) on one execution."""
        # per side cost for 1 unit = legs * ticks_per_leg_side * $/tick
        return abs(unit_change) * self.legs * self.ticks_per_leg_per_side * self.dollars_per_tick

