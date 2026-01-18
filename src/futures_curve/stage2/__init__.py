"""Stage 2: Curve construction, spreads, and roll detection."""

from .contract_ranker import (
    ContractRanker,
    build_curve_panel,
    build_daily_curve_panel,
)
from .spread_calculator import (
    SpreadCalculator,
    build_spread_panel,
    extract_front_spread,
)
from .roll_detector import (
    RollDetector,
    build_roll_volume_panel,
    detect_rolls,
)
from .pipeline import (
    Stage2Pipeline,
    run_stage2,
    read_curve_panel,
    read_spread_panel,
    read_roll_events,
)

__all__ = [
    "ContractRanker",
    "build_curve_panel",
    "build_daily_curve_panel",
    "SpreadCalculator",
    "build_spread_panel",
    "extract_front_spread",
    "RollDetector",
    "build_roll_volume_panel",
    "detect_rolls",
    "Stage2Pipeline",
    "run_stage2",
    "read_curve_panel",
    "read_spread_panel",
    "read_roll_events",
]
