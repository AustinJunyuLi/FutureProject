from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class AnalysisConfig:
    symbol: str = "HG"
    spreads: tuple[int, ...] = (1, 2, 3, 4)

    # Trade-date inputs
    start_date: date | None = None
    end_date: date | None = None

    # Signal observation: US session (buckets 1–7) VWAP proxy (volume-weighted bucket closes)
    signal_buckets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

    # Execution convention: next trade_date bucket 1 (09:00–09:59 CT)
    execution_bucket: int = 1

    # Risk / evaluation
    initial_capital: float = 1_000_000.0
    vol_target_annual: float = 0.10
    trading_days_per_year: int = 252
    max_drawdown_limit: float = 0.15

    # Costs: ticks per leg per side; spread = 2 legs
    ticks_per_leg_per_side: float = 1.0
    cost_multiplier: float = 1.0

    # Vol estimator for position sizing
    ewma_halflife_days: int = 30
    ewma_min_periods: int = 20

    # If True, allow fractional contract counts in sizing (research mode).
    allow_fractional_contracts: bool = True

