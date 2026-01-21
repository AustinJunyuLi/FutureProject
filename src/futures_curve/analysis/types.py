from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyResult:
    name: str
    spread: str
    params: dict[str, object]
    n_days: int
    n_trades: int | None
    net_mean_daily: float
    net_sharpe_annual: float
    max_drawdown: float
    turnover_daily: float

