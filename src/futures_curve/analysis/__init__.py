"""Research analysis utilities (out-of-pipeline).

This package is intentionally separate from the Stage 0–2 pipeline. It provides
lightweight, reproducible research helpers (strategy prototypes, backtests, and
diagnostics) built on top of Stage2 parquet outputs.
"""

from .backtest import BacktestResult, run_backtest_single_series
from .config import AnalysisConfig
from .costs import CostModel
from .data import SpreadDailySeries
from .metrics import compute_drawdown_series, compute_sharpe_annualized, compute_turnover
from .risk import ewma_volatility, position_from_vol_target
from .types import StrategyResult

