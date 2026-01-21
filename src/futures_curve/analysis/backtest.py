from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .costs import CostModel
from .metrics import compute_drawdown_series, compute_sharpe_annualized, compute_turnover
from .risk import ewma_volatility, position_from_vol_target


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    pnl_net: pd.Series
    pnl_gross: pd.Series
    costs: pd.Series
    position: pd.Series
    max_drawdown: float
    sharpe_annual: float
    turnover_daily: float


def run_backtest_single_series(
    *,
    config: AnalysisConfig,
    price_exec: pd.Series,
    signal: pd.Series,
    contract_size: float,
    cost_model: CostModel,
) -> BacktestResult:
    """Backtest a single spread series with causal execution.

    Conventions:
    - `signal` is observed on trade_date t from the signal price (US-session VWAP proxy).
    - position is *executed* on trade_date t+1 using execution bucket price.
    - PnL is computed open-to-open on execution prices.
    """
    price_exec = price_exec.sort_index()
    signal = signal.reindex(price_exec.index).sort_index()

    # Position is applied on t based on signal from t-1 (next-trade_date execution).
    signal_for_exec = signal.shift(1)

    # One-contract $ PnL series for volatility estimation.
    delta_price = price_exec.diff()
    pnl_1 = (delta_price * contract_size).fillna(0.0)

    vol_1 = ewma_volatility(
        pnl_1,
        halflife_days=config.ewma_halflife_days,
        min_periods=config.ewma_min_periods,
    )
    target_dollar_vol = (
        config.initial_capital * config.vol_target_annual / np.sqrt(config.trading_days_per_year)
    )
    position_target = position_from_vol_target(
        signal=signal_for_exec,
        vol_per_contract=vol_1,
        target_dollar_vol=target_dollar_vol,
        allow_fractional=config.allow_fractional_contracts,
    )

    position_target = position_target.fillna(0.0)

    # PnL from holding yesterday's position over (t-1 -> t).
    pnl_gross = (position_target.shift(1).fillna(0.0) * pnl_1).reindex(price_exec.index).fillna(0.0)

    # Trading cost paid when changing position at the open of t.
    delta_pos = position_target.diff().fillna(position_target)
    costs = delta_pos.abs().map(cost_model.cost_per_unit_change) * config.cost_multiplier

    pnl_net = pnl_gross - costs

    equity = pnl_net.cumsum().add(config.initial_capital)
    returns = pnl_net / equity.shift(1).replace(0.0, np.nan)
    returns = returns.fillna(0.0)

    dd = compute_drawdown_series(equity)
    max_dd = float(dd.max()) if len(dd) else 0.0

    sharpe = compute_sharpe_annualized(returns, trading_days=config.trading_days_per_year)
    turnover = compute_turnover(position_target)

    return BacktestResult(
        equity=equity,
        returns=returns,
        pnl_net=pnl_net,
        pnl_gross=pnl_gross,
        costs=costs,
        position=position_target,
        max_drawdown=max_dd,
        sharpe_annual=sharpe,
        turnover_daily=turnover,
    )

