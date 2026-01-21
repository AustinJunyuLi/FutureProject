from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .backtest import run_backtest_single_series
from .config import AnalysisConfig
from .costs import CostModel
from .data import SpreadDailySeries
from .types import StrategyResult
from .strategies.carry import compute_annualized_carry_signal
from .strategies.mean_reversion import MeanReversionParams, compute_dte_conditioned_zscore, positions_from_zscore


def _base_cost_model_for_spread(series: SpreadDailySeries, config: AnalysisConfig) -> CostModel:
    return CostModel(
        dollars_per_tick=series.dollars_per_tick,
        legs=2,
        ticks_per_leg_per_side=config.ticks_per_leg_per_side,
    )


def scan_carry_strategy(
    *,
    config: AnalysisConfig,
    series: SpreadDailySeries,
    thresholds: list[float],
    direction_multipliers: list[int] = [1, -1],
) -> list[StrategyResult]:
    df = series.df
    cost_model = _base_cost_model_for_spread(series, config)

    s_pct = df["s_signal_pct"].astype("float64")
    spacing = (df["far_dte_bdays"] - df["near_dte_bdays"]).astype("float64")
    spacing_years = spacing / float(config.trading_days_per_year)
    annualized = s_pct / spacing_years.replace(0.0, np.nan)

    results: list[StrategyResult] = []
    for thr in thresholds:
        base_dir = -np.sign(annualized).where(annualized.abs() >= thr).fillna(0.0)
        for mult in direction_multipliers:
            signal = (mult * base_dir).astype("float64")
            bt = run_backtest_single_series(
                config=config,
                price_exec=df["s_exec"],
                signal=signal,
                contract_size=series.contract_size,
                cost_model=cost_model,
            )
            net_mean_daily = float(bt.returns.mean())
            if not np.isfinite(net_mean_daily) or net_mean_daily <= 0:
                continue
            if bt.max_drawdown > config.max_drawdown_limit:
                continue

            results.append(
                StrategyResult(
                    name="carry",
                    spread=series.spread,
                    params={"threshold": thr, "direction": mult},
                    n_days=int(len(bt.returns)),
                    n_trades=None,
                    net_mean_daily=net_mean_daily,
                    net_sharpe_annual=float(bt.sharpe_annual),
                    max_drawdown=float(bt.max_drawdown),
                    turnover_daily=float(bt.turnover_daily),
                )
            )
    return results


def scan_mean_reversion_strategy(
    *,
    config: AnalysisConfig,
    series: SpreadDailySeries,
    param_grid: list[MeanReversionParams],
    direction_multipliers: list[int] = [1, -1],
) -> list[StrategyResult]:
    df = series.df
    cost_model = _base_cost_model_for_spread(series, config)

    results: list[StrategyResult] = []
    for params in param_grid:
        # Use normalized spread (pct) for conditioning to reduce scale drift across regimes.
        z = compute_dte_conditioned_zscore(df, params=params, value_col="s_signal_pct")
        base_pos = positions_from_zscore(
            z,
            entry_z=params.entry_z,
            exit_z=params.exit_z,
            max_hold_days=params.max_hold_days,
        )

        for mult in direction_multipliers:
            signal = (mult * base_pos).astype("float64")
            bt = run_backtest_single_series(
                config=config,
                price_exec=df["s_exec"],
                signal=signal,
                contract_size=series.contract_size,
                cost_model=cost_model,
            )
            net_mean_daily = float(bt.returns.mean())
            if not np.isfinite(net_mean_daily) or net_mean_daily <= 0:
                continue
            if bt.max_drawdown > config.max_drawdown_limit:
                continue

            # Approximate number of trades as sign changes in non-zero segments.
            pos = bt.position.replace(0.0, np.nan)
            n_trades = int((pos.notna() & pos.shift(1).isna()).sum())

            results.append(
                StrategyResult(
                    name="mean_reversion",
                    spread=series.spread,
                    params={**asdict(params), "direction": mult},
                    n_days=int(len(bt.returns)),
                    n_trades=n_trades,
                    net_mean_daily=net_mean_daily,
                    net_sharpe_annual=float(bt.sharpe_annual),
                    max_drawdown=float(bt.max_drawdown),
                    turnover_daily=float(bt.turnover_daily),
                )
            )
    return results
