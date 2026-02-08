"""Tests for parameter scan functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.data import SpreadDailySeries
from futures_curve.analysis.scan import (
    scan_carry_strategy,
    scan_mean_reversion_strategy,
)
from futures_curve.analysis.walk_forward import build_cost_model
from futures_curve.analysis.strategies.mean_reversion import MeanReversionParams
from futures_curve.analysis.types import StrategyResult


def _make_synthetic_series(n: int = 500, seed: int = 42) -> SpreadDailySeries:
    """Build a synthetic SpreadDailySeries for scan tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, freq="B")
    price_drift = 0.001  # slight upward drift
    prices = 100.0 + np.cumsum(rng.normal(price_drift, 0.5, n))
    s_signal = rng.normal(0.5, 0.1, n)
    fnear = prices
    ffar = prices + s_signal

    df = pd.DataFrame(
        {
            "s_signal": s_signal,
            "s_signal_pct": s_signal / fnear,
            "s_exec": s_signal + rng.normal(0, 0.01, n),
            "fnear_signal": fnear,
            "ffar_signal": ffar,
            "near_dte_bdays": 20.0,
            "far_dte_bdays": 42.0,
            "near_contract": "HGH24",
            "far_contract": "HGJ24",
        },
        index=idx,
    )
    return SpreadDailySeries(
        spread="S1", df=df, contract_size=25000.0, dollars_per_tick=12.50,
    )


class TestBuildCostModel:
    def test_returns_cost_model(self) -> None:
        series = _make_synthetic_series(n=10)
        config = AnalysisConfig()
        cm = build_cost_model(series, config)
        assert cm.dollars_per_tick == 12.50
        assert cm.legs == 2


class TestScanCarryStrategy:
    def test_returns_list_of_strategy_results(self) -> None:
        series = _make_synthetic_series(n=500, seed=42)
        config = AnalysisConfig(
            initial_capital=100_000.0,
            max_drawdown_limit=0.50,  # relaxed for synthetic data
            ewma_halflife_days=10,
            ewma_min_periods=5,
        )
        results = scan_carry_strategy(
            config=config,
            series=series,
            thresholds=[0.0, 0.05],
            direction_multipliers=[1, -1],
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, StrategyResult)
            assert r.name == "carry"
            assert r.spread == "S1"
            assert r.net_sharpe_annual > 0  # filter keeps only positive

    def test_empty_with_impossible_threshold(self) -> None:
        series = _make_synthetic_series(n=100)
        config = AnalysisConfig(
            initial_capital=100_000.0,
            max_drawdown_limit=0.001,  # extremely tight
            ewma_halflife_days=10,
            ewma_min_periods=5,
        )
        results = scan_carry_strategy(
            config=config, series=series, thresholds=[100.0],
        )
        assert results == []


class TestScanMeanReversionStrategy:
    def test_returns_list_of_strategy_results(self) -> None:
        series = _make_synthetic_series(n=500, seed=123)
        config = AnalysisConfig(
            initial_capital=100_000.0,
            max_drawdown_limit=0.50,
            ewma_halflife_days=10,
            ewma_min_periods=5,
        )
        grid = [
            MeanReversionParams(dte_bin_size=5, lookback_days=100, entry_z=1.0, exit_z=0.2, max_hold_days=20),
        ]
        results = scan_mean_reversion_strategy(
            config=config, series=series, param_grid=grid, direction_multipliers=[1, -1],
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, StrategyResult)
            assert r.name == "mean_reversion"

    def test_empty_with_extreme_params(self) -> None:
        series = _make_synthetic_series(n=100)
        config = AnalysisConfig(
            initial_capital=100_000.0,
            max_drawdown_limit=0.001,
            ewma_halflife_days=10,
            ewma_min_periods=5,
        )
        grid = [MeanReversionParams(dte_bin_size=5, lookback_days=50, entry_z=100.0, exit_z=99.0)]
        results = scan_mean_reversion_strategy(
            config=config, series=series, param_grid=grid,
        )
        assert results == []
