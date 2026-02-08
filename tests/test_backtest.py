"""Tests for the backtest engine, cost model, and position sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_curve.analysis.backtest import BacktestResult, run_backtest_single_series
from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.costs import CostModel
from futures_curve.analysis.risk import ewma_volatility, position_from_vol_target


# ---------- CostModel ----------

class TestCostModel:
    def test_cost_per_unit_change_positive(self) -> None:
        cm = CostModel(dollars_per_tick=12.50, legs=2, ticks_per_leg_per_side=1.0)
        cost = cm.cost_per_unit_change(1.0)
        assert cost == 25.0  # 1 * 2 * 1.0 * 12.50

    def test_cost_zero_for_no_change(self) -> None:
        cm = CostModel(dollars_per_tick=12.50, legs=2, ticks_per_leg_per_side=1.0)
        assert cm.cost_per_unit_change(0.0) == 0.0

    def test_cost_symmetric(self) -> None:
        cm = CostModel(dollars_per_tick=10.0, legs=2, ticks_per_leg_per_side=0.5)
        assert cm.cost_per_unit_change(3.0) == cm.cost_per_unit_change(-3.0)

    def test_cost_scales_linearly(self) -> None:
        cm = CostModel(dollars_per_tick=10.0, legs=2, ticks_per_leg_per_side=1.0)
        assert np.isclose(cm.cost_per_unit_change(5.0), 5 * cm.cost_per_unit_change(1.0))


# ---------- ewma_volatility ----------

class TestEwmaVolatility:
    def test_constant_pnl_zero_vol(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50)
        pnl = pd.Series(0.0, index=idx)
        vol = ewma_volatility(pnl, halflife_days=10, min_periods=5)
        assert (vol.dropna() == 0.0).all()

    def test_volatile_pnl_positive_vol(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=100)
        pnl = pd.Series(rng.normal(0, 100, 100), index=idx)
        vol = ewma_volatility(pnl, halflife_days=10, min_periods=10)
        finite = vol.dropna()
        assert len(finite) > 0
        assert (finite > 0).all()


# ---------- position_from_vol_target ----------

class TestPositionFromVolTarget:
    def test_direction_preserved(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5)
        signal = pd.Series([1.0, -1.0, 0.0, 1.0, -1.0], index=idx)
        vol = pd.Series(10.0, index=idx)
        pos = position_from_vol_target(
            signal=signal, vol_per_contract=vol, target_dollar_vol=50.0, allow_fractional=True,
        )
        # Signs should match signal
        for i in range(len(pos)):
            if signal.iloc[i] > 0:
                assert pos.iloc[i] > 0
            elif signal.iloc[i] < 0:
                assert pos.iloc[i] < 0
            else:
                assert pos.iloc[i] == 0.0

    def test_zero_vol_gives_zero_position(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3)
        signal = pd.Series([1.0, 1.0, 1.0], index=idx)
        vol = pd.Series([0.0, 0.0, 0.0], index=idx)
        pos = position_from_vol_target(
            signal=signal, vol_per_contract=vol, target_dollar_vol=10.0, allow_fractional=True,
        )
        assert (pos == 0.0).all()

    def test_rounding_when_not_fractional(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3)
        signal = pd.Series([1.0, 1.0, 1.0], index=idx)
        vol = pd.Series([3.0, 3.0, 3.0], index=idx)
        pos = position_from_vol_target(
            signal=signal, vol_per_contract=vol, target_dollar_vol=10.0, allow_fractional=False,
        )
        # 1 * 10/3 = 3.33 -> rounds to 3
        assert (pos == 3.0).all()


# ---------- run_backtest_single_series ----------

class TestRunBacktestSingleSeries:
    @pytest.fixture
    def simple_setup(self) -> dict:
        n = 50
        idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
        price = pd.Series(100.0 + np.arange(n) * 0.01, index=idx)
        signal = pd.Series(1.0, index=idx)
        config = AnalysisConfig(
            initial_capital=100_000.0,
            vol_target_annual=0.10,
            allow_fractional_contracts=True,
            ewma_halflife_days=10,
            ewma_min_periods=5,
        )
        cm = CostModel(dollars_per_tick=12.50, legs=2, ticks_per_leg_per_side=1.0)
        return {
            "config": config,
            "price_exec": price,
            "signal": signal,
            "contract_size": 25000.0,
            "cost_model": cm,
        }

    def test_returns_backtest_result(self, simple_setup: dict) -> None:
        bt = run_backtest_single_series(**simple_setup)
        assert isinstance(bt, BacktestResult)
        assert len(bt.equity) == len(simple_setup["price_exec"])
        assert len(bt.returns) == len(simple_setup["price_exec"])
        assert len(bt.pnl_net) == len(simple_setup["price_exec"])

    def test_equity_starts_at_initial_capital(self, simple_setup: dict) -> None:
        bt = run_backtest_single_series(**simple_setup)
        # Equity at first point is initial_capital + first day pnl_net (which is ~0 due to shift)
        first_pnl = bt.pnl_net.iloc[0]
        assert np.isclose(bt.equity.iloc[0], 100_000.0 + first_pnl)

    def test_costs_are_nonnegative(self, simple_setup: dict) -> None:
        bt = run_backtest_single_series(**simple_setup)
        assert (bt.costs >= 0).all()

    def test_pnl_net_equals_gross_minus_costs(self, simple_setup: dict) -> None:
        bt = run_backtest_single_series(**simple_setup)
        diff = (bt.pnl_gross - bt.costs) - bt.pnl_net
        assert np.allclose(diff.values, 0.0, atol=1e-10)

    def test_execution_lag(self) -> None:
        """Signal on day t should not affect position until day t+1."""
        n = 10
        idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
        price = pd.Series(100.0, index=idx)  # flat price -> no PnL from price movement
        # Signal is 0 everywhere except day 0 (signal=1)
        signal = pd.Series(0.0, index=idx)
        signal.iloc[0] = 1.0
        config = AnalysisConfig(
            initial_capital=100_000.0,
            vol_target_annual=0.10,
            allow_fractional_contracts=True,
            ewma_halflife_days=5,
            ewma_min_periods=2,
        )
        cm = CostModel(dollars_per_tick=12.50, legs=2, ticks_per_leg_per_side=1.0)
        bt = run_backtest_single_series(
            config=config,
            price_exec=price,
            signal=signal,
            contract_size=25000.0,
            cost_model=cm,
        )
        # Day 0: signal_for_exec = shift(1) of signal -> NaN -> position = 0
        # Day 1: signal_for_exec = signal[0] = 1.0 -> position > 0
        assert bt.position.iloc[0] == 0.0  # no position on first day

    def test_zero_signal_no_trades(self) -> None:
        n = 20
        idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
        price = pd.Series(100.0, index=idx)
        signal = pd.Series(0.0, index=idx)
        config = AnalysisConfig(initial_capital=100_000.0, ewma_halflife_days=5, ewma_min_periods=2)
        cm = CostModel(dollars_per_tick=12.50, legs=2, ticks_per_leg_per_side=1.0)
        bt = run_backtest_single_series(
            config=config, price_exec=price, signal=signal, contract_size=25000.0, cost_model=cm,
        )
        assert (bt.position == 0.0).all()
        assert (bt.costs == 0.0).all()
        assert (bt.pnl_gross == 0.0).all()
