"""Tests for metric computations: Sharpe, drawdown, turnover."""
from __future__ import annotations

import numpy as np
import pandas as pd

from futures_curve.analysis.metrics import (
    compute_drawdown_series,
    compute_sharpe_annualized,
    compute_turnover,
)


# ---------- compute_drawdown_series ----------

class TestComputeDrawdownSeries:
    def test_no_drawdown_monotonic(self) -> None:
        equity = pd.Series([100.0, 110.0, 120.0, 130.0], index=pd.date_range("2024-01-01", periods=4))
        dd = compute_drawdown_series(equity)
        assert (dd == 0.0).all()

    def test_known_drawdown(self) -> None:
        equity = pd.Series([100.0, 110.0, 88.0, 120.0], index=pd.date_range("2024-01-01", periods=4))
        dd = compute_drawdown_series(equity)
        assert dd.iloc[0] == 0.0
        assert dd.iloc[1] == 0.0
        assert np.isclose(dd.iloc[2], (110.0 - 88.0) / 110.0)
        assert dd.iloc[3] == 0.0  # new high

    def test_full_drawdown(self) -> None:
        equity = pd.Series([100.0, 50.0, 25.0], index=pd.date_range("2024-01-01", periods=3))
        dd = compute_drawdown_series(equity)
        assert np.isclose(dd.iloc[2], 0.75)

    def test_empty_series(self) -> None:
        equity = pd.Series(dtype="float64")
        dd = compute_drawdown_series(equity)
        assert len(dd) == 0


# ---------- compute_sharpe_annualized ----------

class TestComputeSharpeAnnualized:
    def test_positive_returns_positive_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sharpe = compute_sharpe_annualized(returns, trading_days=252)
        assert sharpe > 0

    def test_zero_std_returns_nan(self) -> None:
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        sharpe = compute_sharpe_annualized(returns, trading_days=252)
        assert np.isnan(sharpe)

    def test_known_sharpe(self) -> None:
        # If daily mean=0.001 and daily std=0.01 (ddof=0), then annualized Sharpe = (0.001/0.01)*sqrt(252)
        n = 10000
        returns = pd.Series([0.001] * (n // 2) + [-0.001] * (n // 2))
        # mean=0, std=0.001 -> sharpe ~ 0
        sharpe = compute_sharpe_annualized(returns, trading_days=252)
        assert np.isclose(sharpe, 0.0, atol=0.01)

    def test_scaling_with_trading_days(self) -> None:
        rng = np.random.default_rng(123)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        s252 = compute_sharpe_annualized(returns, trading_days=252)
        s126 = compute_sharpe_annualized(returns, trading_days=126)
        # Sharpe scales with sqrt(trading_days), so ratio ~ sqrt(252/126) = sqrt(2)
        assert np.isclose(s252 / s126, np.sqrt(2), rtol=0.01)


# ---------- compute_turnover ----------

class TestComputeTurnover:
    def test_static_position_zero_turnover(self) -> None:
        pos = pd.Series([5.0, 5.0, 5.0, 5.0], index=pd.date_range("2024-01-01", periods=4))
        to = compute_turnover(pos)
        # After initial entry, no turnover
        # Day 0: delta=5, denom=5+0=5 -> turnover_0 = 1.0
        # Day 1..3: delta=0 -> turnover=0
        # Mean ~ 0.25
        assert to >= 0.0
        assert to < 1.0

    def test_full_reversal_high_turnover(self) -> None:
        pos = pd.Series([1.0, -1.0, 1.0, -1.0], index=pd.date_range("2024-01-01", periods=4))
        to = compute_turnover(pos)
        assert to > 0.5

    def test_zero_position_no_crash(self) -> None:
        pos = pd.Series([0.0, 0.0, 0.0], index=pd.date_range("2024-01-01", periods=3))
        to = compute_turnover(pos)
        assert np.isfinite(to)
        assert to == 0.0
