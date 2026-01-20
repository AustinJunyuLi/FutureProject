"""Tests for Stage 4: Backtesting."""

import pytest
from datetime import date
import pandas as pd
import numpy as np

from futures_curve.stage4.execution import (
    ExecutionEngine,
    ExecutionConfig,
    get_execution_price,
)
from futures_curve.stage4.backtester import Backtester
from futures_curve.stage4.strategies import (
    Strategy,
    DTEStrategy,
    PreExpiryStrategy,
    Signal,
    get_strategy,
)
from futures_curve.stage4.performance import PerformanceAnalyzer


class TestExecutionEngine:
    """Tests for execution engine."""

    def test_open_close_trade(self):
        """Test opening and closing a trade."""
        config = ExecutionConfig(
            slippage_ticks=1,
            tick_size=0.01,
            tick_value=10.0,
            commission_per_contract=2.50,
        )
        engine = ExecutionEngine(config)

        # Open trade
        trade = engine.open_trade(
            strategy="test",
            trade_date=date(2024, 1, 15),
            bucket=1,
            entry_price=100.0,
            direction=1,
        )

        assert trade.status == "open"
        assert trade.entry_price == 100.0

        # Close trade at higher price (profit for long)
        pnl = engine.close_trade(
            trade,
            exit_date=date(2024, 1, 16),
            exit_bucket=1,
            exit_price=100.10,  # +0.10 spread gain
        )

        assert trade.status == "closed"
        assert trade.pnl is not None

    def test_transaction_costs(self):
        """Test transaction costs calculation."""
        config = ExecutionConfig(
            slippage_ticks=2,
            tick_size=0.01,
            tick_value=10.0,
            commission_per_contract=5.0,
        )
        engine = ExecutionEngine(config)

        # Slippage (per side) = 2 ticks/fill * $10/tick * 1 contract * 2 legs = $40
        slippage = engine.calculate_slippage(contracts=1)
        assert abs(slippage - 40.0) < 1e-6

        # Commission (per side) = 5.0 * 1 * 2 legs = 10.0
        commission = engine.calculate_commission(contracts=1)
        assert abs(commission - 10.0) < 1e-6

    def test_equity_tracking(self):
        """Test equity curve tracking."""
        config = ExecutionConfig(initial_capital=100000)
        engine = ExecutionEngine(config)

        # Open and close profitable trade
        trade = engine.open_trade(
            strategy="test",
            trade_date=date(2024, 1, 15),
            bucket=1,
            entry_price=100.0,
            direction=1,
        )

        # Close with profit
        engine.close_trade(
            trade,
            exit_date=date(2024, 1, 16),
            exit_bucket=1,
            exit_price=100.10,
        )

        # Capital should have changed
        # (depends on tick_size and tick_value)
        assert engine.capital != config.initial_capital


class TestCausalExecution:
    """Tests for causal (no look-ahead) execution."""

    def test_next_bucket_same_day(self):
        """Test getting next bucket price same day."""
        df = pd.DataFrame({
            "trade_date": pd.to_datetime([date(2024, 1, 15)] * 3),
            "bucket": [1, 2, 3],
            "S1": [0.10, 0.11, 0.12],
        })

        next_date, next_bucket, next_price = get_execution_price(
            df, date(2024, 1, 15), signal_bucket=1, price_col="S1", delay=1
        )

        assert next_date == date(2024, 1, 15)
        assert next_bucket == 2
        assert next_price == 0.11

    def test_next_bucket_next_day(self):
        """Test getting next bucket price on next day."""
        df = pd.DataFrame({
            "trade_date": pd.to_datetime([date(2024, 1, 15), date(2024, 1, 16)]),
            "bucket": [10, 1],  # Last bucket of day 1, first of day 2
            "S1": [0.10, 0.11],
        })

        next_date, next_bucket, next_price = get_execution_price(
            df, date(2024, 1, 15), signal_bucket=10, price_col="S1", delay=1
        )

        assert next_date == date(2024, 1, 16)
        assert next_bucket == 1
        assert next_price == 0.11

    def test_no_same_bucket_execution(self):
        """Ensure we never execute in the same bucket as signal."""
        df = pd.DataFrame({
            "trade_date": pd.to_datetime([date(2024, 1, 15)] * 2),
            "bucket": [1, 2],
            "S1": [0.10, 0.11],
        })

        # Signal in bucket 1, should execute in bucket 2
        next_date, next_bucket, _ = get_execution_price(
            df, date(2024, 1, 15), signal_bucket=1, price_col="S1", delay=1
        )

        assert next_bucket != 1  # Must be different bucket


class TestStrategies:
    """Tests for trading strategies."""

    def test_dte_strategy_entry_exit(self):
        """Test DTE strategy generates entry and exit signals."""
        strategy = DTEStrategy(dte_entry=15, dte_exit=5)

        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "bucket": [1] * 20,
            "F1_dte_bdays": list(range(20, 0, -1)),  # 20 down to 1
        })

        signals = strategy.generate_signals(df)

        # Should have entry around DTE=15 and exit around DTE=5
        entries = [s for s in signals if s.metadata.get("action") == "entry"]
        exits = [s for s in signals if s.metadata.get("action") == "exit"]

        assert len(entries) >= 1
        assert len(exits) >= 1

    def test_pre_expiry_strategy(self):
        """Test pre-expiry strategy generates entry/exit signals."""
        strategy = PreExpiryStrategy(entry_dte=5, exit_dte=1)

        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=7),
                "bucket": [1] * 7,
                "F1_dte_bdays": [7, 6, 5, 4, 3, 2, 1],
            }
        )

        signals = strategy.generate_signals(df)

        entries = [s for s in signals if s.metadata.get("action") == "entry"]
        exits = [s for s in signals if s.metadata.get("action") == "exit"]

        assert len(entries) >= 1
        assert len(exits) >= 1

        # Entry occurs when within the window (<= entry_dte and > exit_dte)
        assert any(s.metadata.get("action") == "entry" and int(s.metadata.get("dte")) == 5 for s in signals)
        # Exit occurs when at/through the exit threshold
        assert any(s.metadata.get("action") == "exit" and int(s.metadata.get("dte")) == 1 for s in signals)

        # Guard against same-bucket execution in research backtests
        assert all(s.metadata.get("execution", "next") != "same" for s in signals)

    def test_strategy_factory(self):
        """Test strategy factory function."""
        strategy = get_strategy("dte", dte_entry=20, dte_exit=10)
        assert isinstance(strategy, DTEStrategy)

        strategy = get_strategy("pre_expiry", entry_dte=5, exit_dte=1)
        assert isinstance(strategy, PreExpiryStrategy)


class TestPerformanceAnalyzer:
    """Tests for performance analysis."""

    def test_trade_stats(self):
        """Test trade statistics calculation."""
        analyzer = PerformanceAnalyzer()

        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -30, 150],
            "status": ["closed"] * 5,
        })

        stats = analyzer.compute_trade_stats(trades)

        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert abs(stats["win_rate"] - 60.0) < 1e-6
        assert abs(stats["total_pnl"] - 370.0) < 1e-6

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        analyzer = PerformanceAnalyzer()

        # All winners
        trades = pd.DataFrame({
            "pnl": [100, 200, 300],
            "status": ["closed"] * 3,
        })
        stats = analyzer.compute_trade_stats(trades)
        assert stats["win_rate"] == 100.0

        # All losers
        trades = pd.DataFrame({
            "pnl": [-100, -200, -300],
            "status": ["closed"] * 3,
        })
        stats = analyzer.compute_trade_stats(trades)
        assert stats["win_rate"] == 0.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        analyzer = PerformanceAnalyzer()

        trades = pd.DataFrame({
            "pnl": [200, -100],  # Total wins = 200, total losses = 100
            "status": ["closed"] * 2,
        })

        stats = analyzer.compute_trade_stats(trades)
        assert abs(stats["profit_factor"] - 2.0) < 1e-6


def test_backtester_auto_roll_unpriced_closes_trade() -> None:
    """Regression: contract-change with missing prices must not orphan open trades."""

    class OneShotEntryStrategy(Strategy):
        def __init__(self):
            super().__init__("one_shot_entry")

        def generate_signals(self, data: pd.DataFrame, **kwargs):  # type: ignore[override]
            first = data.iloc[0]
            return [
                Signal(
                    date=pd.to_datetime(first["trade_date"]).date(),
                    bucket=int(first["bucket"]),
                    direction=1,
                    metadata={"action": "entry"},
                )
            ]

    df = pd.DataFrame(
        [
            # signal row (executes next observation)
            {"trade_date": "2024-01-01", "bucket": 1, "S1": 0.10, "S1_near": "A", "S1_far": "B"},
            # execution row (opens position)
            {"trade_date": "2024-01-01", "bucket": 2, "S1": 0.12, "S1_near": "A", "S1_far": "B"},
            # temporarily unpriced but still same spread identity
            {"trade_date": "2024-01-01", "bucket": 3, "S1": float("nan"), "S1_near": "A", "S1_far": "B"},
            # contract changes while unpriced -> must close using last priced mark
            {"trade_date": "2024-01-01", "bucket": 4, "S1": float("nan"), "S1_near": "C", "S1_far": "D"},
        ]
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    bt = Backtester(
        df,
        config=ExecutionConfig(slippage_ticks=1, tick_size=0.01, tick_value=10.0, commission_per_contract=1.0),
    )
    res = bt.run_strategy(OneShotEntryStrategy(), spread_col="S1", auto_roll_on_contract_change=True)
    assert res["status"] == "success"

    trades = res["trades"]
    assert len(trades) == 1
    assert set(trades["status"]) == {"closed"}
    assert trades.iloc[0]["exit_reason"] == "auto_roll_unpriced"
