"""Backtester engine for running strategy simulations.

Coordinates between signal generation, execution, and performance tracking.
"""

from pathlib import Path
from typing import Optional, List, Type
import pandas as pd
import numpy as np
from datetime import datetime

from .execution import ExecutionEngine, ExecutionConfig
from .strategies import Strategy, Signal, get_strategy
from .performance import PerformanceAnalyzer, print_performance_report


class Backtester:
    """Main backtester engine."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[ExecutionConfig] = None,
    ):
        """Initialize backtester.

        Args:
            data: Market data with spreads and signals data
            config: Execution configuration
        """
        self.data = data.copy()
        # Ensure trade_date is datetime for consistent comparisons
        if "trade_date" in self.data.columns:
            self.data["trade_date"] = pd.to_datetime(self.data["trade_date"])
        # Add default bucket column if missing (for daily data)
        if "bucket" not in self.data.columns:
            self.data["bucket"] = 1
        self.config = config or ExecutionConfig()
        self.engine = ExecutionEngine(self.config)
        self.analyzer = PerformanceAnalyzer(self.config.initial_capital)
        try:
            from ..stage0.trading_calendar import TradingCalendar

            self.calendar = TradingCalendar("CMEGlobex_Metals")
        except Exception:
            self.calendar = None

    def run_strategy(
        self,
        strategy: Strategy,
        spread_col: str = "S1",
        stop_loss_usd: Optional[float] = None,
        max_holding_bdays: Optional[int] = None,
        auto_roll_on_contract_change: bool = True,
        allow_same_bucket_execution: bool = False,
        **signal_kwargs,
    ) -> dict:
        """Run a strategy backtest.

        Args:
            strategy: Strategy instance
            spread_col: Spread column to trade
            **signal_kwargs: Additional kwargs for signal generation

        Returns:
            Backtest results dictionary
        """
        data = self.data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])

        # Chronological ordering
        if "ts_end_utc" in data.columns:
            data["ts_end_utc"] = pd.to_datetime(data["ts_end_utc"])
            data = data.sort_values("ts_end_utc").reset_index(drop=True)
        else:
            data = data.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

        # Generate signals on the ordered data
        signals = strategy.generate_signals(data, **signal_kwargs)
        if not signals:
            return {"strategy": strategy.name, "status": "no_signals"}

        # Execution schedule: map execution row index -> list of signals
        schedule: dict[int, list[Signal]] = {}
        for sig in signals:
            sig_td = pd.Timestamp(sig.date)
            sig_bucket = int(sig.bucket)
            mask = (data["trade_date"] == sig_td) & (data["bucket"] == sig_bucket)
            if not mask.any():
                continue

            sig_idx = int(np.flatnonzero(mask.to_numpy())[0])
            exec_mode = sig.metadata.get("execution", "next")
            if exec_mode == "same":
                if not allow_same_bucket_execution:
                    raise ValueError(
                        "Same-bucket execution requested by signal metadata, but "
                        "allow_same_bucket_execution=False. This prevents accidental look-ahead bias."
                    )
                delay = 0
            else:
                delay = 1
            exec_idx = sig_idx + delay
            if exec_idx < 0 or exec_idx >= len(data):
                continue

            schedule.setdefault(exec_idx, []).append(sig)

        current_trade = None
        entry_date = None
        prev_row = None

        for i, row in data.iterrows():
            td = row["trade_date"].date() if isinstance(row["trade_date"], pd.Timestamp) else row["trade_date"]
            bucket = int(row.get("bucket", 1))
            price = row.get(spread_col)

            # 1) Execute scheduled signals (Mode A: skip if not fillable)
            for sig in schedule.get(i, []):
                action = sig.metadata.get("action", "")
                if price is None or pd.isna(price):
                    continue

                if action == "entry" and current_trade is None:
                    near = row.get(f"{spread_col}_near")
                    far = row.get(f"{spread_col}_far")
                    current_trade = self.engine.open_trade(
                        strategy=strategy.name,
                        trade_date=td,
                        bucket=bucket,
                        entry_price=float(price),
                        direction=int(sig.direction),
                        near_contract=near,
                        far_contract=far,
                    )
                    entry_date = td

                elif action == "exit" and current_trade is not None:
                    self.engine.close_trade(
                        current_trade,
                        exit_date=td,
                        exit_bucket=bucket,
                        exit_price=float(price),
                        reason="signal",
                    )
                    current_trade = None
                    entry_date = None

            # 2) Auto-roll when the traded spread identity changes while holding
            if auto_roll_on_contract_change and current_trade is not None and prev_row is not None:
                curr_near = row.get(f"{spread_col}_near")
                curr_far = row.get(f"{spread_col}_far")
                if (
                    pd.notna(curr_near)
                    and pd.notna(curr_far)
                    and (curr_near != current_trade.near_contract or curr_far != current_trade.far_contract)
                ):
                    prev_price = prev_row.get(spread_col)
                    if prev_price is not None and pd.notna(prev_price):
                        prev_td = prev_row["trade_date"].date() if isinstance(prev_row["trade_date"], pd.Timestamp) else prev_row["trade_date"]
                        prev_bucket = int(prev_row.get("bucket", 1))
                        self.engine.close_trade(
                            current_trade,
                            exit_date=prev_td,
                            exit_bucket=prev_bucket,
                            exit_price=float(prev_price),
                            reason="auto_roll",
                        )
                        # Open new leg at current price (skip if not fillable)
                        if price is not None and pd.notna(price):
                            current_trade = self.engine.open_trade(
                                strategy=strategy.name,
                                trade_date=td,
                                bucket=bucket,
                                entry_price=float(price),
                                direction=int(current_trade.direction),
                                near_contract=curr_near,
                                far_contract=curr_far,
                            )
                            entry_date = td
                        else:
                            current_trade = None
                            entry_date = None
                    else:
                        # Cannot mark/close old spread: force exit at current close if possible.
                        if price is not None and pd.notna(price):
                            self.engine.close_trade(
                                current_trade,
                                exit_date=td,
                                exit_bucket=bucket,
                                exit_price=float(price),
                                reason="auto_roll_unpriced",
                            )
                        current_trade = None
                        entry_date = None

            # 3) Risk controls (evaluated/filled on close)
            if current_trade is not None and price is not None and pd.notna(price):
                # Time stop (business days)
                if max_holding_bdays is not None and entry_date is not None and self.calendar is not None:
                    held = self.calendar.business_days_between(entry_date, td, include_start=True, include_end=True) - 1
                    if held >= int(max_holding_bdays):
                        self.engine.close_trade(
                            current_trade,
                            exit_date=td,
                            exit_bucket=bucket,
                            exit_price=float(price),
                            reason="time_stop",
                        )
                        current_trade = None
                        entry_date = None

                # Stop loss in USD (gross, excluding costs)
                if stop_loss_usd is not None and current_trade is not None:
                    delta_ticks = (float(price) - float(current_trade.entry_price)) / float(self.config.tick_size)
                    gross_pnl = delta_ticks * float(self.config.tick_value) * float(current_trade.contracts) * float(current_trade.direction)
                    if gross_pnl <= -abs(float(stop_loss_usd)):
                        self.engine.close_trade(
                            current_trade,
                            exit_date=td,
                            exit_bucket=bucket,
                            exit_price=float(price),
                            reason="stop_loss",
                        )
                        current_trade = None
                        entry_date = None

            prev_row = row

        # Close any open position at end (last valid price)
        if current_trade is not None:
            last = data.dropna(subset=[spread_col]).iloc[-1] if data[spread_col].notna().any() else data.iloc[-1]
            last_td = last["trade_date"].date() if isinstance(last["trade_date"], pd.Timestamp) else last["trade_date"]
            last_bucket = int(last.get("bucket", 1))
            last_price = last.get(spread_col)
            if last_price is not None and pd.notna(last_price):
                self.engine.close_trade(
                    current_trade,
                    exit_date=last_td,
                    exit_bucket=last_bucket,
                    exit_price=float(last_price),
                    reason="eod",
                )
            current_trade = None

        # Generate results
        trades = self.engine.get_trade_history()
        equity_curve = self.engine.get_equity_curve()
        report = self.analyzer.generate_report(trades, equity_curve, strategy.name)

        return {
            "strategy": strategy.name,
            "status": "success",
            "signals": len(signals),
            "trades": trades,
            "equity_curve": equity_curve,
            "report": report,
        }

    def run_multiple_strategies(
        self,
        strategies: List[Strategy],
        spread_col: str = "S1",
    ) -> dict:
        """Run multiple strategies for comparison.

        Args:
            strategies: List of Strategy instances
            spread_col: Spread column to trade

        Returns:
            Dictionary with results per strategy
        """
        results = {}

        for strategy in strategies:
            # Reset engine for each strategy
            self.engine = ExecutionEngine(self.config)

            result = self.run_strategy(strategy, spread_col=spread_col)
            results[strategy.name] = result

        return results

    def parameter_sweep(
        self,
        strategy_class: Type[Strategy],
        param_grid: dict,
        spread_col: str = "S1",
    ) -> pd.DataFrame:
        """Run parameter sweep for a strategy.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid: Dictionary of parameter -> list of values
            spread_col: Spread column to trade

        Returns:
            DataFrame with results for each parameter combination
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Reset engine
            self.engine = ExecutionEngine(self.config)

            # Create strategy with params
            strategy = strategy_class(**params)

            # Run backtest
            result = self.run_strategy(strategy, spread_col=spread_col)

            # Extract key metrics
            report = result.get("report", {})
            metrics = {
                **params,
                "total_trades": report.get("total_trades", 0),
                "win_rate": report.get("win_rate", 0),
                "total_pnl": report.get("total_pnl", 0),
                "sharpe_ratio": report.get("sharpe_ratio", 0),
                "max_drawdown_pct": report.get("max_drawdown_pct", 0),
            }
            results.append(metrics)

        return pd.DataFrame(results)


def run_backtest(
    data_dir: str | Path,
    symbol: str,
    strategy_name: str,
    strategy_params: Optional[dict] = None,
    execution_config: Optional[dict] = None,
    stop_loss_usd: Optional[float] = None,
    max_holding_bdays: Optional[int] = None,
    allow_same_bucket_execution: bool = False,
    auto_roll_on_contract_change: bool = True,
) -> dict:
    """Convenience function to run a backtest.

    Args:
        data_dir: Data directory path
        symbol: Commodity symbol
        strategy_name: Strategy name (dte, liquidity, hybrid, eom)
        strategy_params: Strategy parameters
        execution_config: Execution configuration

    Returns:
        Backtest results
    """
    from ..stage2.pipeline import read_spread_panel, read_curve_panel
    from ..stage3.eom_seasonality import build_eom_daily_dataset

    data_dir = Path(data_dir)

    # Load data
    spread_panel = read_spread_panel(data_dir, symbol)

    # For EOM strategy, need EOM labels
    if strategy_name.lower() == "eom":
        spread_panel = build_eom_daily_dataset(spread_panel, spread_col="S1")

    # For liquidity strategy, need volume share
    if strategy_name.lower() in ["liquidity", "hybrid"]:
        roll_path = data_dir / "roll_events" / symbol / "roll_shares.parquet"
        if roll_path.exists():
            roll_shares = pd.read_parquet(roll_path)
            if "frequency" in roll_shares.columns:
                roll_shares = roll_shares[roll_shares["frequency"] == "bucket"].copy()
            join_cols = ["trade_date"]
            if "bucket" in spread_panel.columns and "bucket" in roll_shares.columns:
                join_cols.append("bucket")
            cols = join_cols + [c for c in ["volume_share", "volume_share_smooth"] if c in roll_shares.columns]
            spread_panel = spread_panel.merge(roll_shares[cols], on=join_cols, how="left")

    # Create config
    config = ExecutionConfig(**(execution_config or {}))

    # Create strategy
    strategy = get_strategy(strategy_name, **(strategy_params or {}))

    # Run backtest
    backtester = Backtester(spread_panel, config)
    result = backtester.run_strategy(
        strategy,
        spread_col='S1',
        stop_loss_usd=stop_loss_usd,
        max_holding_bdays=max_holding_bdays,
        auto_roll_on_contract_change=auto_roll_on_contract_change,
        allow_same_bucket_execution=allow_same_bucket_execution,
    )

    return result
