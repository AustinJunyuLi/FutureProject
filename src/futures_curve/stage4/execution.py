"""Execution model for backtesting.

Implements causal signal handling, transaction costs, and trade mechanics.

Key requirements:
- Signals use only data available at signal time (no look-ahead)
- Entry/exit are executed on the next observation (no same-observation fills by default)
- Explicit transaction costs (slippage + commissions)
- Roll mechanics for spread positions
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List
import pandas as pd
import numpy as np


@dataclass
class Trade:
    """Record of a single trade."""

    trade_id: int
    strategy: str
    trade_date: date
    bucket: int
    entry_price: float
    exit_price: Optional[float] = None
    exit_date: Optional[date] = None
    exit_bucket: Optional[int] = None
    direction: int = 1  # 1 = long spread, -1 = short spread
    contracts: int = 1
    near_contract: Optional[str] = None
    far_contract: Optional[str] = None
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    pnl: Optional[float] = None
    status: str = "open"  # open, closed, rolled
    exit_reason: Optional[str] = None


@dataclass
class ExecutionConfig:
    """Configuration for execution model."""

    # Slippage is measured in ticks *per fill* (per leg, per side).
    slippage_ticks: float = 1.0
    tick_size: float = 0.0005  # HG copper tick size
    tick_value: float = 12.50  # Dollars per tick, per contract
    commission_per_contract: float = 2.50  # Per contract per side
    initial_capital: float = 100000


class ExecutionEngine:
    """Execute trades with realistic mechanics."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize engine.

        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()
        self.trade_counter = 0
        self.trades: List[Trade] = []
        self.capital = self.config.initial_capital

    def calculate_slippage(self, contracts: int = 1) -> float:
        """Calculate slippage cost for one side (entry OR exit) of a spread trade.

        Args:
            contracts: Number of contracts

        Returns:
            Slippage cost in dollars
        """
        # A calendar spread has 2 legs, so 2 fills per side.
        fills_per_side = 2
        return float(self.config.slippage_ticks) * float(self.config.tick_value) * float(contracts) * fills_per_side

    def calculate_commission(self, contracts: int = 1) -> float:
        """Calculate commission cost for one side (entry OR exit) of a spread trade.

        Args:
            contracts: Number of contracts

        Returns:
            Commission in dollars
        """
        fills_per_side = 2
        return float(self.config.commission_per_contract) * float(contracts) * fills_per_side

    def open_trade(
        self,
        strategy: str,
        trade_date: date,
        bucket: int,
        entry_price: float,
        direction: int = 1,
        contracts: int = 1,
        near_contract: Optional[str] = None,
        far_contract: Optional[str] = None,
    ) -> Trade:
        """Open a new trade.

        Args:
            strategy: Strategy name
            trade_date: Trade date
            bucket: Entry bucket (signal was in previous bucket)
            entry_price: Spread entry price
            direction: 1 for long spread, -1 for short
            contracts: Number of contracts
            near_contract: Near leg contract code
            far_contract: Far leg contract code

        Returns:
            Trade object
        """
        self.trade_counter += 1

        slippage = self.calculate_slippage(contracts)
        commission = self.calculate_commission(contracts)

        trade = Trade(
            trade_id=self.trade_counter,
            strategy=strategy,
            trade_date=trade_date,
            bucket=bucket,
            entry_price=entry_price,
            direction=direction,
            contracts=contracts,
            near_contract=near_contract,
            far_contract=far_contract,
            slippage_cost=slippage,
            commission_cost=commission,
            status="open",
        )

        self.trades.append(trade)
        return trade

    def close_trade(
        self,
        trade: Trade,
        exit_date: date,
        exit_bucket: int,
        exit_price: float,
        reason: str | None = None,
    ) -> float:
        """Close an existing trade.

        Args:
            trade: Trade to close
            exit_date: Exit date
            exit_bucket: Exit bucket
            exit_price: Spread exit price

        Returns:
            PnL in dollars
        """
        trade.exit_date = exit_date
        trade.exit_bucket = exit_bucket
        trade.exit_price = exit_price

        # Add exit-side costs
        trade.slippage_cost += self.calculate_slippage(trade.contracts)
        trade.commission_cost += self.calculate_commission(trade.contracts)

        # Gross PnL (USD): Δspread (price units) -> ticks -> USD
        delta_ticks = (exit_price - trade.entry_price) / float(self.config.tick_size)
        gross_pnl = delta_ticks * float(self.config.tick_value) * float(trade.contracts) * float(trade.direction)

        total_costs = float(trade.slippage_cost) + float(trade.commission_cost)
        trade.pnl = gross_pnl - total_costs
        trade.status = "closed"
        trade.exit_reason = reason

        self.capital += trade.pnl

        return trade.pnl

    def roll_trade(
        self,
        trade: Trade,
        roll_date: date,
        roll_bucket: int,
        old_spread_price: float,
        new_spread_price: float,
        new_near: str,
        new_far: str,
    ) -> Trade:
        """Roll a trade to new contracts.

        Args:
            trade: Trade to roll
            roll_date: Roll date
            roll_bucket: Roll bucket
            old_spread_price: Price to close old spread
            new_spread_price: Price to open new spread
            new_near: New near contract
            new_far: New far contract

        Returns:
            New trade object
        """
        # Close old trade
        self.close_trade(trade, roll_date, roll_bucket, old_spread_price, reason="roll")
        trade.status = "rolled"

        # Open new trade
        new_trade = self.open_trade(
            strategy=trade.strategy,
            trade_date=roll_date,
            bucket=roll_bucket,
            entry_price=new_spread_price,
            direction=trade.direction,
            contracts=trade.contracts,
            near_contract=new_near,
            far_contract=new_far,
        )

        return new_trade

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [t for t in self.trades if t.status == "open"]

    def get_trade_history(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "trade_id": t.trade_id,
                "strategy": t.strategy,
                "entry_date": t.trade_date,
                "entry_bucket": t.bucket,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_bucket": t.exit_bucket,
                "exit_price": t.exit_price,
                "direction": t.direction,
                "contracts": t.contracts,
                "near_contract": t.near_contract,
                "far_contract": t.far_contract,
                "slippage_cost": t.slippage_cost,
                "commission_cost": t.commission_cost,
                "pnl": t.pnl,
                "status": t.status,
                "exit_reason": t.exit_reason,
            })

        return pd.DataFrame(records)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve from closed trades."""
        trades_df = self.get_trade_history()
        closed = trades_df[trades_df["status"].isin(["closed", "rolled"])].copy()

        if len(closed) == 0:
            return pd.DataFrame()

        closed = closed.sort_values("exit_date")
        closed["cumulative_pnl"] = closed["pnl"].cumsum()
        closed["equity"] = self.config.initial_capital + closed["cumulative_pnl"]

        return closed[["exit_date", "pnl", "cumulative_pnl", "equity"]]


def get_execution_price(
    df: pd.DataFrame,
    signal_date: date,
    signal_bucket: int,
    price_col: str = "S1",
    delay: int = 1,
) -> tuple[Optional[date], Optional[int], Optional[float]]:
    """Get execution price at (signal_row + delay) in *chronological* order.

    Prefer `ts_end_utc` ordering if present; otherwise fall back to
    (trade_date, bucket) ordering. This avoids assuming bucket IDs are
    sequential/chronological.
    """
    data = df.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])

    if "ts_end_utc" in data.columns:
        data["ts_end_utc"] = pd.to_datetime(data["ts_end_utc"])
        data = data.sort_values("ts_end_utc").reset_index(drop=True)
    else:
        data = data.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

    sig_td = pd.Timestamp(signal_date)
    mask = (data["trade_date"] == sig_td) & (data["bucket"] == signal_bucket)
    if not mask.any():
        return None, None, None

    i = int(np.flatnonzero(mask.to_numpy())[0])
    j = i + int(delay)
    if j < 0 or j >= len(data):
        return None, None, None

    row = data.iloc[j]
    td = row["trade_date"]
    if isinstance(td, pd.Timestamp):
        td = td.date()
    return td, int(row["bucket"]), row.get(price_col)
