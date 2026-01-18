"""Performance analytics for backtesting.

Computes strategy performance metrics, risk statistics, and generates reports.
"""

from typing import Optional
import pandas as pd
import numpy as np


class PerformanceAnalyzer:
    """Analyze backtest performance."""

    def __init__(self, initial_capital: float = 100000):
        """Initialize analyzer.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital

    def compute_trade_stats(self, trades: pd.DataFrame) -> dict:
        """Compute trade-level statistics.

        Args:
            trades: DataFrame with trade records (pnl column required)

        Returns:
            Dictionary with trade statistics
        """
        if len(trades) == 0:
            return {}

        closed = trades[trades["status"].isin(["closed", "rolled"])]
        if len(closed) == 0:
            return {}

        pnl = closed["pnl"]
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]

        stats = {
            "total_trades": len(closed),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(closed) * 100,
            "total_pnl": pnl.sum(),
            "avg_pnl": pnl.mean(),
            "avg_win": winners.mean() if len(winners) > 0 else 0,
            "avg_loss": losers.mean() if len(losers) > 0 else 0,
            "largest_win": winners.max() if len(winners) > 0 else 0,
            "largest_loss": losers.min() if len(losers) > 0 else 0,
            "profit_factor": abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else np.inf,
        }

        # Transaction costs
        if "slippage_cost" in closed.columns:
            stats["total_slippage"] = closed["slippage_cost"].sum()
        if "commission_cost" in closed.columns:
            stats["total_commission"] = closed["commission_cost"].sum()
        stats["total_costs"] = stats.get("total_slippage", 0) + stats.get("total_commission", 0)

        return stats

    def compute_return_stats(
        self,
        equity_curve: pd.DataFrame,
        periods_per_year: Optional[float] = None,
        timestamp_col: str = "exit_date",
    ) -> dict:
        """Compute return-based statistics.

        Args:
            equity_curve: DataFrame with equity column
            periods_per_year: Annualization factor. If None, infer from timestamp spacing.
            timestamp_col: Timestamp column used for spacing inference.

        Returns:
            Dictionary with return statistics
        """
        if len(equity_curve) < 2:
            return {}

        eq = equity_curve.copy()
        if timestamp_col in eq.columns:
            ts = pd.to_datetime(eq[timestamp_col])
        else:
            # Best-effort fallback
            ts = pd.to_datetime(eq.index)

        # Calculate returns
        equity = eq["equity"].values
        returns = np.diff(equity) / equity[:-1]

        # Handle empty or invalid returns
        if len(returns) == 0 or np.all(np.isnan(returns)):
            return {}

        # Infer annualization factor from average timestamp spacing
        if periods_per_year is None:
            deltas = np.diff(ts.values.astype("datetime64[ns]"))
            # Convert to days (float)
            days = deltas.astype("timedelta64[s]").astype("float64") / 86400.0
            avg_days = float(np.nanmean(days)) if len(days) else np.nan
            periods_per_year = 365.25 / avg_days if avg_days and avg_days > 0 else 252.0

        periods_per_year = float(periods_per_year)

        # Basic stats
        total_return = (equity[-1] / self.initial_capital - 1) * 100
        avg_return = np.nanmean(returns) * 100
        volatility = np.nanstd(returns) * np.sqrt(periods_per_year) * 100

        stats = {
            "total_return_pct": total_return,
            "avg_return_pct": avg_return,
            "annualized_volatility_pct": volatility,
            "final_equity": equity[-1],
        }

        # Sharpe ratio (assuming 0% risk-free)
        if volatility > 0:
            annualized_return = np.nanmean(returns) * periods_per_year
            annualized_vol = np.nanstd(returns) * np.sqrt(periods_per_year)
            stats["sharpe_ratio"] = annualized_return / annualized_vol
        else:
            stats["sharpe_ratio"] = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        stats["max_drawdown_pct"] = np.min(drawdown) * 100

        # Calmar ratio
        if stats["max_drawdown_pct"] != 0:
            years = len(equity) / periods_per_year
            cagr = ((equity[-1] / self.initial_capital) ** (1 / years) - 1) if years > 0 else 0
            stats["calmar_ratio"] = cagr / abs(stats["max_drawdown_pct"] / 100)
        else:
            stats["calmar_ratio"] = np.inf

        return stats

    def compute_monthly_returns(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute monthly return breakdown.

        Args:
            trades: DataFrame with trade records

        Returns:
            DataFrame with monthly returns
        """
        if len(trades) == 0:
            return pd.DataFrame()

        closed = trades[trades["status"].isin(["closed", "rolled"])].copy()
        if len(closed) == 0 or "exit_date" not in closed.columns:
            return pd.DataFrame()

        closed["exit_date"] = pd.to_datetime(closed["exit_date"])
        closed["year"] = closed["exit_date"].dt.year
        closed["month"] = closed["exit_date"].dt.month

        monthly = closed.groupby(["year", "month"]).agg({
            "pnl": ["sum", "count", lambda x: (x > 0).sum()],
        })
        monthly.columns = ["total_pnl", "num_trades", "winning_trades"]
        monthly = monthly.reset_index()
        monthly["win_rate"] = monthly["winning_trades"] / monthly["num_trades"] * 100

        return monthly

    def generate_report(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        strategy_name: str = "Strategy",
    ) -> dict:
        """Generate comprehensive performance report.

        Args:
            trades: DataFrame with trade records
            equity_curve: DataFrame with equity curve
            strategy_name: Strategy name for report

        Returns:
            Dictionary with full report
        """
        report = {
            "strategy": strategy_name,
            "initial_capital": self.initial_capital,
        }

        # Trade statistics
        trade_stats = self.compute_trade_stats(trades)
        report.update(trade_stats)

        # Return statistics
        return_stats = self.compute_return_stats(equity_curve)
        report.update(return_stats)

        # Monthly breakdown
        monthly = self.compute_monthly_returns(trades)
        report["monthly_returns"] = monthly

        return report


def print_performance_report(report: dict) -> None:
    """Print formatted performance report.

    Args:
        report: Report dictionary from generate_report
    """
    print(f"\n{'='*60}")
    print(f"Performance Report: {report.get('strategy', 'Unknown')}")
    print(f"{'='*60}")

    print(f"\nCapital: ${report.get('initial_capital', 0):,.2f} -> ${report.get('final_equity', 0):,.2f}")
    print(f"Total Return: {report.get('total_return_pct', 0):.2f}%")

    print(f"\n--- Trade Statistics ---")
    print(f"Total Trades: {report.get('total_trades', 0)}")
    print(f"Win Rate: {report.get('win_rate', 0):.1f}%")
    print(f"Profit Factor: {report.get('profit_factor', 0):.2f}")
    print(f"Average P&L: ${report.get('avg_pnl', 0):,.2f}")
    print(f"Total P&L: ${report.get('total_pnl', 0):,.2f}")
    print(f"Total Costs: ${report.get('total_costs', 0):,.2f}")

    print(f"\n--- Risk Statistics ---")
    print(f"Sharpe Ratio: {report.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {report.get('max_drawdown_pct', 0):.2f}%")
    print(f"Calmar Ratio: {report.get('calmar_ratio', 0):.2f}")
    print(f"Volatility: {report.get('annualized_volatility_pct', 0):.2f}%")

    print(f"\n--- Trade Breakdown ---")
    print(f"Winning: {report.get('winning_trades', 0)} (Avg: ${report.get('avg_win', 0):,.2f})")
    print(f"Losing: {report.get('losing_trades', 0)} (Avg: ${report.get('avg_loss', 0):,.2f})")
    print(f"Largest Win: ${report.get('largest_win', 0):,.2f}")
    print(f"Largest Loss: ${report.get('largest_loss', 0):,.2f}")
