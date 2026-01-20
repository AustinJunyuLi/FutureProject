"""Stage 4 Pipeline: Strategy backtesting.

Runs backtests for various strategies and generates performance reports.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
from datetime import datetime

from .backtester import run_backtest
from .execution import ExecutionConfig
from .performance import PerformanceAnalyzer, print_performance_report


class Stage4Pipeline:
    """Complete Stage 4 backtesting pipeline."""

    def __init__(
        self,
        data_dir: str | Path,
        config: Optional[ExecutionConfig] = None,
        risk_config: Optional[dict] = None,
    ):
        """Initialize pipeline.

        Args:
            data_dir: Base data directory
            config: Execution configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config or ExecutionConfig()

        # Risk/discipline controls (kept separate from ExecutionConfig)
        risk_config = risk_config or {}
        self.stop_loss_usd = risk_config.get('stop_loss_usd')
        self.max_holding_bdays = risk_config.get('max_holding_bdays')
        self.auto_roll_on_contract_change = risk_config.get('auto_roll_on_contract_change', True)
        self.allow_same_bucket_execution = risk_config.get('allow_same_bucket_execution', False)

    def process_symbol(
        self,
        symbol: str,
        strategies: Optional[List[str]] = None,
        data_frequency: str = "bucket",
        daily_agg_config: Optional[dict] = None,
        verbose: bool = True,
    ) -> dict:
        """Run all backtests for a symbol.

        Args:
            symbol: Commodity symbol
            strategies: List of strategy names (default: all)
            verbose: Print progress

        Returns:
            Dictionary with backtest results
        """
        start_time = datetime.now()

        if strategies is None:
            strategies = ["pre_expiry"]  # Default strategy

        if verbose:
            print(f"\nStage 4: Backtesting {symbol}")
            print("=" * 60)

        results = {"symbol": symbol, "strategies": {}}

        for strategy_name in strategies:
            if verbose:
                print(f"\n  Running {strategy_name} strategy...")

            try:
                result = run_backtest(
                    self.data_dir,
                    symbol,
                    strategy_name,
                    execution_config={
                        "slippage_ticks": self.config.slippage_ticks,
                        "tick_size": self.config.tick_size,
                        "tick_value": self.config.tick_value,
                        "commission_per_contract": self.config.commission_per_contract,
                        "initial_capital": self.config.initial_capital,
                    },
                    data_frequency=data_frequency,
                    daily_agg_config=daily_agg_config,
                    stop_loss_usd=self.stop_loss_usd,
                    max_holding_bdays=self.max_holding_bdays,
                    allow_same_bucket_execution=self.allow_same_bucket_execution,
                    auto_roll_on_contract_change=self.auto_roll_on_contract_change,
                )

                results["strategies"][strategy_name] = result

                if verbose and result.get("report"):
                    report = result["report"]
                    print(f"    Trades: {report.get('total_trades', 0)}")
                    print(f"    Win Rate: {report.get('win_rate', 0):.1f}%")
                    print(f"    Total P&L: ${report.get('total_pnl', 0):,.2f}")
                    print(f"    Sharpe: {report.get('sharpe_ratio', 0):.2f}")

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                results["strategies"][strategy_name] = {"status": "error", "error": str(e)}

        # Save results
        self._save_results(symbol, results)

        elapsed = (datetime.now() - start_time).total_seconds()
        results["elapsed_seconds"] = round(elapsed, 2)

        if verbose:
            print(f"\nCompleted {symbol} backtesting in {elapsed:.1f}s")

        return results

    def _save_results(self, symbol: str, results: dict) -> None:
        """Save backtest results to files.

        Args:
            symbol: Commodity symbol
            results: Results dictionary
        """
        output_dir = self.data_dir / "trades" / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        for strategy_name, result in results.get("strategies", {}).items():
            if result.get("status") == "error":
                continue

            # Save trades
            trades = result.get("trades")
            if trades is not None and len(trades) > 0:
                trades.to_parquet(
                    output_dir / f"{strategy_name}_trades.parquet",
                    index=False,
                )

            # Save equity curve
            equity = result.get("equity_curve")
            if equity is not None and len(equity) > 0:
                equity.to_parquet(
                    output_dir / f"{strategy_name}_equity.parquet",
                    index=False,
                )

        # Save summary
        summary_records = []
        for strategy_name, result in results.get("strategies", {}).items():
            report = result.get("report", {})
            summary_records.append({
                "symbol": symbol,
                "strategy": strategy_name,
                "total_trades": report.get("total_trades", 0),
                "win_rate": report.get("win_rate", 0),
                "total_pnl": report.get("total_pnl", 0),
                "sharpe_ratio": report.get("sharpe_ratio", 0),
                "max_drawdown_pct": report.get("max_drawdown_pct", 0),
                "profit_factor": report.get("profit_factor", 0),
            })

        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_df.to_parquet(output_dir / "summary.parquet", index=False)

    def run_pre_expiry_sweep(
        self,
        symbol: str,
        entry_dte_min: int = 2,
        entry_dte_max: int = 10,
        exit_dte_min: int = 0,
        exit_dte_max: int = 3,
        min_trades: int = 1,
        data_frequency: str = "bucket",
        daily_agg_config: Optional[dict] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Grid-search expiry-window parameters for the `pre_expiry` strategy.

        The search space is restricted to "last few days" by default. We rank
        combinations primarily by net P&L (after costs).

        Args:
            symbol: Commodity symbol
            entry_dte_min: Minimum entry DTE (inclusive)
            entry_dte_max: Maximum entry DTE (inclusive)
            exit_dte_min: Minimum exit DTE (inclusive)
            exit_dte_max: Maximum exit DTE (inclusive)
            min_trades: Minimum number of trades required to keep a result row
            verbose: Print progress and best parameters
        """
        entry_dte_min = int(entry_dte_min)
        entry_dte_max = int(entry_dte_max)
        exit_dte_min = int(exit_dte_min)
        exit_dte_max = int(exit_dte_max)
        min_trades = int(min_trades)

        if entry_dte_min <= 0 or entry_dte_max <= 0:
            raise ValueError("entry_dte_min/entry_dte_max must be >= 1")
        if exit_dte_min < 0 or exit_dte_max < 0:
            raise ValueError("exit_dte_min/exit_dte_max must be >= 0")
        if entry_dte_min > entry_dte_max:
            raise ValueError("entry_dte_min must be <= entry_dte_max")
        if exit_dte_min > exit_dte_max:
            raise ValueError("exit_dte_min must be <= exit_dte_max")

        records: list[dict] = []

        # Load data once (sweep can be 30-50 runs; avoid re-reading parquet each time).
        from ..stage2.pipeline import read_spread_panel
        from .aggregation import build_us_session_daily_vwap_panel
        from .backtester import Backtester
        from .execution import ExecutionEngine
        from .performance import PerformanceAnalyzer
        from .strategies import get_strategy

        panel = read_spread_panel(self.data_dir, symbol)
        freq = str(data_frequency or "bucket").lower()
        if freq in {"bucket", "buckets"}:
            pass
        elif freq in {"daily_us_vwap", "us_session_daily_vwap"}:
            panel = build_us_session_daily_vwap_panel(panel, **(daily_agg_config or {}))
        else:
            raise ValueError("data_frequency must be one of {'bucket','daily_us_vwap'}")

        backtester = Backtester(panel, self.config)

        for entry_dte in range(entry_dte_min, entry_dte_max + 1):
            for exit_dte in range(exit_dte_min, exit_dte_max + 1):
                if exit_dte >= entry_dte:
                    continue

                if verbose:
                    print(f"  Testing pre_expiry(entry_dte={entry_dte}, exit_dte={exit_dte})...")

                # Reset per-run state
                backtester.engine = ExecutionEngine(backtester.config)
                backtester.analyzer = PerformanceAnalyzer(backtester.config.initial_capital)

                strat_params = {"entry_dte": entry_dte, "exit_dte": exit_dte}
                if freq in {"daily_us_vwap", "us_session_daily_vwap"}:
                    # Daily has a single observation per day; execute on the same daily mark
                    # to keep DTE params interpretable (signal depends only on DTE).
                    strat_params["execution"] = "same"
                strategy = get_strategy("pre_expiry", **strat_params)

                allow_same = bool(self.allow_same_bucket_execution) or (strat_params.get("execution") == "same")
                result = backtester.run_strategy(
                    strategy,
                    spread_col="S1",
                    stop_loss_usd=self.stop_loss_usd,
                    max_holding_bdays=self.max_holding_bdays,
                    auto_roll_on_contract_change=self.auto_roll_on_contract_change,
                    allow_same_bucket_execution=allow_same,
                )

                report = result.get("report", {}) or {}
                total_trades = int(report.get("total_trades", 0) or 0)
                if total_trades < min_trades:
                    continue

                records.append(
                    {
                        "symbol": symbol,
                        "strategy": "pre_expiry",
                        "data_frequency": str(data_frequency),
                        "daily_pair_policy": (daily_agg_config or {}).get("pair_policy") if str(data_frequency).lower() != "bucket" else None,
                        "entry_dte": entry_dte,
                        "exit_dte": exit_dte,
                        "total_trades": total_trades,
                        "win_rate": float(report.get("win_rate", 0) or 0),
                        "total_pnl": float(report.get("total_pnl", 0) or 0),
                        "total_costs": float(report.get("total_costs", 0) or 0),
                        "sharpe_ratio": float(report.get("sharpe_ratio", 0) or 0),
                        "max_drawdown_pct": float(report.get("max_drawdown_pct", 0) or 0),
                        "profit_factor": float(report.get("profit_factor", 0) or 0),
                    }
                )

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = df.sort_values(["total_pnl", "sharpe_ratio"], ascending=[False, False]).reset_index(drop=True)

        out_dir = self.data_dir / "trades" / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        freq = str(data_frequency or "bucket").lower()
        if freq in {"bucket", "buckets"}:
            suffix = "bucket"
        elif freq in {"daily_us_vwap", "us_session_daily_vwap"}:
            suffix = "daily_us_vwap"
        else:
            suffix = freq
        pair_policy = (daily_agg_config or {}).get("pair_policy")
        if suffix != "bucket" and pair_policy:
            suffix = f"{suffix}_{pair_policy}"
        df.to_parquet(out_dir / f"pre_expiry_sweep_{suffix}.parquet", index=False)

        if verbose:
            best = df.iloc[0]
            print(
                f"\nBest pre_expiry params: entry_dte={int(best['entry_dte'])}, "
                f"exit_dte={int(best['exit_dte'])} | "
                f"trades={int(best['total_trades'])} | "
                f"net P&L=${float(best['total_pnl']):,.2f} | Sharpe={float(best['sharpe_ratio']):.2f}"
            )

        return df

    def run_cost_sensitivity(
        self,
        symbol: str,
        strategy_name: str = "dte",
        slippage_range: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run cost sensitivity analysis.

        Args:
            symbol: Commodity symbol
            strategy_name: Strategy to test
            slippage_range: List of slippage tick values
            verbose: Print progress

        Returns:
            DataFrame with results for each slippage level
        """
        if slippage_range is None:
            slippage_range = [0, 1, 2, 3, 5]

        results = []

        for slippage in slippage_range:
            if verbose:
                print(f"  Testing slippage = {slippage} ticks...")

            result = run_backtest(
                self.data_dir,
                symbol,
                strategy_name,
                execution_config={
                    "slippage_ticks": slippage,
                    "tick_size": self.config.tick_size,
                    "tick_value": self.config.tick_value,
                    "commission_per_contract": self.config.commission_per_contract,
                    "initial_capital": self.config.initial_capital,
                },
                stop_loss_usd=self.stop_loss_usd,
                max_holding_bdays=self.max_holding_bdays,
                allow_same_bucket_execution=self.allow_same_bucket_execution,
                auto_roll_on_contract_change=self.auto_roll_on_contract_change,
            )

            report = result.get("report", {})
            results.append({
                "slippage_ticks": slippage,
                "total_pnl": report.get("total_pnl", 0),
                "total_costs": report.get("total_costs", 0),
                "win_rate": report.get("win_rate", 0),
                "sharpe_ratio": report.get("sharpe_ratio", 0),
            })

        return pd.DataFrame(results)

    def run_stop_loss_sensitivity(
        self,
        symbol: str,
        strategy_name: str = "pre_expiry",
        stop_loss_values: Optional[List[Optional[float]]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run stop-loss sensitivity analysis.

        Notes
        -----
        The stop loss is evaluated on **gross** P&L (excluding costs) and is
        triggered on the close of each observation.

        Args:
            symbol: Commodity symbol
            strategy_name: Strategy to test
            stop_loss_values: List of stop-loss USD thresholds; include None to disable
            verbose: Print progress

        Returns:
            DataFrame with results for each stop-loss threshold
        """
        if stop_loss_values is None:
            stop_loss_values = [None, 25, 50, 100, 200, 500, 1000]

        results = []

        for stop_loss in stop_loss_values:
            if verbose:
                label = "None" if stop_loss is None else f"${float(stop_loss):.0f}"
                print(f"  Testing stop_loss_usd = {label}...")

            result = run_backtest(
                self.data_dir,
                symbol,
                strategy_name,
                execution_config={
                    "slippage_ticks": self.config.slippage_ticks,
                    "tick_size": self.config.tick_size,
                    "tick_value": self.config.tick_value,
                    "commission_per_contract": self.config.commission_per_contract,
                    "initial_capital": self.config.initial_capital,
                },
                stop_loss_usd=stop_loss,
                max_holding_bdays=self.max_holding_bdays,
                allow_same_bucket_execution=self.allow_same_bucket_execution,
                auto_roll_on_contract_change=self.auto_roll_on_contract_change,
            )

            report = result.get("report", {})
            total_pnl = float(report.get("total_pnl", 0) or 0)
            total_costs = float(report.get("total_costs", 0) or 0)

            results.append(
                {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "stop_loss_usd": float(stop_loss) if stop_loss is not None else None,
                    "total_trades": report.get("total_trades", 0),
                    "win_rate": report.get("win_rate", 0),
                    "total_pnl": total_pnl,
                    "gross_pnl": total_pnl + total_costs,
                    "total_costs": total_costs,
                    "sharpe_ratio": report.get("sharpe_ratio", 0),
                    "max_drawdown_pct": report.get("max_drawdown_pct", 0),
                }
            )

        return pd.DataFrame(results)


def run_stage4(
    data_dir: str,
    symbols: list[str],
    strategies: Optional[List[str]] = None,
    execution_config: Optional[dict] = None,
    data_frequency: str = "bucket",
    daily_agg_config: Optional[dict] = None,
) -> dict:
    """Run Stage 4 pipeline for specified symbols.

    Args:
        data_dir: Base data directory
        symbols: List of commodity symbols
        strategies: List of strategies to run

    Returns:
        Dictionary with results per symbol
    """
    if execution_config:
        from dataclasses import fields

        allowed = {f.name for f in fields(ExecutionConfig)}
        filtered = {k: v for k, v in execution_config.items() if k in allowed}
        config_obj = ExecutionConfig(**filtered)
    else:
        config_obj = None
    risk_cfg: dict = {}
    if execution_config:
        risk_keys = {"stop_loss_usd", "max_holding_bdays", "auto_roll_on_contract_change", "allow_same_bucket_execution"}
        risk_cfg = {k: v for k, v in execution_config.items() if k in risk_keys}

    pipeline = Stage4Pipeline(data_dir, config=config_obj, risk_config=risk_cfg)

    results = {}
    for symbol in symbols:
        result = pipeline.process_symbol(
            symbol,
            strategies=strategies,
            data_frequency=data_frequency,
            daily_agg_config=daily_agg_config,
        )
        results[symbol] = result

    return results
