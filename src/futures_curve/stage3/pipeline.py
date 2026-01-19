"""Stage 3 Pipeline: Analysis layer.

Provides seasonality analysis, lifecycle studies, and diagnostics.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from ..stage1.parquet_writer import read_bucket_data
from ..stage2.pipeline import read_curve_panel, read_spread_panel, read_roll_events
from .eom_seasonality import EOMSeasonality, build_eom_daily_dataset
from .lifecycle_analysis import LifecycleAnalyzer, build_lifecycle_dataset
from .diagnostics import run_full_diagnostics


class Stage3Pipeline:
    """Complete Stage 3 analysis pipeline."""

    def __init__(self, data_dir: str | Path):
        """Initialize pipeline.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        # Default mirrors the repository layout: data_parquet/ (data) and research_outputs/ (tables/figures).
        # The CLI can override this (e.g., for reproducible runs to a separate output root).
        self.research_dir = self.data_dir.parent / "research_outputs"

    def run_eom_analysis(
        self,
        symbol: str,
        entry_offset: int = 3,
        exit_offset: int = 1,
        verbose: bool = True,
    ) -> dict:
        """Run EOM seasonality analysis.

        Args:
            symbol: Commodity symbol
            entry_offset: Entry day (EOM-N)
            exit_offset: Exit day (EOM-N)
            verbose: Print progress

        Returns:
            Dictionary with analysis results
        """
        if verbose:
            print(f"Running EOM analysis for {symbol}...")

        # Load spread data
        spread_panel = read_spread_panel(self.data_dir, symbol)

        # Build daily dataset with EOM labels
        eom_daily = build_eom_daily_dataset(spread_panel, spread_col="S1")

        # Compute EOM returns
        analyzer = EOMSeasonality()
        eom_returns = analyzer.compute_eom_returns(
            eom_daily,
            spread_col="S1_pct",
            entry_offset=entry_offset,
            exit_offset=exit_offset,
        )

        # Seasonal summary
        seasonal_summary = analyzer.seasonal_summary(eom_returns)

        # Save outputs
        output_dir = self.research_dir / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        eom_returns.to_parquet(output_dir / f"{symbol}_eom_returns.parquet", index=False)
        seasonal_summary.to_parquet(output_dir / f"{symbol}_seasonal_summary.parquet", index=False)

        # Also save daily dataset
        eom_daily.to_parquet(output_dir / f"{symbol}_eom_daily.parquet", index=False)

        results = {
            "symbol": symbol,
            "total_trades": len(eom_returns),
            "mean_return": eom_returns["spread_return"].mean() if len(eom_returns) > 0 else None,
            "win_rate": (eom_returns["spread_return"] > 0).mean() * 100 if len(eom_returns) > 0 else None,
            "best_month": None,
            "worst_month": None,
        }

        if len(seasonal_summary) > 0:
            best_idx = seasonal_summary["mean_return"].idxmax()
            worst_idx = seasonal_summary["mean_return"].idxmin()
            results["best_month"] = seasonal_summary.loc[best_idx, "month_name"]
            results["worst_month"] = seasonal_summary.loc[worst_idx, "month_name"]

        if verbose:
            print(f"  {len(eom_returns)} EOM trades analyzed")
            if results["win_rate"]:
                print(f"  Win rate: {results['win_rate']:.1f}%")

        return results

    def run_lifecycle_analysis(
        self,
        symbol: str,
        verbose: bool = True,
    ) -> dict:
        """Run lifecycle/DTE analysis.

        Args:
            symbol: Commodity symbol
            verbose: Print progress

        Returns:
            Dictionary with analysis results
        """
        if verbose:
            print(f"Running lifecycle analysis for {symbol}...")

        # Load data
        curve_panel = read_curve_panel(self.data_dir, symbol)
        spread_panel = read_spread_panel(self.data_dir, symbol)
        roll_events = read_roll_events(self.data_dir, symbol)

        # Build lifecycle dataset
        lifecycle_df = build_lifecycle_dataset(curve_panel, spread_panel)

        # DTE profile
        analyzer = LifecycleAnalyzer()
        dte_profile = analyzer.dte_profile(lifecycle_df, spread_col="S1_pct", dte_col="F1_dte_bdays")

        # Roll event study
        # Prefer daily roll events if available (more stable); fall back to bucket.
        if "frequency" in roll_events.columns:
            daily_events = roll_events[roll_events["frequency"] == "daily"].copy()
            roll_events_use = daily_events if len(daily_events) > 0 else roll_events[roll_events["frequency"] == "bucket"].copy()
        else:
            roll_events_use = roll_events

        event_study = analyzer.roll_period_analysis(
            spread_panel,
            roll_events_use,
            spread_col="S1_pct",
        )
        avg_event_curve = analyzer.average_event_curve(event_study, spread_col="S1_pct")

        # Save outputs
        output_dir = self.research_dir / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        dte_profile.to_parquet(output_dir / f"{symbol}_dte_profile.parquet", index=False)
        if len(event_study) > 0:
            event_study.to_parquet(output_dir / f"{symbol}_event_study.parquet", index=False)
        if len(avg_event_curve) > 0:
            avg_event_curve.to_parquet(output_dir / f"{symbol}_avg_event_curve.parquet", index=False)

        results = {
            "symbol": symbol,
            "dte_bins": len(dte_profile),
            "roll_events_studied": len(roll_events),
            "event_observations": len(event_study),
        }

        if verbose:
            print(f"  {len(dte_profile)} DTE bins analyzed")
            print(f"  {len(roll_events)} roll events studied")

        return results

    def run_diagnostics(
        self,
        symbol: str,
        verbose: bool = True,
    ) -> dict:
        """Run comprehensive diagnostics.

        Args:
            symbol: Commodity symbol
            verbose: Print progress

        Returns:
            Dictionary with diagnostic results
        """
        if verbose:
            print(f"Running diagnostics for {symbol}...")

        # Load all data
        bucket_data = read_bucket_data(self.data_dir, symbol)
        curve_panel = read_curve_panel(self.data_dir, symbol)
        spread_panel = read_spread_panel(self.data_dir, symbol)

        # Run diagnostics
        results = run_full_diagnostics(
            bucket_data, curve_panel, spread_panel, symbol
        )

        # Save diagnostic report
        output_dir = self.research_dir / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        report = {
            "symbol": symbol,
            "ohlc_issues": results["ohlc_issues_count"],
            "zscore_events": results.get("zscore_events_count", 0),
            "expiry_violations": results["expiry_violations_count"],
            "spread_discrepancies": results["spread_discrepancies_count"],
            "data_gaps": results["data_gaps_count"],
        }

        report_df = pd.DataFrame([report])
        report_df.to_parquet(output_dir / f"{symbol}_diagnostics.parquet", index=False)

        if verbose:
            print(f"  OHLC issues: {results['ohlc_issues_count']}")
            print(f"  Expiry violations: {results['expiry_violations_count']}")
            print(f"  Spread discrepancies: {results['spread_discrepancies_count']}")

        return results

    def process_symbol(
        self,
        symbol: str,
        verbose: bool = True,
    ) -> dict:
        """Run all Stage 3 analyses for a symbol.

        Args:
            symbol: Commodity symbol
            verbose: Print progress

        Returns:
            Dictionary with all analysis results
        """
        start_time = datetime.now()

        if verbose:
            print(f"\nStage 3: Analyzing {symbol}")
            print("=" * 60)

        results = {"symbol": symbol}

        # EOM analysis
        results["eom"] = self.run_eom_analysis(symbol, verbose=verbose)

        # Lifecycle analysis
        results["lifecycle"] = self.run_lifecycle_analysis(symbol, verbose=verbose)

        # Diagnostics
        results["diagnostics"] = self.run_diagnostics(symbol, verbose=verbose)

        elapsed = (datetime.now() - start_time).total_seconds()
        results["elapsed_seconds"] = round(elapsed, 2)

        if verbose:
            print(f"\nCompleted {symbol} analysis in {elapsed:.1f}s")

        return results


def run_stage3(
    data_dir: str,
    symbols: list[str],
    research_dir: str | Path | None = None,
) -> dict:
    """Run Stage 3 pipeline for specified symbols.

    Args:
        data_dir: Base data directory
        symbols: List of commodity symbols

    Returns:
        Dictionary with analysis results per symbol
    """
    pipeline = Stage3Pipeline(data_dir)
    if research_dir is not None:
        pipeline.research_dir = Path(research_dir)

    results = {}
    for symbol in symbols:
        stats = pipeline.process_symbol(symbol)
        results[symbol] = stats

    return results
