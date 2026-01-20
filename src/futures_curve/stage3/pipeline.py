"""Stage 3 Pipeline: Analysis layer.

Provides lifecycle studies and diagnostics.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from ..stage1.parquet_writer import read_bucket_data
from ..stage2.pipeline import read_curve_panel, read_spread_panel, read_roll_events
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
