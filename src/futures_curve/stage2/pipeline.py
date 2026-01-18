"""Stage 2 Pipeline: Curve construction, spreads, and roll detection.

Builds F1..F12 curves, S1..S11 spreads, and detects roll events.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from ..stage1.parquet_writer import read_bucket_data, read_daily_data, ParquetWriter
from .contract_ranker import build_curve_panel, build_daily_curve_panel
from .spread_calculator import build_spread_panel, SpreadCalculator
from .roll_detector import RollDetector, RollShareConfig


class Stage2Pipeline:
    """Complete Stage 2 processing pipeline."""

    def __init__(
        self,
        data_dir: str | Path,
        max_contracts: int = 12,
    ):
        """Initialize pipeline.

        Args:
            data_dir: Base data directory (with buckets/, daily/ subdirs)
            max_contracts: Maximum front-month positions
        """
        self.data_dir = Path(data_dir)
        self.max_contracts = max_contracts
        self.writer = ParquetWriter(self.data_dir)

    def process_symbol(
        self,
        symbol: str,
        zscore_window: int = 20,
        roll_threshold: float = 0.25,
        verbose: bool = True,
    ) -> dict:
        """Process curve construction for a symbol.

        Args:
            symbol: Commodity symbol (e.g., "HG")
            zscore_window: Window for spread z-scores
            roll_threshold: Volume share threshold for roll detection
            verbose: Print progress

        Returns:
            Dictionary with processing statistics
        """
        start_time = datetime.now()

        if verbose:
            print(f"Stage 2: Processing {symbol}")

        # Load bucket data
        if verbose:
            print("  Loading bucket data...")
        try:
            bucket_data = read_bucket_data(self.data_dir, symbol)
        except FileNotFoundError:
            return {"symbol": symbol, "status": "no_bucket_data"}

        if verbose:
            print(f"    {len(bucket_data):,} bucket records")

        # Load daily data
        if verbose:
            print("  Loading daily data...")
        try:
            daily_data = read_daily_data(self.data_dir, symbol)
        except FileNotFoundError:
            daily_data = None
            if verbose:
                print("    No daily data found")

        # Build bucket-level curve panel
        if verbose:
            print("  Building bucket curve panel...")
        curve_panel = build_curve_panel(
            bucket_data,
            symbol,
            max_contracts=self.max_contracts,
        )
        if verbose:
            print(f"    {len(curve_panel):,} curve records")

        # Build daily curve panel
        daily_curve_panel = None
        if daily_data is not None:
            if verbose:
                print("  Building daily curve panel...")
            daily_curve_panel = build_daily_curve_panel(
                daily_data,
                symbol,
                max_contracts=self.max_contracts,
            )
            if verbose:
                print(f"    {len(daily_curve_panel):,} daily curve records")

        # Calculate spreads
        if verbose:
            print("  Calculating spreads...")
        spread_panel = build_spread_panel(
            curve_panel,
            include_zscore=True,
            zscore_window=zscore_window,
        )
        if verbose:
            # Count non-null S1 values
            s1_count = spread_panel["S1"].notna().sum()
            print(f"    {s1_count:,} S1 spread observations")

        # Build roll share panels + detect roll events (bucket + daily)
        if verbose:
            print("  Detecting roll events...")

        # Legacy single-threshold arg maps to start_threshold; keep peak/end defaults.
        # Bucket-level defaults: persistence=2, smoothing_window=3.
        bucket_cfg = RollShareConfig(
            start_threshold=float(roll_threshold),
            peak_threshold=0.50,
            end_threshold=0.75,
            persistence=2,
            smoothing_window=3,
            min_total_volume=1.0,
            strict_gt=True,
        )
        daily_cfg = RollShareConfig(
            start_threshold=float(roll_threshold),
            peak_threshold=0.50,
            end_threshold=0.75,
            persistence=1,   # daily series is already smoother
            smoothing_window=1,
            min_total_volume=1.0,
            strict_gt=True,
        )

        roll_bucket_panel = RollDetector(config=bucket_cfg).build_roll_share_panel(curve_panel, frequency="bucket")
        roll_daily_panel = RollDetector(config=daily_cfg).build_roll_share_panel(curve_panel, frequency="daily")
        roll_shares = pd.concat([roll_bucket_panel, roll_daily_panel], ignore_index=True) if not roll_daily_panel.empty else roll_bucket_panel

        roll_events_bucket = RollDetector(config=bucket_cfg).detect_roll_events(roll_bucket_panel)
        roll_events_daily = RollDetector(config=daily_cfg).detect_roll_events(roll_daily_panel) if not roll_daily_panel.empty else pd.DataFrame()
        roll_events = pd.concat([roll_events_bucket, roll_events_daily], ignore_index=True) if not roll_events_daily.empty else roll_events_bucket

        # Report number of contract cycles with a start signal (coverage)
        if len(roll_events) > 0:
            start_coverage = roll_events["roll_start_ts_utc"].notna().mean() * 100
        else:
            start_coverage = 0.0
        if verbose:
            print(f"    {len(roll_events)} contract cycles (start coverage: {start_coverage:.1f}%)")

        # Write outputs
        if verbose:
            print("  Writing outputs...")

        # Curve panel
        curve_dir = self.data_dir / "curve" / symbol
        curve_dir.mkdir(parents=True, exist_ok=True)
        curve_panel.to_parquet(curve_dir / "curve_panel.parquet", index=False)

        if daily_curve_panel is not None:
            daily_curve_panel.to_parquet(
                curve_dir / "daily_curve_panel.parquet", index=False
            )

        # Spread panel
        spread_dir = self.data_dir / "spreads" / symbol
        spread_dir.mkdir(parents=True, exist_ok=True)
        spread_panel.to_parquet(spread_dir / "spreads_panel.parquet", index=False)

        # Roll events
        roll_dir = self.data_dir / "roll_events" / symbol
        roll_dir.mkdir(parents=True, exist_ok=True)
        roll_events.to_parquet(roll_dir / "roll_events.parquet", index=False)
        roll_shares.to_parquet(roll_dir / "roll_shares.parquet", index=False)

        # Compute statistics
        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            "symbol": symbol,
            "status": "success",
            "bucket_records": len(bucket_data),
            "curve_records": len(curve_panel),
            "daily_curve_records": len(daily_curve_panel) if daily_curve_panel is not None else 0,
            "spread_observations": spread_panel["S1"].notna().sum(),
            "roll_events": len(roll_events),
            "elapsed_seconds": round(elapsed, 2),
        }

        if verbose:
            print(f"Completed {symbol} in {elapsed:.1f}s")

        return stats


def run_stage2(
    data_dir: str,
    symbols: list[str],
    zscore_window: int = 20,
    roll_threshold: float = 0.25,
) -> dict:
    """Run Stage 2 pipeline for specified symbols.

    Args:
        data_dir: Base data directory
        symbols: List of commodity symbols
        zscore_window: Window for spread z-scores
        roll_threshold: Volume share threshold for roll detection

    Returns:
        Dictionary with processing statistics per symbol
    """
    pipeline = Stage2Pipeline(data_dir)

    results = {}
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Stage 2: {symbol}")
        print(f"{'='*60}")

        stats = pipeline.process_symbol(
            symbol,
            zscore_window=zscore_window,
            roll_threshold=roll_threshold,
        )
        results[symbol] = stats

    return results


# Convenience functions for reading Stage 2 outputs
def read_curve_panel(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    """Read curve panel for a symbol."""
    path = Path(data_dir) / "curve" / symbol / "curve_panel.parquet"
    return pd.read_parquet(path)


def read_spread_panel(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    """Read spread panel for a symbol."""
    path = Path(data_dir) / "spreads" / symbol / "spreads_panel.parquet"
    return pd.read_parquet(path)


def read_roll_events(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    """Read roll events for a symbol."""
    path = Path(data_dir) / "roll_events" / symbol / "roll_events.parquet"
    return pd.read_parquet(path)
