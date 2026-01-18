"""Calendar spread calculator.

Computes S1..S11 spreads from F1..F12 curve panel.

Spread definitions:
- S1 = F2 - F1 (front spread)
- S2 = F3 - F2
- ...
- S11 = F12 - F11
"""

from typing import Optional
import pandas as pd
import numpy as np


class SpreadCalculator:
    """Calculate calendar spreads from curve panel."""

    def __init__(self, max_spreads: int = 11):
        """Initialize calculator.

        Args:
            max_spreads: Maximum number of spreads (default 11 for S1..S11)
        """
        self.max_spreads = max_spreads

    def calculate_spreads(
        self,
        curve_panel: pd.DataFrame,
        include_zscore: bool = True,
        zscore_window: int = 20,
    ) -> pd.DataFrame:
        """Calculate spreads from curve panel.

        Args:
            curve_panel: DataFrame with F1..F12 prices
            include_zscore: Include rolling z-scores
            zscore_window: Window for z-score calculation

        Returns:
            DataFrame with S1..S11 spreads and metadata
        """
        df = curve_panel.copy()

        # Calculate each spread
        for i in range(1, self.max_spreads + 1):
            near_col = f"F{i}_price"
            far_col = f"F{i+1}_price"

            if near_col in df.columns and far_col in df.columns:
                # Spread = Far - Near (positive = contango)
                df[f"S{i}"] = df[far_col] - df[near_col]
                # Also store normalized version for descriptive analysis/plots
                df[f"S{i}_pct"] = (df[f"S{i}"] / df[near_col]).where(df[near_col].notna() & (df[near_col] != 0))

                # Near and far contracts
                if f"F{i}_contract" in df.columns:
                    df[f"S{i}_near"] = df[f"F{i}_contract"]
                if f"F{i+1}_contract" in df.columns:
                    df[f"S{i}_far"] = df[f"F{i+1}_contract"]

                # DTE of near leg
                if f"F{i}_dte_bdays" in df.columns:
                    df[f"S{i}_dte_bdays"] = df[f"F{i}_dte_bdays"]
                if f"F{i}_dte_hours" in df.columns:
                    df[f"S{i}_dte_hours"] = df[f"F{i}_dte_hours"]

        # Calculate z-scores (rolling, causal)
        if include_zscore:
            df = self._add_rolling_zscores(df, zscore_window)

        return df

    def _add_rolling_zscores(
        self,
        df: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """Add rolling z-scores for spreads.

        Uses causal (backward-looking) rolling windows only.

        Args:
            df: DataFrame with spread columns
            window: Rolling window size

        Returns:
            DataFrame with z-score columns added
        """
        df = df.copy()

        # Sort chronologically for rolling windows. For bucket-level panels,
        # bucket numbers are not chronological (bucket 8/9 are on prior calendar
        # day of the trade_date). Prefer ts_end_utc if present.
        if "ts_end_utc" in df.columns:
            df = df.sort_values(["ts_end_utc"])
        else:
            date_cols = ["trade_date"]
            if "bucket" in df.columns:
                date_cols.append("bucket")
            df = df.sort_values(date_cols)

        for i in range(1, self.max_spreads + 1):
            spread_col = f"S{i}"
            if spread_col not in df.columns:
                continue

            # Rolling mean and std (causal - min_periods=window for full lookback)
            rolling = df[spread_col].rolling(window=window, min_periods=window)
            mean = rolling.mean()
            std = rolling.std()

            # Z-score = (value - mean) / std
            df[f"S{i}_zscore"] = (df[spread_col] - mean) / std

        return df

    def get_spread_summary(
        self,
        spread_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get summary statistics for spreads.

        Args:
            spread_panel: DataFrame with spread columns

        Returns:
            DataFrame with summary stats per spread
        """
        records = []

        for i in range(1, self.max_spreads + 1):
            spread_col = f"S{i}"
            if spread_col not in spread_panel.columns:
                continue

            data = spread_panel[spread_col].dropna()

            if len(data) == 0:
                continue

            records.append({
                "spread": f"S{i}",
                "count": len(data),
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "median": data.median(),
                "pct_positive": (data > 0).mean() * 100,  # % contango
            })

        return pd.DataFrame(records)


def build_spread_panel(
    curve_panel: pd.DataFrame,
    include_zscore: bool = True,
    zscore_window: int = 20,
) -> pd.DataFrame:
    """Build spread panel from curve panel.

    Args:
        curve_panel: DataFrame with F1..F12 prices
        include_zscore: Include rolling z-scores
        zscore_window: Window for z-score calculation

    Returns:
        DataFrame with S1..S11 spreads
    """
    calc = SpreadCalculator()
    return calc.calculate_spreads(
        curve_panel,
        include_zscore=include_zscore,
        zscore_window=zscore_window,
    )


def extract_front_spread(
    spread_panel: pd.DataFrame,
    spread_num: int = 1,
) -> pd.DataFrame:
    """Extract a single spread series for analysis.

    Args:
        spread_panel: Full spread panel
        spread_num: Spread number (1 for S1, etc.)

    Returns:
        DataFrame with just that spread's data
    """
    base_cols = ["trade_date"]
    if "bucket" in spread_panel.columns:
        base_cols.append("bucket")

    spread_cols = [
        f"S{spread_num}",
        f"S{spread_num}_pct",
        f"S{spread_num}_near",
        f"S{spread_num}_far",
        f"S{spread_num}_dte_bdays",
        f"S{spread_num}_dte_hours",
    ]

    if f"S{spread_num}_zscore" in spread_panel.columns:
        spread_cols.append(f"S{spread_num}_zscore")

    cols = base_cols + [c for c in spread_cols if c in spread_panel.columns]
    return spread_panel[cols].copy()
