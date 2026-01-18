"""Diagnostics and data quality analysis.

Provides tools for detecting anomalies, checking data quality,
and investigating z-score widening events.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


class DataDiagnostics:
    """Data quality diagnostics and anomaly detection."""

    def __init__(self):
        """Initialize diagnostics."""
        pass

    def check_ohlc_integrity(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Check OHLC data integrity.

        Validates:
        - high >= low
        - high >= open and close
        - low <= open and close
        - volume >= 0

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with rows that fail integrity checks
        """
        checks = pd.DataFrame(index=df.index)

        # High/Low check
        checks["hl_valid"] = df["high"] >= df["low"]

        # High >= Open/Close
        checks["ho_valid"] = df["high"] >= df["open"]
        checks["hc_valid"] = df["high"] >= df["close"]

        # Low <= Open/Close
        checks["lo_valid"] = df["low"] <= df["open"]
        checks["lc_valid"] = df["low"] <= df["close"]

        # Volume non-negative
        if "volume" in df.columns:
            checks["vol_valid"] = df["volume"] >= 0

        # Overall validity
        checks["all_valid"] = checks.all(axis=1)

        # Return failed rows
        failed = df[~checks["all_valid"]].copy()
        for col in checks.columns:
            failed[col] = checks.loc[~checks["all_valid"], col]

        return failed

    def detect_zscore_events(
        self,
        df: pd.DataFrame,
        spread_col: str = "S1",
        window: int = 50,
        z_threshold: float = 1.5,
        cooldown_hours: float = 3.0,
        min_abs_spread: float = 0.02,
    ) -> pd.DataFrame:
        """Detect abnormal spread exceedance events via a rolling z-score.

        Legacy-style diagnostic (not a roll definition):
        - rolling window: 50 buckets
        - threshold: |z| > 1.5 (widening if z>0, narrowing if z<0)
        - cooldown: 3 hours between events
        - minimum absolute spread magnitude: 0.02 USD/lb (2 cents)
        """
        if spread_col not in df.columns:
            return pd.DataFrame()

        data = df.copy()

        # Chronological ordering (bucket numbers are not time-ordered).
        if "ts_end_utc" in data.columns:
            data["ts_end_utc"] = pd.to_datetime(data["ts_end_utc"])
            data = data.sort_values("ts_end_utc")
            ts_col = "ts_end_utc"
        else:
            data["trade_date"] = pd.to_datetime(data["trade_date"])
            data = data.sort_values(["trade_date"] + (["bucket"] if "bucket" in data.columns else []))
            ts_col = "trade_date"

        x = data[spread_col].astype("float64")
        roll = x.rolling(window=window, min_periods=window)
        mean = roll.mean()
        std = roll.std()
        z = (x - mean) / std
        data["_z"] = z

        # Event candidates: abs spread magnitude and z-score threshold
        valid_spread = x.abs() >= float(min_abs_spread)
        extreme = valid_spread & z.abs().ge(float(z_threshold))

        if not extreme.any():
            return pd.DataFrame()

        cooldown = pd.Timedelta(hours=float(cooldown_hours))

        events = []
        last_ts = None
        for idx in np.flatnonzero(extreme.to_numpy()):
            row = data.iloc[int(idx)]
            ts = row[ts_col]
            if last_ts is not None and pd.notna(ts) and pd.notna(last_ts):
                if ts - last_ts < cooldown:
                    continue
            last_ts = ts

            direction = "widening" if row["_z"] > 0 else "narrowing"
            events.append(
                {
                    "ts": ts,
                    "trade_date": row.get("trade_date"),
                    "bucket": row.get("bucket"),
                    "spread": row[spread_col],
                    "zscore": row["_z"],
                    "direction": direction,
                    "window": int(window),
                    "z_threshold": float(z_threshold),
                    "cooldown_hours": float(cooldown_hours),
                    "min_abs_spread": float(min_abs_spread),
                }
            )

        return pd.DataFrame(events)

    def coverage_report(
        self,
        df: pd.DataFrame,
        date_col: str = "trade_date",
    ) -> dict:
        """Generate data coverage report.

        Args:
            df: DataFrame to analyze
            date_col: Date column name

        Returns:
            Dictionary with coverage statistics
        """
        dates = pd.to_datetime(df[date_col])

        return {
            "min_date": dates.min(),
            "max_date": dates.max(),
            "total_days": (dates.max() - dates.min()).days + 1,
            "observed_days": dates.nunique(),
            "coverage_pct": dates.nunique() / ((dates.max() - dates.min()).days + 1) * 100,
            "total_rows": len(df),
            "null_counts": df.isnull().sum().to_dict(),
        }

    def gap_analysis(
        self,
        df: pd.DataFrame,
        date_col: str = "trade_date",
        max_gap_days: int = 5,
    ) -> pd.DataFrame:
        """Find gaps in data coverage.

        Args:
            df: DataFrame to analyze
            date_col: Date column name
            max_gap_days: Threshold for flagging gaps

        Returns:
            DataFrame with gap records
        """
        dates = pd.to_datetime(df[date_col]).sort_values().unique()

        if len(dates) < 2:
            return pd.DataFrame()

        gaps = []
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > max_gap_days:
                gaps.append({
                    "start_date": dates[i-1],
                    "end_date": dates[i],
                    "gap_days": gap,
                })

        return pd.DataFrame(gaps)


class SpreadDiagnostics:
    """Spread-specific diagnostics."""

    def __init__(self):
        """Initialize diagnostics."""
        pass

    def spread_consistency_check(
        self,
        curve_panel: pd.DataFrame,
        spread_panel: pd.DataFrame,
        spread_num: int = 1,
    ) -> pd.DataFrame:
        """Verify spread calculation consistency.

        Args:
            curve_panel: Curve panel with F prices
            spread_panel: Spread panel with S values
            spread_num: Spread number to check

        Returns:
            DataFrame with discrepancies
        """
        # Merge panels
        merge_cols = ["trade_date"]
        if "bucket" in curve_panel.columns:
            merge_cols.append("bucket")

        merged = curve_panel[merge_cols + [f"F{spread_num}_price", f"F{spread_num+1}_price"]].merge(
            spread_panel[merge_cols + [f"S{spread_num}"]],
            on=merge_cols,
        )

        # Calculate expected spread
        merged["expected_spread"] = merged[f"F{spread_num+1}_price"] - merged[f"F{spread_num}_price"]

        # Find discrepancies
        have_both = merged[f"S{spread_num}"].notna() & merged["expected_spread"].notna()
        merged = merged[have_both].copy()
        merged["discrepancy"] = (merged[f"S{spread_num}"] - merged["expected_spread"]).abs()
        merged["is_consistent"] = merged["discrepancy"] < 1e-6

        return merged[~merged["is_consistent"]]

    def expiry_ordering_check(
        self,
        curve_panel: pd.DataFrame,
        max_contracts: int = 12,
    ) -> pd.DataFrame:
        """Verify expiry ordering is maintained.

        Args:
            curve_panel: Curve panel with F expiries
            max_contracts: Number of contracts to check

        Returns:
            DataFrame with ordering violations
        """
        violations = []

        for _, row in curve_panel.iterrows():
            for i in range(1, max_contracts):
                exp_i = row.get(f"F{i}_expiry_ts_utc")
                exp_i1 = row.get(f"F{i+1}_expiry_ts_utc")

                if pd.notna(exp_i) and pd.notna(exp_i1) and exp_i > exp_i1:
                    violations.append(
                        {
                            "trade_date": row.get("trade_date"),
                            "bucket": row.get("bucket"),
                            "position_i": i,
                            "position_i1": i + 1,
                            "expiry_i_utc": exp_i,
                            "expiry_i1_utc": exp_i1,
                        }
                    )

        return pd.DataFrame(violations)


def run_full_diagnostics(
    bucket_data: pd.DataFrame,
    curve_panel: pd.DataFrame,
    spread_panel: pd.DataFrame,
    symbol: str,
) -> dict:
    """Run comprehensive diagnostics on processed data.

    Args:
        bucket_data: Raw bucket data
        curve_panel: Curve panel
        spread_panel: Spread panel
        symbol: Commodity symbol

    Returns:
        Dictionary with diagnostic results
    """
    data_diag = DataDiagnostics()
    spread_diag = SpreadDiagnostics()

    results = {
        "symbol": symbol,
        "bucket_coverage": data_diag.coverage_report(bucket_data),
        "curve_coverage": data_diag.coverage_report(curve_panel),
        "spread_coverage": data_diag.coverage_report(spread_panel),
    }

    # OHLC integrity
    ohlc_issues = data_diag.check_ohlc_integrity(bucket_data)
    results["ohlc_issues_count"] = len(ohlc_issues)

    # Legacy-style diagnostic: abnormal exceedance events via z-score
    zscore_events = data_diag.detect_zscore_events(spread_panel, spread_col="S1")
    results["zscore_events_count"] = len(zscore_events)
    results["zscore_events"] = zscore_events

    # Expiry ordering
    expiry_violations = spread_diag.expiry_ordering_check(curve_panel)
    results["expiry_violations_count"] = len(expiry_violations)

    # Spread consistency
    spread_discrepancies = spread_diag.spread_consistency_check(curve_panel, spread_panel)
    results["spread_discrepancies_count"] = len(spread_discrepancies)

    # Gaps
    gaps = data_diag.gap_analysis(bucket_data)
    results["data_gaps_count"] = len(gaps)
    results["data_gaps"] = gaps

    return results
