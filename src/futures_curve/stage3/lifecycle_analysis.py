"""Lifecycle and Days-to-Expiry (DTE) analysis.

Analyzes spread behavior as a function of time to expiry.
"""

from typing import Optional
import pandas as pd
import numpy as np

from ..stage0.trading_calendar import TradingCalendar

class LifecycleAnalyzer:
    """Analyze spread behavior across contract lifecycle."""

    def __init__(
        self,
        dte_bins: Optional[list[int]] = None,
        calendar: Optional[TradingCalendar] = None,
    ):
        """Initialize analyzer.

        Args:
            dte_bins: Custom DTE bin edges (default: [0, 5, 10, 20, 30, 60, 90, 180, 365])
            calendar: Trading calendar for business-day calculations
        """
        self.dte_bins = dte_bins or [0, 5, 10, 20, 30, 60, 90, 180, 365]
        self.dte_labels = self._create_bin_labels()
        self.calendar = calendar or TradingCalendar("CMEGlobex_Metals")

    def _create_bin_labels(self) -> list[str]:
        """Create labels for DTE bins."""
        labels = []
        for i in range(len(self.dte_bins) - 1):
            labels.append(f"{self.dte_bins[i]}-{self.dte_bins[i+1]}")
        return labels

    def add_dte_bin(
        self,
        df: pd.DataFrame,
        dte_col: str = "F1_dte_bdays",
    ) -> pd.DataFrame:
        """Add DTE bin column to data.

        Args:
            df: DataFrame with DTE column
            dte_col: Name of DTE column

        Returns:
            DataFrame with dte_bin column
        """
        df = df.copy()

        df["dte_bin"] = pd.cut(
            df[dte_col],
            bins=self.dte_bins,
            labels=self.dte_labels,
            right=True,
        )

        return df

    def dte_profile(
        self,
        df: pd.DataFrame,
        spread_col: str = "S1",
        dte_col: str = "F1_dte_bdays",
    ) -> pd.DataFrame:
        """Compute spread statistics by DTE bin.

        Args:
            df: DataFrame with spread and DTE columns
            spread_col: Spread column to analyze
            dte_col: DTE column

        Returns:
            DataFrame with statistics per DTE bin
        """
        df = self.add_dte_bin(df, dte_col)

        profile = df.groupby("dte_bin").agg({
            spread_col: ["count", "mean", "std", "min", "max", "median"],
        })

        profile.columns = ["count", "mean", "std", "min", "max", "median"]
        profile = profile.reset_index()

        # Add percentage positive (contango)
        pct_pos = df.groupby("dte_bin")[spread_col].apply(
            lambda x: (x > 0).mean() * 100
        ).reset_index()
        pct_pos.columns = ["dte_bin", "pct_contango"]

        profile = profile.merge(pct_pos, on="dte_bin")

        return profile

    def roll_period_analysis(
        self,
        curve_panel: pd.DataFrame,
        roll_events: pd.DataFrame,
        spread_col: str = "S1",
        window: int = 10,
    ) -> pd.DataFrame:
        """Event study around roll dates.

        Methodological note
        -------------------
        The research report defines the "roll day" as the point at which the
        deferred contract's volume share crosses 50%. In this codebase, that
        corresponds to `roll_peak_trade_date` (derived from the configured
        `peak_threshold`, default 0.50).

        For backward compatibility, if `roll_peak_trade_date` is not present or
        is missing for a given event, we fall back to `roll_start_trade_date`.

        Args:
            curve_panel: Bucket-level or daily curve panel with F1_contract and trade_date.
            roll_events: Roll events dataframe (expects at least roll_start_trade_date and roll_end_trade_date; optionally roll_peak_trade_date).
            spread_col: Spread column to analyze (e.g., "S1" or "S1_pct").
            window: Business-day window before/after the roll window.

        Returns:
            DataFrame with relative business days around the roll anchor.
        """
        # Normalize roll_events column naming for backward compatibility.
        if "F1_contract" not in roll_events.columns and "f1_contract" in roll_events.columns:
            roll_events = roll_events.rename(columns={"f1_contract": "F1_contract"})

        # Require minimal columns
        required = {"F1_contract", "roll_start_trade_date", "roll_end_trade_date"}
        missing = required - set(roll_events.columns)
        if missing:
            raise ValueError(f"roll_events missing required columns: {sorted(missing)}")

        # Use roll_peak as anchor when available
        has_peak = "roll_peak_trade_date" in roll_events.columns

        events = roll_events.dropna(subset=["roll_start_trade_date", "roll_end_trade_date"]).copy()
        if len(events) == 0:
            return pd.DataFrame()

        # Normalize date types
        for col in ["roll_start_trade_date", "roll_end_trade_date"]:
            events[col] = pd.to_datetime(events[col], errors="coerce").dt.date
        if has_peak:
            events["roll_peak_trade_date"] = pd.to_datetime(events["roll_peak_trade_date"], errors="coerce").dt.date

        # Select columns used for merge
        merge_cols = ["F1_contract", "roll_start_trade_date", "roll_end_trade_date"]
        if has_peak:
            merge_cols.append("roll_peak_trade_date")

        # Merge curve data with roll events
        merged = curve_panel.merge(events[merge_cols].drop_duplicates(), on="F1_contract", how="inner")
        if len(merged) == 0:
            return pd.DataFrame()

        merged = merged.copy()
        merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.date

        # Anchor: peak if present else start
        if has_peak:
            merged["roll_anchor_trade_date"] = merged["roll_peak_trade_date"].where(
                merged["roll_peak_trade_date"].notna(), merged["roll_start_trade_date"]
            )
        else:
            merged["roll_anchor_trade_date"] = merged["roll_start_trade_date"]

        # Compute relative business day
        def _rel_bday(row) -> float:
            anchor = row["roll_anchor_trade_date"]
            td = row["trade_date"]
            if pd.isna(anchor) or pd.isna(td):
                return float("nan")
            return self.calendar.business_days_between(anchor, td, include_start=True, include_end=True) - 1

        merged["rel_bday"] = merged.apply(_rel_bday, axis=1)

        # Keep a symmetric window around the roll window, but index relative to the anchor.
        # Determine event bounds relative to anchor (start/end are kept for context).
        window_start = merged["roll_start_trade_date"].apply(
            lambda d: self.calendar.add_business_days(d, -window)
        )
        window_end = merged["roll_end_trade_date"].apply(
            lambda d: self.calendar.add_business_days(d, window)
        )

        mask = (merged["trade_date"] >= window_start) & (merged["trade_date"] <= window_end)
        event_data = merged.loc[mask].copy()

        return event_data


    def average_event_curve(
        self,
        event_data: pd.DataFrame,
        spread_col: str = "S1",
    ) -> pd.DataFrame:
        """Compute average spread curve around roll events.

        Args:
            event_data: Data from roll_period_analysis
            spread_col: Spread column

        Returns:
            DataFrame with average curve by relative day
        """
        if len(event_data) == 0:
            return pd.DataFrame()

        rel_col = "rel_bday" if "rel_bday" in event_data.columns else "rel_day"
        avg_curve = event_data.groupby(rel_col).agg({
            spread_col: ["mean", "std", "count"],
        })

        avg_curve.columns = ["mean", "std", "count"]
        avg_curve = avg_curve.reset_index().rename(columns={rel_col: "rel_bday"})

        # Add confidence interval
        avg_curve["ci_lower"] = avg_curve["mean"] - 1.96 * avg_curve["std"] / np.sqrt(avg_curve["count"])
        avg_curve["ci_upper"] = avg_curve["mean"] + 1.96 * avg_curve["std"] / np.sqrt(avg_curve["count"])

        return avg_curve


def build_lifecycle_dataset(
    curve_panel: pd.DataFrame,
    spread_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build dataset for lifecycle analysis.

    Args:
        curve_panel: Curve panel with F1..F12 data
        spread_panel: Spread panel with S1..S11 data

    Returns:
        Combined dataset with lifecycle metrics
    """
    # Merge curve and spread data
    merge_cols = ["trade_date"]
    if "bucket" in curve_panel.columns:
        merge_cols.append("bucket")

    # Select relevant columns from each
    curve_cols = merge_cols + [col for col in curve_panel.columns if col.startswith("F1_") or col.startswith("F2_")]

    spread_cols = merge_cols + [col for col in spread_panel.columns if col.startswith("S1")]

    df = curve_panel[curve_cols].merge(
        spread_panel[spread_cols],
        on=merge_cols,
        how="inner",
    )

    return df
