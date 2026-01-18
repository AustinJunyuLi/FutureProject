"""End-of-Month (EOM) seasonality analysis.

Analyzes spread behavior around month-end dates for seasonality patterns.
"""

from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from ..stage0.trading_calendar import TradingCalendar


class EOMSeasonality:
    """Analyze end-of-month patterns in spreads."""

    def __init__(self, calendar: Optional[TradingCalendar] = None):
        """Initialize analyzer.

        Args:
            calendar: Trading calendar (default CMEGlobex_Metals)
        """
        self.calendar = calendar or TradingCalendar("CMEGlobex_Metals")

    def add_eom_labels(
        self,
        df: pd.DataFrame,
        date_col: str = "trade_date",
    ) -> pd.DataFrame:
        """Add EOM relative day labels to data.

        Labels indicate business days from month end:
        - EOM-0: Last business day of month
        - EOM-1: Second-to-last business day
        - EOM-2, EOM-3, etc.

        Args:
            df: DataFrame with date column
            date_col: Name of date column

        Returns:
            DataFrame with eom_label and eom_offset columns
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Build mapping for all business days in the span.
        start = df[date_col].min().date()
        end = df[date_col].max().date()

        bdays = self.calendar.get_business_days(start, end)
        if len(bdays) == 0:
            df["eom_offset"] = np.nan
            df["eom_label"] = None
            return df

        cal = pd.DataFrame({"trade_date": pd.to_datetime(bdays)})
        cal["year"] = cal["trade_date"].dt.year
        cal["month"] = cal["trade_date"].dt.month
        cal["idx_in_month"] = cal.groupby(["year", "month"]).cumcount()
        cal["n_in_month"] = cal.groupby(["year", "month"])["trade_date"].transform("size")
        cal["eom_offset"] = cal["n_in_month"] - 1 - cal["idx_in_month"]
        cal["eom_label"] = "EOM-" + cal["eom_offset"].astype(int).astype(str)

        if date_col == "trade_date":
            return df.merge(cal[["trade_date", "eom_offset", "eom_label"]], on="trade_date", how="left")

        out = df.merge(
            cal[["trade_date", "eom_offset", "eom_label"]],
            left_on=date_col,
            right_on="trade_date",
            how="left",
        )
        return out.drop(columns=["trade_date"])

    def compute_eom_returns(
        self,
        df: pd.DataFrame,
        spread_col: str = "S1",
        entry_offset: int = 3,
        exit_offset: int = 1,
    ) -> pd.DataFrame:
        """Compute spread returns for EOM window.

        Default strategy: Long spread at EOM-3, exit at EOM-1.

        Args:
            df: DataFrame with spread and eom_offset columns
            spread_col: Spread column to analyze
            entry_offset: Entry day (EOM-N)
            exit_offset: Exit day (EOM-N)

        Returns:
            DataFrame with trade records
        """
        if "eom_offset" not in df.columns:
            df = self.add_eom_labels(df)

        # Group by year-month
        df = df.copy()
        df["year"] = pd.to_datetime(df["trade_date"]).dt.year
        df["month"] = pd.to_datetime(df["trade_date"]).dt.month

        trades = []

        for (year, month), group in df.groupby(["year", "month"]):
            # Find entry and exit rows
            entry_rows = group[group["eom_offset"] == entry_offset]
            exit_rows = group[group["eom_offset"] == exit_offset]

            if len(entry_rows) == 0 or len(exit_rows) == 0:
                continue

            entry_row = entry_rows.iloc[0]
            exit_row = exit_rows.iloc[0]

            entry_spread = entry_row[spread_col]
            exit_spread = exit_row[spread_col]

            if pd.isna(entry_spread) or pd.isna(exit_spread):
                continue

            # Long spread return = exit - entry (spread widening is profit)
            spread_return = exit_spread - entry_spread

            trades.append({
                "year": year,
                "month": month,
                "entry_date": entry_row["trade_date"],
                "exit_date": exit_row["trade_date"],
                "entry_spread": entry_spread,
                "exit_spread": exit_spread,
                "spread_return": spread_return,
                "holding_days": (exit_row["trade_date"] - entry_row["trade_date"]).days,
            })

        return pd.DataFrame(trades)

    def seasonal_summary(
        self,
        eom_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute seasonal summary statistics.

        Args:
            eom_returns: DataFrame from compute_eom_returns

        Returns:
            DataFrame with monthly summary statistics
        """
        if len(eom_returns) == 0:
            return pd.DataFrame()

        summary = eom_returns.groupby("month").agg({
            "spread_return": ["count", "mean", "std", "sum"],
        })

        summary.columns = ["count", "mean_return", "std_return", "total_return"]
        summary = summary.reset_index()

        # Win rate
        monthly_wins = eom_returns.groupby("month")["spread_return"].apply(
            lambda x: (x > 0).sum() / len(x) * 100
        ).reset_index()
        monthly_wins.columns = ["month", "win_rate"]

        summary = summary.merge(monthly_wins, on="month")

        # Add month names
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        summary["month_name"] = summary["month"].map(month_names)

        return summary


def build_eom_daily_dataset(
    spread_panel: pd.DataFrame,
    spread_col: str = "S1",
) -> pd.DataFrame:
    """Build daily dataset with EOM labels for analysis.

    Args:
        spread_panel: Spread panel from Stage 2
        spread_col: Spread column to include

    Returns:
        DataFrame with daily US-session VWAP proxy (09:00-15:59 CT) and EOM labels
    """
    df = spread_panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # US session buckets (09:00-15:59 CT): 1-7
    us = df[df["bucket"].between(1, 7)].copy() if "bucket" in df.columns else df.copy()
    if us.empty:
        return pd.DataFrame()

    # Weights for S1 proxy: combined front two contract volumes
    has_v1 = "F1_volume" in us.columns
    has_v2 = "F2_volume" in us.columns
    if has_v1 and has_v2:
        us["w_s1"] = us["F1_volume"].fillna(0.0) + us["F2_volume"].fillna(0.0)
    else:
        us["w_s1"] = 0.0

    # Price weights (for optional daily F1 proxy)
    if "F1_volume" in us.columns:
        us["w_f1"] = us["F1_volume"].fillna(0.0)
    else:
        us["w_f1"] = 0.0

    def _weighted_or_equal(values: pd.Series, weights: pd.Series) -> tuple[float, bool]:
        values = values.astype("float64")
        weights = weights.astype("float64").fillna(0.0)
        mask = values.notna()
        if not mask.any():
            return (np.nan, False)
        v = values[mask]
        w = weights[mask]
        if w.sum() > 0:
            return (float(np.average(v, weights=w)), False)
        return (float(v.mean()), True)

    records: list[dict] = []
    near_col = f"{spread_col}_near"
    far_col = f"{spread_col}_far"

    for td, g in us.groupby("trade_date"):
        # Choose a representative (near, far) pair: use last available US bucket.
        g_sorted = g.sort_values("bucket") if "bucket" in g.columns else g
        near = (
            g_sorted[near_col].dropna().iloc[-1]
            if near_col in g_sorted.columns and g_sorted[near_col].notna().any()
            else None
        )
        far = (
            g_sorted[far_col].dropna().iloc[-1]
            if far_col in g_sorted.columns and g_sorted[far_col].notna().any()
            else None
        )

        # Detect intraday pair changes inside the US session (important on expiry/roll days).
        if near_col in g.columns and far_col in g.columns:
            pair_changes = g[[near_col, far_col]].dropna().drop_duplicates().shape[0] > 1
        else:
            pair_changes = False

        # IMPORTANT: avoid mixing spreads from different (near, far) pairs in a single daily value.
        # We compute the daily proxy using only rows matching the representative pair.
        g_use = g
        pair_filter_applied = False
        if near is not None and far is not None and near_col in g.columns and far_col in g.columns:
            g_filtered = g[(g[near_col] == near) & (g[far_col] == far)].copy()
            if not g_filtered.empty:
                pair_filter_applied = len(g_filtered) < len(g)
                g_use = g_filtered

        # Weighted average (or equal-weight fallback) computed on the filtered subset.
        s1_val, s1_equal = (
            _weighted_or_equal(g_use[spread_col], g_use["w_s1"]) if spread_col in g_use.columns else (np.nan, False)
        )
        f1_val, f1_equal = (
            _weighted_or_equal(g_use["F1_price"], g_use["w_f1"]) if "F1_price" in g_use.columns else (np.nan, False)
        )

        s1_pct = (s1_val / f1_val) if (pd.notna(s1_val) and pd.notna(f1_val) and f1_val != 0) else np.nan

        records.append(
            {
                "trade_date": td,
                spread_col: s1_val,
                f"{spread_col}_pct": s1_pct,
                near_col: near,
                far_col: far,
                # Diagnostics: what was available vs what was used
                "us_buckets_present": int(g_use["bucket"].nunique()) if "bucket" in g_use.columns else np.nan,
                "us_buckets_with_spread": int(g_use[spread_col].notna().sum()) if spread_col in g_use.columns else 0,
                "us_total_weight": float(g_use["w_s1"].sum()),
                "us_equal_weight_fallback": bool(s1_equal),
                "us_pair_changes": bool(pair_changes),
                "us_pair_filter_applied": bool(pair_filter_applied),
                "F1_us_vwap": f1_val,
                "F1_us_equal_weight_fallback": bool(f1_equal),
            }
        )

    daily = pd.DataFrame(records).sort_values("trade_date").reset_index(drop=True)

    analyzer = EOMSeasonality()
    daily = analyzer.add_eom_labels(daily, date_col="trade_date")
    return daily
