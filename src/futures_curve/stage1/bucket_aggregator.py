"""10-bucket time aggregation for futures data.

Aggregates minute-level data into session buckets per CME trade date.

Exchange conventions:
- Exchange timezone: US/Central
- CME trade date boundary: 17:00 CT
- Maintenance break: 16:00-16:59 CT (bucket 0, QC only)

Bucket schema (US/Central):
| Bucket | Time Range    | Description                        |
|--------|---------------|------------------------------------|
| 0      | 16:00-16:59   | Maintenance hour (QC only)         |
| 1      | 09:00-09:59   | US session hour 1                  |
| 2      | 10:00-10:59   | US session hour 2                  |
| 3      | 11:00-11:59   | US session hour 3                  |
| 4      | 12:00-12:59   | US session hour 4                  |
| 5      | 13:00-13:59   | US session hour 5                  |
| 6      | 14:00-14:59   | US session hour 6                  |
| 7      | 15:00-15:59   | US session hour 7                  |
| 8      | 17:00-20:59   | Post reopen (Asia/Europe overlap)  |
| 9      | 21:00-02:59   | Overnight (cross-midnight)         |
| 10     | 03:00-08:59   | Pre-US session                     |
"""

from datetime import date, datetime
from typing import Optional
import pandas as pd
import numpy as np

from ..utils.timezone import CENTRAL_TZ, get_trade_date, get_bucket_number


class BucketAggregator:
    """Aggregate minute data into 10-bucket OHLCV."""

    def __init__(self):
        """Initialize aggregator."""
        pass

    def add_bucket_columns(
        self,
        df: pd.DataFrame,
        raw_timezone: str | None = None,
    ) -> pd.DataFrame:
        """Add trade_date and bucket columns to minute data.

        Args:
            df: DataFrame with timestamp column (naive or tz-aware)
            raw_timezone: If timestamps are naive, interpret them in this timezone
                before converting to US/Central. If None, naive timestamps are
                assumed to already be US/Central.

        Returns:
            DataFrame with trade_date and bucket columns added
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure timestamps are in US/Central exchange time
        if df["timestamp"].dt.tz is None:
            if raw_timezone is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(CENTRAL_TZ)
            else:
                localized = df["timestamp"].dt.tz_localize(
                    raw_timezone,
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
                df["timestamp"] = localized.dt.tz_convert(CENTRAL_TZ)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(CENTRAL_TZ)

        # Compute trade date and bucket for each row
        df["trade_date"] = df["timestamp"].apply(get_trade_date)
        df["bucket"] = df["timestamp"].apply(get_bucket_number)

        return df

    def aggregate_to_buckets(
        self,
        df: pd.DataFrame,
        contract_col: str = "contract",
    ) -> pd.DataFrame:
        """Aggregate minute data to bucket-level OHLCV.

        Args:
            df: DataFrame with timestamp, OHLCV, and contract columns
            contract_col: Name of contract column

        Returns:
            DataFrame with bucket-level OHLCV:
            - trade_date: Trade date
            - bucket: Bucket number (1-10)
            - contract: Contract code
            - open: First open in bucket
            - high: Max high in bucket
            - low: Min low in bucket
            - close: Last close in bucket
            - volume: Sum of volume in bucket
            - tick_count: Number of minute bars in bucket
        """
        # Add bucket columns if not present
        if "bucket" not in df.columns or "trade_date" not in df.columns:
            df = self.add_bucket_columns(df)

        # Group by trade_date, bucket, contract
        group_cols = ["trade_date", "bucket"]
        if contract_col in df.columns:
            group_cols.append(contract_col)

        # Sort by timestamp within each group to get correct open/close
        df = df.sort_values("timestamp")

        # Aggregation functions
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "timestamp": "count",  # tick_count
        }

        result = df.groupby(group_cols, as_index=False).agg(agg_dict)
        result = result.rename(columns={"timestamp": "tick_count"})

        # Add symbol if present
        if "symbol" in df.columns:
            # Get symbol from first row of each group
            symbol_map = df.groupby(contract_col)["symbol"].first()
            result["symbol"] = result[contract_col].map(symbol_map)

        return result

    def aggregate_to_daily(
        self,
        df: pd.DataFrame,
        contract_col: str = "contract",
        us_session_only: bool = True,
    ) -> pd.DataFrame:
        """Aggregate to daily OHLCV using US session for proxy.

        The daily price uses US session (buckets 1-7) for settlement proxy.
        Volume includes all buckets.

        Args:
            df: DataFrame with minute or bucket data
            contract_col: Name of contract column
            us_session_only: If True, OHLC from US session only

        Returns:
            DataFrame with daily OHLCV
        """
        # Add bucket columns if not present
        if "bucket" not in df.columns or "trade_date" not in df.columns:
            df = self.add_bucket_columns(df)

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        group_cols = ["trade_date"]
        if contract_col in df.columns:
            group_cols.append(contract_col)

        if us_session_only:
            # OHLC from US session (buckets 1-7) only
            us_df = df[df["bucket"].between(1, 7)]

            if len(us_df) == 0:
                # No US session data, return empty
                return pd.DataFrame()

            ohlc_result = us_df.groupby(group_cols, as_index=False).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })

            # Volume from all sessions
            vol_result = df.groupby(group_cols, as_index=False).agg({
                "volume": "sum",
            })

            # Merge
            result = ohlc_result.merge(vol_result, on=group_cols)
        else:
            result = df.groupby(group_cols, as_index=False).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })

        # Add symbol if present
        if "symbol" in df.columns and contract_col in df.columns:
            symbol_map = df.groupby(contract_col)["symbol"].first()
            result["symbol"] = result[contract_col].map(symbol_map)

        return result

    def compute_vwap(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.Series:
        """Compute VWAP from bucket or minute data.

        Args:
            df: DataFrame with price and volume columns
            price_col: Price column to use (default close)

        Returns:
            Series with VWAP values (NaN for zero volume)
        """
        total_value = (df[price_col] * df["volume"]).sum()
        total_volume = df["volume"].sum()

        if total_volume == 0:
            return np.nan
        return total_value / total_volume


def process_contract_to_buckets(
    df: pd.DataFrame,
    contract: str,
    symbol: str,
) -> pd.DataFrame:
    """Process a single contract's minute data to buckets.

    Args:
        df: DataFrame with minute OHLCV (timestamp, O, H, L, C, V)
        contract: Contract code (e.g., "HGF24")
        symbol: Symbol (e.g., "HG")

    Returns:
        DataFrame with bucket-level OHLCV
    """
    df = df.copy()
    df["contract"] = contract
    df["symbol"] = symbol

    agg = BucketAggregator()
    return agg.aggregate_to_buckets(df, contract_col="contract")


def build_bucket_panel(
    bucket_data: list[pd.DataFrame],
) -> pd.DataFrame:
    """Combine bucket data from multiple contracts into panel.

    Args:
        bucket_data: List of bucket DataFrames

    Returns:
        Combined DataFrame sorted by trade_date, bucket, contract
    """
    if not bucket_data:
        return pd.DataFrame()

    combined = pd.concat(bucket_data, ignore_index=True)
    combined = combined.sort_values(
        ["trade_date", "bucket", "contract"]
    ).reset_index(drop=True)

    return combined
