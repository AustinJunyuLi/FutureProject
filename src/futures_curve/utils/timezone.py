"""Timezone + session utilities for futures data.

Design goals:
- Treat the *exchange time* as US/Central (CME).
- Define the CME *trade date* boundary at 17:00 CT (Globex reopen).
- Keep a 10-bucket session schema that does **not** straddle the 17:00 boundary.

Important note on raw data:
- Raw timestamps in vendor TXT files are typically *naive* and may be in a
  timezone that is not US/Central (e.g., US/Eastern). Stage 1 is responsible
  for inferring the raw timezone and converting to exchange time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Union, Optional, Iterable
import pytz
import pandas as pd

# Primary timezones
CENTRAL_TZ = pytz.timezone("US/Central")
EASTERN_TZ = pytz.timezone("US/Eastern")
UTC_TZ = pytz.UTC

# CME trade date boundary (US/Central)
TRADE_DATE_CUTOFF = time(17, 0)  # 5:00 PM CT

# Maintenance break (approx; actual schedule can vary by product/holiday)
MAINTENANCE_START = time(16, 0)  # 4:00 PM CT
MAINTENANCE_END = time(17, 0)    # 5:00 PM CT


@dataclass(frozen=True)
class BucketDefinition:
    """Bucket definition in exchange time (US/Central)."""

    bucket_id: int
    start: time
    end: time
    crosses_midnight: bool
    description: str


# 10-bucket schema (exchange time, US/Central).
#
# Buckets are defined so that the *trade date* boundary at 17:00 CT is also the
# start of bucket 8. The maintenance break (16:00-16:59 CT) is treated as
# bucket 0 for QC only.
BUCKETS: dict[int, BucketDefinition] = {
    0: BucketDefinition(
        bucket_id=0,
        start=time(16, 0),
        end=time(16, 59, 59),
        crosses_midnight=False,
        description="Maintenance hour (16:00-16:59 CT) - should be empty; QC only",
    ),
    1: BucketDefinition(1, time(9, 0), time(9, 59, 59), False, "US session hour 1"),
    2: BucketDefinition(2, time(10, 0), time(10, 59, 59), False, "US session hour 2"),
    3: BucketDefinition(3, time(11, 0), time(11, 59, 59), False, "US session hour 3"),
    4: BucketDefinition(4, time(12, 0), time(12, 59, 59), False, "US session hour 4"),
    5: BucketDefinition(5, time(13, 0), time(13, 59, 59), False, "US session hour 5"),
    6: BucketDefinition(6, time(14, 0), time(14, 59, 59), False, "US session hour 6"),
    7: BucketDefinition(7, time(15, 0), time(15, 59, 59), False, "US session hour 7"),
    8: BucketDefinition(8, time(17, 0), time(20, 59, 59), False, "Asia/Europe overlap (post reopen)"),
    9: BucketDefinition(9, time(21, 0), time(2, 59, 59), True, "Overnight (cross-midnight)"),
    10: BucketDefinition(10, time(3, 0), time(8, 59, 59), False, "Pre-US session"),
}


def localize_to_central(dt: Union[datetime, pd.Timestamp]) -> Union[datetime, pd.Timestamp]:
    """Localize a naive datetime to US/Central.

    Args:
        dt: Naive datetime or Timestamp

    Returns:
        Localized datetime in US/Central
    """
    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            return dt.tz_localize(CENTRAL_TZ)
        return dt.tz_convert(CENTRAL_TZ)

    if dt.tzinfo is None:
        return CENTRAL_TZ.localize(dt)
    return dt.astimezone(CENTRAL_TZ)


def to_utc(dt: Union[datetime, pd.Timestamp]) -> Union[datetime, pd.Timestamp]:
    """Convert a datetime to UTC.

    Args:
        dt: Datetime (naive assumed Central, or timezone-aware)

    Returns:
        Datetime in UTC
    """
    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            dt = dt.tz_localize(CENTRAL_TZ)
        return dt.tz_convert(UTC_TZ)

    if dt.tzinfo is None:
        dt = CENTRAL_TZ.localize(dt)
    return dt.astimezone(UTC_TZ)


def get_trade_date(dt: Union[datetime, pd.Timestamp]) -> date:
    """Get the trade date for a given timestamp.

    The trade date is the date the position settles on. For CME:
    - 17:00 CT Sunday through 16:00 CT Friday = next calendar date
    - Session starting at 17:00 CT on Monday trades for Tuesday's date

    The key cutoff is 17:00 CT (5 PM Central):
    - Before 17:00 CT: same calendar date
    - 17:00 CT onwards: next calendar date

    Args:
        dt: Timestamp in any timezone (will be converted to Central)

    Returns:
        Trade date
    """
    # Convert to Central time
    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            ct = dt.tz_localize(CENTRAL_TZ)
        else:
            ct = dt.tz_convert(CENTRAL_TZ)
        ct_time = ct.time()
        ct_date = ct.date()
    else:
        if dt.tzinfo is None:
            ct = CENTRAL_TZ.localize(dt)
        else:
            ct = dt.astimezone(CENTRAL_TZ)
        ct_time = ct.time()
        ct_date = ct.date()

    # If time is >= 17:00, trade date is next calendar day
    if ct_time >= TRADE_DATE_CUTOFF:
        return ct_date + timedelta(days=1)
    return ct_date


def get_bucket_number(dt: Union[datetime, pd.Timestamp]) -> int:
    """Assign a time bucket based on US/Central exchange time.

    Bucket 0 represents the 16:00-16:59 CT maintenance hour and is intended
    for QC only (many datasets should have ~0 activity there). Downstream
    analytics/backtests should typically exclude bucket 0.

    Args:
        dt: Timestamp (naive assumed Central)

    Returns:
        Bucket number 1-10
    """
    # Convert to Central
    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            ct = dt.tz_localize(CENTRAL_TZ)
        else:
            ct = dt.tz_convert(CENTRAL_TZ)
        hour = ct.hour
    else:
        if dt.tzinfo is None:
            ct = CENTRAL_TZ.localize(dt)
        else:
            ct = dt.astimezone(CENTRAL_TZ)
        hour = ct.hour

    # Maintenance hour (bucket 0)
    if hour == 16:
        return 0

    # US session hours (buckets 1-7)
    if 9 <= hour <= 15:
        return hour - 8  # 9->1, 10->2, ..., 15->7

    # Post reopen (bucket 8)
    if 17 <= hour <= 20:
        return 8

    # Overnight (bucket 9)
    if hour >= 21 or hour <= 2:
        return 9

    # Pre-US (bucket 10)
    if 3 <= hour <= 8:
        return 10

    # Should not reach here (e.g., 16 handled above)
    raise ValueError(f"Unexpected hour: {hour}")


def bucket_time_range(bucket: int) -> tuple[time, time]:
    """Get the start and end time for a bucket.

    Args:
        bucket: Bucket number 1-10

    Returns:
        Tuple of (start_time, end_time) in Central time
    """
    if bucket not in BUCKETS:
        raise ValueError(f"Invalid bucket: {bucket}")
    b = BUCKETS[bucket]
    return (b.start, b.end)


def bucket_start_end_local(trade_date: date, bucket: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get timezone-aware start/end timestamps for a (trade_date, bucket).

    Trade date is the CME trade date (17:00 CT boundary). Buckets 8 and 9
    start on the previous calendar day relative to trade_date.
    """
    if bucket not in BUCKETS:
        raise ValueError(f"Invalid bucket: {bucket}")

    b = BUCKETS[bucket]

    # Buckets that begin on the prior calendar day for this trade_date
    if bucket in {8, 9}:
        start_day = trade_date - timedelta(days=1)
    else:
        start_day = trade_date

    # End day for cross-midnight bucket 9 is on trade_date
    if bucket == 9:
        end_day = trade_date
    # Buckets on prior day that do not cross midnight (bucket 8)
    elif bucket == 8:
        end_day = trade_date - timedelta(days=1)
    else:
        end_day = trade_date

    start_ts = CENTRAL_TZ.localize(datetime.combine(start_day, b.start))
    end_ts = CENTRAL_TZ.localize(datetime.combine(end_day, b.end))
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts)


def bucket_sort_key(bucket: int) -> int:
    """Stable chronological ordering of bucket IDs within a trade_date.

    The numeric bucket IDs are *not* chronological because bucket 8/9 occur
    on the prior calendar day of the trade_date. Use this key (or ts_end_utc)
    for time-series sorting.
    """
    order = {
        8: 0,
        9: 1,
        10: 2,
        1: 3,
        2: 4,
        3: 5,
        4: 6,
        5: 7,
        6: 8,
        7: 9,
        0: 10,
    }
    if bucket not in order:
        raise ValueError(f"Invalid bucket: {bucket}")
    return order[bucket]


BUCKET_DESCRIPTIONS = {
    0: "Maintenance hour (16:00-16:59 CT) - QC only",
    1: "US session hour 1 (09:00-09:59 CT)",
    2: "US session hour 2 (10:00-10:59 CT)",
    3: "US session hour 3 (11:00-11:59 CT)",
    4: "US session hour 4 (12:00-12:59 CT)",
    5: "US session hour 5 (13:00-13:59 CT)",
    6: "US session hour 6 (14:00-14:59 CT)",
    7: "US session hour 7 (15:00-15:59 CT)",
    8: "Post reopen (17:00-20:59 CT)",
    9: "Overnight (21:00-02:59 CT)",
    10: "Pre-US session (03:00-08:59 CT)",
}
