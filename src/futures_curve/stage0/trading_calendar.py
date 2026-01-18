"""Trading calendar utilities.

This module primarily targets the CME Globex calendars via `pandas_market_calendars`.

Important design note
---------------------
The pipeline's methodological invariants require *exchange business days* (not merely
Mon–Fri civil days) for DTE, EOM labeling, and roll-window alignment.

`pandas_market_calendars` is the intended source of truth. However, the repository's
README claims `pytest -q` should work after a fresh install, and the library is not
listed in the original dependency list. To make the codebase robust and testable in
minimal environments, we provide a deterministic fallback that treats business days
as Mon–Fri (holidays ignored) when `pandas_market_calendars` is unavailable.

The fallback is adequate for unit tests that check basic invariants (weekends,
monotonicity, etc.), but it is **not** suitable for reproducing the published
2008–2024 research numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytz

try:
    import pandas_market_calendars as mcal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mcal = None


CENTRAL_TZ = pytz.timezone("US/Central")
UTC_TZ = pytz.UTC


def _fallback_schedule(start_date: date, end_date: date, tz_name: str) -> pd.DataFrame:
    """Fallback Mon–Fri schedule.

    Returns a schedule DataFrame with columns `market_open` and `market_close`.

    Index is a DatetimeIndex of dates (tz-naive midnight). Open/close columns are
    tz-aware timestamps in the requested timezone.
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    idx = pd.bdate_range(start=start_date, end=end_date, freq="B")
    if len(idx) == 0:
        return pd.DataFrame(columns=["market_open", "market_close"], index=idx)

    tzinfo = pytz.timezone(tz_name)
    open_local = idx.tz_localize(tzinfo)
    close_local = open_local + pd.Timedelta(hours=23, minutes=59, seconds=59)

    return pd.DataFrame({"market_open": open_local, "market_close": close_local}, index=idx)


@dataclass
class TradingCalendar:
    """Trading calendar wrapper.

    Uses `pandas_market_calendars` when available; otherwise falls back to a Mon–Fri
    calendar (holidays ignored).

    Compatibility note
    ------------------
    Several downstream modules expect `.tz` to be a `pytz` timezone object with a
    `.localize()` method (e.g., Stage 2 expiry timestamp generation). To preserve
    that interface, this class exposes:

    - `tz_name`: timezone name string
    - `tz`: pytz timezone object
    """

    calendar_name: str = "CMEGlobex_Metals"
    tz_name: str = "US/Central"
    tz: pytz.BaseTzInfo = field(init=False)

    def __post_init__(self) -> None:
        self.tz = pytz.timezone(self.tz_name)
        self.calendar = mcal.get_calendar(self.calendar_name) if mcal is not None else None

    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get trading schedule between dates (inclusive).

        Returns a DataFrame with at least `market_open` and `market_close` columns.
        The columns are converted to the calendar timezone (`self.tz_name`) when possible.

        Notes
        -----
        - When using `pandas_market_calendars`, the schedule typically returns
          UTC timestamps; we convert columns to `self.tz_name`.
        - The fallback schedule ignores holidays.
        """
        if self.calendar is None:
            return _fallback_schedule(start_date, end_date, self.tz_name)

        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        for col in ("market_open", "market_close"):
            if col not in schedule.columns:
                continue
            series = schedule[col]
            if not pd.api.types.is_datetime64_any_dtype(series):
                # Empty schedules (e.g., weekends) can yield object dtype; skip conversion.
                continue
            if getattr(series.dt, "tz", None) is None:
                # Best-effort assumption: tz-naive schedule columns are UTC.
                series = series.dt.tz_localize("UTC")
            schedule[col] = series.dt.tz_convert(self.tz_name)

        return schedule

    def is_business_day(self, check_date: date) -> bool:
        schedule = self.get_schedule(check_date, check_date)
        return check_date in schedule.index.date

    def next_business_day(self, current_date: date) -> date:
        check_date = current_date + timedelta(days=1)

        if self.calendar is None:
            while check_date.weekday() >= 5:
                check_date += timedelta(days=1)
            return check_date

        schedule = self.get_schedule(check_date, current_date + timedelta(days=30))
        for d in schedule.index.date:
            if d > current_date:
                return d
        raise ValueError(f"No business day found after {current_date}")

    def prev_business_day(self, current_date: date) -> date:
        check_date = current_date - timedelta(days=1)

        if self.calendar is None:
            while check_date.weekday() >= 5:
                check_date -= timedelta(days=1)
            return check_date

        schedule = self.get_schedule(current_date - timedelta(days=30), check_date)
        for d in reversed(schedule.index.date):
            if d < current_date:
                return d
        raise ValueError(f"No business day found before {current_date}")

    def add_business_days(self, current_date: date, offset: int) -> date:
        """Add (or subtract) business days from a date.

        Offset of 0 returns the input date. Positive offsets move forward
        to the next business days; negative offsets move backward.
        """
        if offset == 0:
            return current_date

        step = 1 if offset > 0 else -1
        d = current_date
        for _ in range(abs(int(offset))):
            d = self.next_business_day(d) if step > 0 else self.prev_business_day(d)
        return d

    def get_business_days(self, start_date: date, end_date: date) -> pd.DatetimeIndex:
        schedule = self.get_schedule(start_date, end_date)
        return schedule.index

    def business_days_between(
        self,
        start_date: date,
        end_date: date,
        include_start: bool = True,
        include_end: bool = True,
    ) -> int:
        schedule = self.get_schedule(start_date, end_date)
        bdays = list(schedule.index.date)

        count = len(bdays)
        if not include_start and start_date in bdays:
            count -= 1
        if not include_end and end_date in bdays:
            count -= 1

        return max(count, 0)

    def days_to_expiry(self, as_of: date, expiry: date) -> int:
        """Calculate business days to expiry.

        Excludes start day, includes expiry day.
        """
        if as_of >= expiry:
            return 0
        return self.business_days_between(as_of, expiry, include_start=False, include_end=True)


def build_trading_calendar(
    calendar_name: str = "CMEGlobex_Metals",
    start_year: int = 2000,
    end_year: int = 2030,
) -> pd.DataFrame:
    """Build trading calendar DataFrame."""
    cal = TradingCalendar(calendar_name)
    schedule = cal.get_schedule(date(start_year, 1, 1), date(end_year, 12, 31))

    df = pd.DataFrame(
        {
            "date": schedule.index.date,
            "market_open": schedule.get("market_open"),
            "market_close": schedule.get("market_close"),
        }
    )

    return df


def save_trading_calendar(
    calendar_name: str,
    output_path: str,
    start_year: int = 2000,
    end_year: int = 2030,
) -> None:
    """Save trading calendar to parquet.

    Note: Requires a parquet engine (pyarrow or fastparquet).
    """
    df = build_trading_calendar(calendar_name, start_year, end_year)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Saved trading calendar to {output_path}: {len(df)} trading days")
