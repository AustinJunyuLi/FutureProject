"""Contract expiry schedule calculation.

The pipeline defines curve positions (F1..F12) strictly by expiry ordering, so
expiry-date correctness is foundational.

This implementation primarily relies on `pandas_market_calendars` (CME Globex
Metals calendar) to correctly handle exchange holidays. If that dependency is
missing, we provide a deterministic fallback (Mon–Fri, holidays ignored) so unit
tests can run in minimal environments.

The fallback is **not** adequate for reproducing the published research numbers
or for production use where holiday handling matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

from ..utils.month_codes import format_contract, parse_contract_code



CENTRAL_TZ = pytz.timezone("US/Central")
UTC_TZ = pytz.UTC
try:
    import pandas_market_calendars as mcal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mcal = None


@dataclass
class ExpiryCalculator:
    """Calculate contract expiry dates for futures contracts."""

    calendar_name: str = "CMEGlobex_Metals"

    def __post_init__(self) -> None:
        # Defer calendar creation to runtime and allow a deterministic fallback.
        self.calendar = mcal.get_calendar(self.calendar_name) if mcal is not None else None

    def _get_business_days_range(self, start_date: date, end_date: date) -> list[date]:
        """Get list of business days between start and end dates.

        Uses the CME Globex calendar when available; otherwise falls back to
        Mon–Fri business days (holidays ignored).
        """
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        if self.calendar is None:
            return [d.date() for d in pd.bdate_range(start=start_date, end=end_date, freq="B")]

        # For contract expiry calculations, CME product calendars can differ from
        # the exchange's *trading* calendar on a small set of dates (notably
        # Good Friday, which is typically a non-trading day but still counted as
        # a business day for some expiry rules). The Archive codebase uses the
        # CME contract calendar directly; empirically, the only mismatch in our
        # HG dataset was Good Friday-in-March edge cases.
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
        open_days = set(schedule.index.date)

        weekdays = pd.bdate_range(start=start_date, end=end_date, freq="B")

        try:
            from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday

            class _GoodFridayCalendar(AbstractHolidayCalendar):
                rules = [GoodFriday]

            good_fridays = set(_GoodFridayCalendar().holidays(start=str(start_date), end=str(end_date)).date)
        except Exception:  # pragma: no cover - defensive: holiday API should be available in pandas
            good_fridays = set()

        business_days = [d.date() for d in weekdays if (d.date() in open_days) or (d.date() in good_fridays)]
        return business_days

    def third_last_business_day(self, year: int, month: int) -> date:
        """Get third-to-last business day of a month.

        This rule is used by CME base metals such as HG.
        """
        start = date(year, month, 1)
        # Last calendar day of the month (timezone/calendar not relevant at day granularity).
        last_day = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()

        business_days = self._get_business_days_range(start, last_day)
        if len(business_days) < 3:
            raise ValueError(f"Not enough business days in {year}-{month:02d}")
        return business_days[-3]

    def compute_expiry(self, symbol: str, year: int, month: int) -> date:
        """Compute expiry date for a contract.

        For now, HG uses the third-to-last business day rule.
        """
        if symbol.upper() == "HG":
            return self.third_last_business_day(year, month)
        raise ValueError(f"Unsupported symbol: {symbol}")

    def compute_expiry_for_contract(self, contract: str) -> date:
        """Compute expiry date given a contract code."""
        info = parse_contract_code(contract)
        if info is None:
            raise ValueError(f"Invalid contract code: {contract}")
        return self.compute_expiry(info.symbol, info.year_full, info.month)


    def compute_expiry_timestamp_local_for_contract(
        self,
        contract: str,
        expiry_hour_local: int = 16,
    ) -> pd.Timestamp:
        """Compute a timezone-aware local expiry timestamp for a contract.

        The project standardizes on an expiry timestamp expressed in the exchange
        timezone (US/Central) at `expiry_hour_local:00:00`. This timestamp is
        used for:
        - eligibility (expiry_ts > as_of_ts)
        - DTE hours

        Notes
        -----
        The precise intraday expiry time can differ by product and exchange.
        For the purposes of this repository's deterministic curve labeling, this
        is a configurable convention.
        """
        expiry_date = self.compute_expiry_for_contract(contract)
        dt_local = datetime.combine(expiry_date, time(hour=int(expiry_hour_local), minute=0, second=0))
        return pd.Timestamp(CENTRAL_TZ.localize(dt_local))

    def compute_expiry_timestamp_utc_for_contract(
        self,
        contract: str,
        expiry_hour_local: int = 16,
    ) -> pd.Timestamp:
        """Compute the UTC expiry timestamp for a contract."""
        local_ts = self.compute_expiry_timestamp_local_for_contract(contract, expiry_hour_local=expiry_hour_local)
        return local_ts.tz_convert(UTC_TZ)


def build_expiry_table(
    symbol: str,
    start_year: int,
    end_year: int,
    calendar_name: str = "CMEGlobex_Metals",
) -> pd.DataFrame:
    """Build expiry table for a symbol."""
    calc = ExpiryCalculator(calendar_name)

    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            try:
                expiry_date = calc.compute_expiry(symbol, year, month)
            except Exception:
                continue

            records.append(
                {
                    "contract": format_contract(symbol, year, month),
                    "symbol": symbol,
                    "year": year,
                    "month": month,
                    "expiry_date": expiry_date,
                }
            )

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df

    df = df.sort_values("expiry_date").reset_index(drop=True)
    return df


def save_expiry_table(df: pd.DataFrame, output_path: str) -> None:
    """Save expiry table to parquet."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved expiry table to {output_path}: {len(df)} contracts")
