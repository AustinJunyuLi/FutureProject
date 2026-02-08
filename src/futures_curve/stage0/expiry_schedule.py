"""Contract expiry schedule calculation.

The pipeline defines curve positions (F1..F12) strictly by expiry ordering, so
expiry-date correctness is foundational.

This implementation primarily relies on `pandas_market_calendars` (CME Globex
Metals calendar) to correctly handle exchange holidays. If that dependency is
missing, we provide a deterministic fallback (Mon-Fri, holidays ignored) so unit
tests can run in minimal environments.

The fallback is **not** adequate for reproducing the published research numbers
or for production use where holiday handling matters.

Expiry rules are loaded from ``config/expiry_rules.yaml`` so that adding a new
commodity requires zero code changes -- only a config entry and raw data.
"""

from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import yaml

from ..utils.month_codes import format_contract, parse_contract_code


CENTRAL_TZ = pytz.timezone("US/Central")
UTC_TZ = pytz.UTC
try:
    import pandas_market_calendars as mcal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mcal = None

# Default path for expiry rules config, relative to project root.
_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "config" / "expiry_rules.yaml"


def _load_expiry_rules(config_path: Path | str | None = None) -> dict[str, dict]:
    """Load expiry rules from YAML, returning a flat {SYMBOL: rule_dict} mapping.

    The YAML nests rules under exchanges -> sector -> symbol.  This helper
    flattens that into a single dict keyed by uppercase symbol.
    """
    if config_path is None:
        config_path = _DEFAULT_RULES_PATH
    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    rules: dict[str, dict] = {}
    exchanges = raw.get("exchanges", {})
    for _exchange_name, sectors in exchanges.items():
        for _sector_name, symbols in sectors.items():
            for symbol, rule_dict in symbols.items():
                rules[symbol.upper()] = rule_dict
    return rules


@dataclass
class ExpiryCalculator:
    """Calculate contract expiry dates for futures contracts.

    Rules are loaded from ``config/expiry_rules.yaml`` at construction time.
    The calculator dispatches ``compute_expiry`` to the appropriate rule
    function based on the symbol's configured rule name.

    Parameters
    ----------
    calendar_name : str
        ``pandas_market_calendars`` calendar name.
    rules_config : str | Path | None
        Path to expiry rules YAML.  ``None`` uses the default.
    rules : dict | None
        Pre-loaded rules dict (symbol -> rule_dict).  If provided,
        ``rules_config`` is ignored.
    """

    calendar_name: str = "CMEGlobex_Metals"
    rules_config: Optional[str | Path] = None
    rules: Optional[dict[str, dict]] = field(default=None, repr=False)

    # Dispatch table: rule name -> method name
    _RULE_DISPATCH: dict[str, str] = field(
        default_factory=lambda: {
            "third_last_business_day": "_rule_third_last_business_day",
            "third_business_day_before_25th": "_rule_third_business_day_before_25th",
            "three_business_days_before_first_of_month": "_rule_three_business_days_before_first_of_month",
            "business_day_before_15th": "_rule_business_day_before_15th",
        },
        repr=False,
        init=False,
    )

    def __post_init__(self) -> None:
        # Defer calendar creation to runtime and allow a deterministic fallback.
        self.calendar = mcal.get_calendar(self.calendar_name) if mcal is not None else None
        # Load rules from config if not provided directly.
        if self.rules is None:
            self.rules = _load_expiry_rules(self.rules_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_business_days_range(self, start_date: date, end_date: date) -> list[date]:
        """Get list of business days between start and end dates.

        Uses the CME Globex calendar when available; otherwise falls back to
        Mon-Fri business days (holidays ignored).
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

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def third_last_business_day(self, year: int, month: int) -> date:
        """Get third-to-last business day of a month.

        This rule is used by CME base metals such as HG.
        """
        return self._rule_third_last_business_day(year, month)

    def _rule_third_last_business_day(self, year: int, month: int) -> date:
        """Third-to-last business day of the delivery month (HG, GC, SI)."""
        start = date(year, month, 1)
        last_day = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()

        business_days = self._get_business_days_range(start, last_day)
        if len(business_days) < 3:
            raise ValueError(f"Not enough business days in {year}-{month:02d}")
        return business_days[-3]

    def _rule_third_business_day_before_25th(self, year: int, month: int) -> date:
        """Third business day prior to the 25th of the month preceding delivery (CL).

        CL expiry is in the month *before* the delivery month.  E.g., the
        February 2024 contract (CLG24) expires in January 2024.
        """
        # Move to prior month for the 25th reference
        if month == 1:
            ref_year, ref_month = year - 1, 12
        else:
            ref_year, ref_month = year, month - 1

        ref_date = date(ref_year, ref_month, 25)
        # Get business days of the reference month up to the 25th
        start = date(ref_year, ref_month, 1)
        business_days = self._get_business_days_range(start, ref_date)

        # If the 25th is not a business day, start from the last business day <= 25th
        # Then count back 3 business days from that point
        if business_days[-1] == ref_date:
            # 25th is a business day -- count back 3 from it (including it as day 0)
            idx = len(business_days) - 1
        else:
            # 25th is not a business day -- start from last bday before 25th
            idx = len(business_days) - 1

        # We need the 3rd business day *before* the reference point
        target_idx = idx - 3
        if target_idx < 0:
            raise ValueError(
                f"Not enough business days before 25th of {ref_year}-{ref_month:02d}"
            )
        return business_days[target_idx]

    def _rule_three_business_days_before_first_of_month(self, year: int, month: int) -> date:
        """Three business days prior to the first calendar day of the delivery month (NG)."""
        first_of_month = date(year, month, 1)
        # Get business days in the prior month (generous window)
        if month == 1:
            prior_year, prior_month = year - 1, 12
        else:
            prior_year, prior_month = year, month - 1

        start = date(prior_year, prior_month, 1)
        end = first_of_month
        business_days = self._get_business_days_range(start, end)

        # Exclude the first of delivery month itself if it's in the list
        business_days = [d for d in business_days if d < first_of_month]

        if len(business_days) < 3:
            raise ValueError(
                f"Not enough business days before {year}-{month:02d}-01"
            )
        return business_days[-3]

    def _rule_business_day_before_15th(self, year: int, month: int) -> date:
        """Business day preceding the 15th of the contract month (ZC, ZS, ZW)."""
        the_15th = date(year, month, 15)
        start = date(year, month, 1)
        business_days = self._get_business_days_range(start, the_15th)

        # Find the business day *before* the 15th
        before_15th = [d for d in business_days if d < the_15th]
        if not before_15th:
            raise ValueError(f"No business day before 15th of {year}-{month:02d}")
        return before_15th[-1]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_expiry(self, symbol: str, year: int, month: int) -> date:
        """Compute expiry date for a contract.

        Dispatches to the appropriate rule function based on the symbol's
        configured rule in ``config/expiry_rules.yaml``.
        """
        sym = symbol.upper()
        rule_dict = self.rules.get(sym) if self.rules else None
        if rule_dict is None:
            raise ValueError(
                f"No expiry rule configured for symbol '{sym}'. "
                f"Add it to config/expiry_rules.yaml."
            )

        rule_name = rule_dict["rule"]
        method_name = self._RULE_DISPATCH.get(rule_name)
        if method_name is None:
            raise ValueError(
                f"Unknown expiry rule '{rule_name}' for symbol '{sym}'. "
                f"Known rules: {list(self._RULE_DISPATCH.keys())}"
            )

        method = getattr(self, method_name)
        return method(year, month)

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
