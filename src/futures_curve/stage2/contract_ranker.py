"""Deterministic futures curve construction (F1..F12) by expiry timestamp.

Key principles:
- Curve membership is determined by *expiry timestamp only* (never by volume).
- Eligibility is strict: a contract is eligible iff `expiry_ts_utc > as_of_ts_utc`.
- Curve membership is independent of whether a contract traded in that bucket.
  If a contract has no print at a bucket timestamp, prices are NA and volumes are 0.

This module provides:
- ContractRanker: expiry-ts utilities + ranking
- build_curve_panel: deterministic bucket-level curve panel builder
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..stage0.expiry_schedule import ExpiryCalculator
from ..stage0.trading_calendar import TradingCalendar
from ..utils.timezone import bucket_start_end_local, bucket_sort_key, to_utc


class ContractRanker:
    """Rank contracts by expiry timestamp for curve construction."""

    def __init__(self, max_contracts: int = 12):
        self.max_contracts = max_contracts
        self.expiry_calc = ExpiryCalculator()
        self._expiry_ts_utc_cache: dict[str, pd.Timestamp] = {}
        self._expiry_ts_local_cache: dict[str, pd.Timestamp] = {}

    def get_expiry_ts_utc(self, contract: str) -> pd.Timestamp:
        if contract not in self._expiry_ts_utc_cache:
            self._expiry_ts_utc_cache[contract] = self.expiry_calc.compute_expiry_timestamp_utc_for_contract(
                contract
            )
        return self._expiry_ts_utc_cache[contract]

    def get_expiry_ts_local(self, contract: str) -> pd.Timestamp:
        if contract not in self._expiry_ts_local_cache:
            self._expiry_ts_local_cache[contract] = self.expiry_calc.compute_expiry_timestamp_local_for_contract(
                contract
            )
        return self._expiry_ts_local_cache[contract]

    def rank_contracts(
        self,
        contracts: list[str],
        as_of: date | datetime | pd.Timestamp,
    ) -> dict[str, int]:
        """Rank contracts by expiry timestamp at a given `as_of`.

        Args:
            contracts: Contract codes
            as_of: Reference timestamp (date implies 00:00 CT)
        """
        if isinstance(as_of, date) and not isinstance(as_of, datetime):
            as_of_ts = pd.Timestamp(datetime.combine(as_of, datetime.min.time()))
        else:
            as_of_ts = pd.Timestamp(as_of)
        as_of_utc = to_utc(as_of_ts)

        eligible: list[tuple[pd.Timestamp, str]] = []
        for c in contracts:
            try:
                exp_utc = self.get_expiry_ts_utc(c)
            except (ValueError, KeyError):
                continue
            if exp_utc > as_of_utc:
                eligible.append((exp_utc, c))

        eligible.sort(key=lambda x: x[0])
        ranks: dict[str, int] = {}
        for i, (_, c) in enumerate(eligible[: self.max_contracts], start=1):
            ranks[c] = i
        return ranks


def _build_business_day_index(
    calendar: TradingCalendar,
    start_date: date,
    end_date: date,
) -> dict[date, int]:
    """Map business day -> index for fast business-day distance calculations."""
    bdays = calendar.get_business_days(start_date, end_date)
    # schedule index is Timestamp; convert to python date
    return {d.date(): i for i, d in enumerate(bdays)}


def build_curve_panel(
    bucket_data: pd.DataFrame,
    symbol: str,
    expiry_table: Optional[pd.DataFrame] = None,
    max_contracts: int = 12,
    exclude_maintenance_bucket: bool = True,
    calendar: Optional[TradingCalendar] = None,
) -> pd.DataFrame:
    """Build a deterministic bucket-level curve panel.

    Inputs:
        bucket_data: contract-level bucket OHLCV with columns at least:
            trade_date, bucket, contract, close, volume

    Outputs:
        One row per (trade_date, bucket) with:
            - ts_end_local, ts_end_utc
            - F1..F12 contract IDs (deterministic)
            - F1..F12 close prices (observed; NA if missing)
            - F1..F12 volumes (observed; 0 if missing)
            - F1..F12 expiry_ts_local/utc
            - F1..F12 dte_hours (bucket timestamp)
            - F1..F12 dte_bdays (trade_date-based)
    """
    df = bucket_data.copy()
    if "trade_date" not in df.columns or "bucket" not in df.columns or "contract" not in df.columns:
        raise ValueError("bucket_data must include trade_date, bucket, contract")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if exclude_maintenance_bucket:
        df = df[df["bucket"] != 0].copy()

    # Universe of contracts we will rank over (discovered from data).
    universe = sorted(df["contract"].dropna().unique().tolist())
    if not universe:
        return pd.DataFrame()

    ranker = ContractRanker(max_contracts=max_contracts)
    calendar = calendar or TradingCalendar("CMEGlobex_Metals")

    # Precompute expiry timestamps for universe and sort by expiry.
    expiry_pairs: list[tuple[np.datetime64, str, pd.Timestamp, pd.Timestamp]] = []
    for c in universe:
        try:
            exp_local = ranker.get_expiry_ts_local(c)
            exp_utc = ranker.get_expiry_ts_utc(c)
        except (ValueError, KeyError):
            continue
        expiry_pairs.append((exp_utc.to_datetime64(), c, exp_local, exp_utc))

    if not expiry_pairs:
        return pd.DataFrame()

    expiry_pairs.sort(key=lambda x: x[0])
    expiry_ts_utc_sorted = np.array([p[0] for p in expiry_pairs])
    contracts_sorted = [p[1] for p in expiry_pairs]

    # expiry lookup for joins
    expiry_lookup = pd.DataFrame(
        {
            "contract": [p[1] for p in expiry_pairs],
            "expiry_ts_local": [p[2] for p in expiry_pairs],
            "expiry_ts_utc": [p[3] for p in expiry_pairs],
        }
    )
    expiry_lookup["expiry_date"] = pd.to_datetime(expiry_lookup["expiry_ts_local"]).dt.date

    # Build unique (trade_date, bucket) timeline and its bucket-end timestamps.
    # IMPORTANT: bucket numbers are not chronological (bucket 8/9 occur on the
    # prior calendar day of the trade_date). Use bucket_sort_key for ordering.
    times = df[["trade_date", "bucket"]].drop_duplicates()
    times["bucket_order"] = times["bucket"].map(bucket_sort_key)
    times = times.sort_values(["trade_date", "bucket_order"]).reset_index(drop=True)
    ts_end_local = []
    ts_end_utc = []
    for td, b in zip(times["trade_date"], times["bucket"]):
        end_local = bucket_start_end_local(td.date(), int(b))[1]
        ts_end_local.append(end_local)
        ts_end_utc.append(end_local.tz_convert("UTC"))
    times["ts_end_local"] = ts_end_local
    times["ts_end_utc"] = ts_end_utc
    times = times.drop(columns=["bucket_order"])

    # Assign deterministic contracts per timestamp.
    for i in range(1, max_contracts + 1):
        times[f"F{i}_contract"] = None

    for idx, t_utc in enumerate(times["ts_end_utc"]):
        start = int(np.searchsorted(expiry_ts_utc_sorted, t_utc.to_datetime64(), side="right"))
        strip = contracts_sorted[start : start + max_contracts]
        for j, c in enumerate(strip, start=1):
            times.iat[idx, times.columns.get_loc(f"F{j}_contract")] = c

    # Observed close/volume at each (trade_date, bucket, contract)
    obs = df[["trade_date", "bucket", "contract"]].copy()
    if "close" in df.columns:
        obs["price"] = df["close"].astype("float64")
    else:
        obs["price"] = np.nan
    if "volume" in df.columns:
        obs["volume"] = df["volume"].astype("float64")
    else:
        obs["volume"] = 0.0

    # Long form mapping of (trade_date,bucket,pos)->contract
    contract_cols = [f"F{i}_contract" for i in range(1, max_contracts + 1)]
    long = times.melt(
        id_vars=["trade_date", "bucket", "ts_end_local", "ts_end_utc"],
        value_vars=contract_cols,
        var_name="position",
        value_name="contract",
    ).dropna(subset=["contract"])
    long["pos_num"] = long["position"].str.extract(r"F(\d+)_contract").astype(int)

    # Join prices/volumes + expiry timestamps
    long = long.merge(obs, on=["trade_date", "bucket", "contract"], how="left")
    long = long.merge(expiry_lookup, on="contract", how="left")

    # Normalize missingness rules
    long["volume"] = long["volume"].fillna(0.0)

    # DTE hours at bucket end timestamp
    long["dte_hours"] = (long["expiry_ts_utc"] - long["ts_end_utc"]).dt.total_seconds() / 3600.0

    # DTE business days at trade_date (not bucket-specific)
    # Precompute business-day index over full span of trade_dates..max(expiry_date).
    start_date = times["trade_date"].min().date()
    end_date = max(expiry_lookup["expiry_date"].dropna().max(), start_date)
    bday_index = _build_business_day_index(calendar, start_date, end_date)

    td_dates = long["trade_date"].dt.date
    exp_dates = long["expiry_date"]
    td_idx = td_dates.map(bday_index)
    exp_idx = exp_dates.map(bday_index)
    long["dte_bdays"] = exp_idx - td_idx

    # Pivot back to wide form
    idx_cols = ["trade_date", "bucket", "ts_end_local", "ts_end_utc"]

    def _pivot(values_col: str, prefix: str) -> pd.DataFrame:
        wide = long.pivot_table(index=idx_cols, columns="pos_num", values=values_col, aggfunc="first")
        wide.columns = [f"{prefix}{int(c)}" for c in wide.columns]
        return wide

    wide_price = _pivot("price", "F")
    wide_price = wide_price.rename(columns={c: f"{c}_price" for c in wide_price.columns})
    wide_vol = _pivot("volume", "F")
    wide_vol = wide_vol.rename(columns={c: f"{c}_volume" for c in wide_vol.columns})
    wide_exp_local = _pivot("expiry_ts_local", "F")
    wide_exp_local = wide_exp_local.rename(columns={c: f"{c}_expiry_ts_local" for c in wide_exp_local.columns})
    wide_exp_utc = _pivot("expiry_ts_utc", "F")
    wide_exp_utc = wide_exp_utc.rename(columns={c: f"{c}_expiry_ts_utc" for c in wide_exp_utc.columns})
    wide_dte_h = _pivot("dte_hours", "F")
    wide_dte_h = wide_dte_h.rename(columns={c: f"{c}_dte_hours" for c in wide_dte_h.columns})
    wide_dte_b = _pivot("dte_bdays", "F")
    wide_dte_b = wide_dte_b.rename(columns={c: f"{c}_dte_bdays" for c in wide_dte_b.columns})

    out = times.set_index(idx_cols)
    out = out.join(wide_price, how="left").join(wide_vol, how="left")
    out = out.join(wide_exp_local, how="left").join(wide_exp_utc, how="left")
    out = out.join(wide_dte_h, how="left").join(wide_dte_b, how="left")
    out = out.reset_index()

    # Enforce volume type (int-ish) and missing price semantics.
    for i in range(1, max_contracts + 1):
        vcol = f"F{i}_volume"
        if vcol in out.columns:
            out[vcol] = out[vcol].fillna(0.0)

    # Sort chronologically (bucket numbers are not time-ordered).
    return out.sort_values(["ts_end_utc"]).reset_index(drop=True)


def build_daily_curve_panel(
    daily_data: pd.DataFrame,
    symbol: str,
    max_contracts: int = 12,
) -> pd.DataFrame:
    """Compatibility helper (daily curve panel).

    The primary pipeline uses bucket-level curve construction. Keep a minimal
    daily panel to support any legacy consumers.
    """
    daily_data = daily_data.copy()
    daily_data["trade_date"] = pd.to_datetime(daily_data["trade_date"])
    universe = sorted(daily_data["contract"].dropna().unique().tolist())
    if not universe:
        return pd.DataFrame()

    ranker = ContractRanker(max_contracts=max_contracts)
    expiry_pairs = [(ranker.get_expiry_ts_utc(c).to_datetime64(), c) for c in universe]
    expiry_pairs.sort(key=lambda x: x[0])
    expiry_ts_utc_sorted = np.array([p[0] for p in expiry_pairs])
    contracts_sorted = [p[1] for p in expiry_pairs]

    rows = []
    for td in daily_data["trade_date"].drop_duplicates().sort_values():
        as_of = pd.Timestamp(datetime.combine(td.date(), datetime.min.time()))
        as_of_utc = to_utc(as_of).to_datetime64()
        start = int(np.searchsorted(expiry_ts_utc_sorted, as_of_utc, side="right"))
        strip = contracts_sorted[start : start + max_contracts]
        row = {"trade_date": td}
        for i, c in enumerate(strip, start=1):
            row[f"F{i}_contract"] = c
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
