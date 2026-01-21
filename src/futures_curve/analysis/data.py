from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..stage0.contract_specs import get_contract_spec
from .config import AnalysisConfig


@dataclass(frozen=True)
class SpreadDailySeries:
    spread: str
    df: pd.DataFrame
    contract_size: float
    dollars_per_tick: float


def _vwap_from_bucket_close(
    price: pd.Series,
    volume: pd.Series,
) -> float:
    vol = volume.astype("float64")
    px = price.astype("float64")

    mask = px.notna() & vol.notna() & (vol > 0)
    if not mask.any():
        return float("nan")

    vol = vol.loc[mask]
    px = px.loc[mask]
    denom = vol.sum()
    if denom <= 0:
        return float("nan")
    return float((px * vol).sum() / denom)


def load_spreads_panel(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_daily_spread_series(
    *,
    config: AnalysisConfig,
    spreads_panel: pd.DataFrame,
    spread_num: int,
) -> SpreadDailySeries:
    """Build daily spread series for one S{n} from bucket-level spreads_panel.

    Outputs a DataFrame indexed by trade_date with:
    - near/far contract ids
    - near/far dte_bdays
    - signal prices: Fnear_us_vwap, Ffar_us_vwap, S_us_vwap, S_pct_us_vwap
    - execution prices: Fnear_exec, Ffar_exec, S_exec (bucket close at execution_bucket)
    """
    if spread_num < 1:
        raise ValueError("spread_num must be >= 1")

    spread = f"S{spread_num}"
    near_price = f"F{spread_num}_price"
    far_price = f"F{spread_num + 1}_price"
    near_vol = f"F{spread_num}_volume"
    far_vol = f"F{spread_num + 1}_volume"

    near_contract = f"F{spread_num}_contract"
    far_contract = f"F{spread_num + 1}_contract"
    near_dte = f"F{spread_num}_dte_bdays"
    far_dte = f"F{spread_num + 1}_dte_bdays"

    cols_needed = [
        "trade_date",
        "bucket",
        near_price,
        far_price,
        near_vol,
        far_vol,
        near_contract,
        far_contract,
        near_dte,
        far_dte,
    ]
    missing = [c for c in cols_needed if c not in spreads_panel.columns]
    if missing:
        raise ValueError(f"spreads_panel missing columns: {missing}")

    df = spreads_panel[cols_needed].copy()

    # Apply date filters
    if config.start_date is not None:
        df = df[df["trade_date"].dt.date >= config.start_date]
    if config.end_date is not None:
        df = df[df["trade_date"].dt.date <= config.end_date]

    # Signal prices: US session buckets VWAP proxy for each leg, then spread.
    sig_df = df[df["bucket"].isin(config.signal_buckets)].copy()

    grouped = sig_df.groupby("trade_date", sort=True)
    records: list[dict[str, object]] = []
    for td, g in grouped:
        fnear_vwap = _vwap_from_bucket_close(g[near_price], g[near_vol])
        ffar_vwap = _vwap_from_bucket_close(g[far_price], g[far_vol])
        s_vwap = ffar_vwap - fnear_vwap if (np.isfinite(ffar_vwap) and np.isfinite(fnear_vwap)) else float("nan")
        s_pct = (s_vwap / fnear_vwap) if (np.isfinite(s_vwap) and np.isfinite(fnear_vwap) and fnear_vwap != 0) else float("nan")

        # Contracts/DTE should be constant across buckets for a trade_date; take first non-null.
        near_c = g[near_contract].dropna().iloc[0] if g[near_contract].notna().any() else None
        far_c = g[far_contract].dropna().iloc[0] if g[far_contract].notna().any() else None
        near_d = float(g[near_dte].dropna().iloc[0]) if g[near_dte].notna().any() else float("nan")
        far_d = float(g[far_dte].dropna().iloc[0]) if g[far_dte].notna().any() else float("nan")

        records.append(
            {
                "trade_date": td,
                "near_contract": near_c,
                "far_contract": far_c,
                "near_dte_bdays": near_d,
                "far_dte_bdays": far_d,
                "fnear_signal": fnear_vwap,
                "ffar_signal": ffar_vwap,
                "s_signal": s_vwap,
                "s_signal_pct": s_pct,
            }
        )
    daily = pd.DataFrame.from_records(records).sort_values("trade_date").set_index("trade_date")

    # Execution price:
    # - default: bucket close at `execution_bucket`
    # - if enabled: fall back to the earliest US-session bucket in [execution_bucket..7]
    #   where both legs have prints (reduces sample loss).
    exec_candidates = df[df["bucket"].between(config.execution_bucket, 7)].copy()
    exec_candidates = exec_candidates.sort_values(["trade_date", "bucket"])
    exec_candidates = exec_candidates[exec_candidates[near_price].notna() & exec_candidates[far_price].notna()].copy()

    if config.execution_fallback_to_earliest_us_bucket:
        exec_selected = exec_candidates.groupby("trade_date", as_index=False).head(1).copy()
    else:
        exec_selected = exec_candidates[exec_candidates["bucket"] == config.execution_bucket].copy()

    exec_selected = exec_selected.set_index("trade_date").sort_index()
    daily["exec_bucket"] = exec_selected["bucket"]
    daily["fnear_exec"] = exec_selected[near_price]
    daily["ffar_exec"] = exec_selected[far_price]
    daily["s_exec"] = daily["ffar_exec"] - daily["fnear_exec"]
    daily["s_exec_pct"] = (daily["s_exec"] / daily["fnear_exec"]).where(daily["fnear_exec"].notna() & (daily["fnear_exec"] != 0))

    # Drop dates where execution price is missing.
    daily = daily.dropna(subset=["s_exec"]).copy()

    spec = get_contract_spec(config.symbol)
    if spec is None:
        raise ValueError(f"Missing contract spec for symbol={config.symbol}")

    return SpreadDailySeries(
        spread=spread,
        df=daily,
        contract_size=spec.contract_size,
        dollars_per_tick=spec.point_value,
    )
