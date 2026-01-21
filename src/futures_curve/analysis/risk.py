from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_volatility(
    pnl_1_contract: pd.Series,
    *,
    halflife_days: int,
    min_periods: int,
) -> pd.Series:
    """EWMA stdev of 1-contract daily PnL in dollars."""
    x = pnl_1_contract.astype("float64").fillna(0.0)
    # Use pandas ewm with halflife for variance estimate.
    var = x.ewm(halflife=halflife_days, min_periods=min_periods, adjust=False).var(bias=False)
    vol = np.sqrt(var)
    return vol


def position_from_vol_target(
    *,
    signal: pd.Series,
    vol_per_contract: pd.Series,
    target_dollar_vol: float,
    allow_fractional: bool,
) -> pd.Series:
    """Scale a direction signal into contract counts via vol targeting."""
    signal = signal.astype("float64").fillna(0.0)
    vol = vol_per_contract.astype("float64")
    raw = signal * (target_dollar_vol / vol.replace(0.0, np.nan))
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not allow_fractional:
        raw = raw.round()
    return raw

