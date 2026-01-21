from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MeanReversionParams:
    dte_bin_size: int = 5
    lookback_days: int = 252 * 3
    entry_z: float = 1.0
    exit_z: float = 0.2
    max_hold_days: int | None = 20


def _mad(x: pd.Series) -> float:
    med = x.median()
    return float((x - med).abs().median())


def compute_dte_conditioned_zscore(
    df: pd.DataFrame,
    *,
    params: MeanReversionParams,
    value_col: str = "s_signal",
) -> pd.Series:
    """Compute a causal DTE-conditioned robust z-score for the spread level.

    Steps (all causal):
    1) bucket observations by DTE (near_dte_bdays) into bins of size `dte_bin_size`
    2) within each bin, compute rolling median and rolling MAD over `lookback_days`
    3) z = (s_signal - median) / MAD
    """
    dte = df["near_dte_bdays"].astype("float64")
    if value_col not in df.columns:
        raise ValueError(f"Missing column: {value_col}")
    s = df[value_col].astype("float64")

    bin_id = (dte // params.dte_bin_size).astype("Int64")

    out = pd.Series(index=df.index, dtype="float64")
    for b, idx in bin_id.groupby(bin_id).groups.items():
        if b is pd.NA:
            continue
        sub = s.loc[idx].copy()
        # Rolling window on time index (trade_date), bounded by `lookback_days` observations.
        roll = sub.rolling(window=params.lookback_days, min_periods=max(20, params.lookback_days // 10))
        med = roll.median()
        mad = roll.apply(_mad, raw=False)
        z = (sub - med) / mad.replace(0.0, np.nan)
        out.loc[idx] = z

    return out.replace([np.inf, -np.inf], np.nan)


def positions_from_zscore(
    z: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    max_hold_days: int | None = None,
) -> pd.Series:
    """Generate discrete MR positions in {-1,0,1} from z-score with hysteresis."""
    z = z.astype("float64")
    pos = pd.Series(index=z.index, dtype="float64")
    state = 0.0
    hold_days = 0
    for t, v in z.items():
        if not np.isfinite(v):
            pos.loc[t] = state
            continue
        if state == 0.0:
            if v >= entry_z:
                state = -1.0
                hold_days = 0
            elif v <= -entry_z:
                state = 1.0
                hold_days = 0
        elif state == 1.0:
            if v >= -exit_z:
                # exit long when z reverts back toward 0
                state = 0.0
        elif state == -1.0:
            if v <= exit_z:
                state = 0.0

        if state != 0.0 and max_hold_days is not None:
            hold_days += 1
            if hold_days >= max_hold_days:
                state = 0.0
                hold_days = 0
        pos.loc[t] = state
    return pos.fillna(0.0)
