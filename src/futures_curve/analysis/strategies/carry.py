from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SpreadDailySeries
from ..params import CarryClippedParams, CarrySignParams
from .base import BaseStrategy


def compute_annualized_carry_signal(
    df: pd.DataFrame,
    *,
    min_spacing_bdays: int = 5,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualized carry proxy for an adjacent spread.

    Uses normalized spread level (S / F_near) divided by time between maturities.

    Convention:
    - Positive `s_signal_pct` (contango) implies *negative* carry for long outrights.
    - For a spread trade, we take the *roll-down* direction: short contango, long backwardation.

    Returns:
        Directional signal in {-1, +1} with 0 when unavailable.
    """
    s_pct = df["s_signal_pct"].astype("float64")
    spacing = (df["far_dte_bdays"] - df["near_dte_bdays"]).astype("float64")
    spacing_years = spacing / float(trading_days_per_year)

    annualized = s_pct / spacing_years.replace(0.0, np.nan)

    # Roll-down direction: short if carry positive contango, long if backwardation.
    direction = -np.sign(annualized)
    direction = direction.where(spacing >= float(min_spacing_bdays))
    return direction.fillna(0.0)


def _compute_carry_series(
    df: pd.DataFrame, trading_days_per_year: int = 252
) -> pd.Series:
    """Compute annualized carry from spread DataFrame.

    Shared helper used by both CarrySignStrategy and CarryClippedStrategy.
    Returns the raw annualized carry (with inf/nan cleaned).
    """
    s_pct = df["s_signal_pct"].astype("float64")
    spacing = (df["far_dte_bdays"] - df["near_dte_bdays"]).astype("float64")
    spacing_years = spacing / float(trading_days_per_year)
    carry = (s_pct / spacing_years.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    return carry


class CarrySignStrategy(BaseStrategy):
    """Sign-based carry: signal = -sign(carry) * 1[|carry| >= threshold]."""

    @property
    def name(self) -> str:
        return "carry_sign"

    def generate_signal(
        self,
        series: SpreadDailySeries,
        params: CarrySignParams,
        trading_days_per_year: int = 252,
    ) -> pd.Series:
        df = series.df
        carry = _compute_carry_series(df, trading_days_per_year)
        sig = (
            -np.sign(carry)
            .where(carry.abs() >= float(params.threshold), 0.0)
            .fillna(0.0)
        )
        return sig.astype("float64")

    def default_param_grid(self) -> list[CarrySignParams]:
        return [
            CarrySignParams(threshold=t)
            for t in [0.02, 0.05, 0.08]
        ]

    def positions_from_signal(
        self, signal: pd.Series, params: CarrySignParams
    ) -> pd.Series:
        # For carry, signal IS the position (no state/hysteresis).
        return signal

    def fold_params_dict(
        self, params: CarrySignParams, direction: int
    ) -> dict[str, object]:
        return {
            "type": "sign",
            "threshold": params.threshold,
            "scale": None,
            "direction": direction,
        }


class CarryClippedStrategy(BaseStrategy):
    """Clipped continuous carry: signal = clip(-carry/scale, -1, 1) * 1[|carry| >= threshold]."""

    @property
    def name(self) -> str:
        return "carry_clipped"

    def generate_signal(
        self,
        series: SpreadDailySeries,
        params: CarryClippedParams,
        trading_days_per_year: int = 252,
    ) -> pd.Series:
        df = series.df
        carry = _compute_carry_series(df, trading_days_per_year)
        raw = -(carry / float(params.scale))
        sig = (
            raw.clip(lower=-1.0, upper=1.0)
            .where(carry.abs() >= float(params.threshold), 0.0)
            .fillna(0.0)
        )
        return sig.astype("float64")

    def default_param_grid(self) -> list[CarryClippedParams]:
        return [
            CarryClippedParams(threshold=t, scale=s)
            for t in [0.02, 0.05, 0.08]
            for s in [0.05, 0.10, 0.20]
        ]

    def positions_from_signal(
        self, signal: pd.Series, params: CarryClippedParams
    ) -> pd.Series:
        return signal

    def fold_params_dict(
        self, params: CarryClippedParams, direction: int
    ) -> dict[str, object]:
        return {
            "type": "clipped",
            "threshold": params.threshold,
            "scale": params.scale,
            "direction": direction,
        }
