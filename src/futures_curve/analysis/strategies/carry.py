from __future__ import annotations

import numpy as np
import pandas as pd


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

