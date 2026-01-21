from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series as a fraction from running peak (0..1)."""
    equity = equity.astype("float64")
    running_max = equity.cummax()
    dd = (running_max - equity) / running_max.replace(0.0, np.nan)
    return dd.fillna(0.0)


def compute_sharpe_annualized(returns: pd.Series, trading_days: int = 252) -> float:
    r = returns.astype("float64")
    mu = r.mean()
    sigma = r.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return float((mu / sigma) * np.sqrt(trading_days))


def compute_turnover(position: pd.Series) -> float:
    """Average daily turnover fraction (0..1-ish) based on position changes.

    Defined as mean_t |Δpos_t| / (|pos_t| + |pos_{t-1}| + eps).
    This is scale-invariant and can be inverted as a rough holding-period proxy.
    """
    pos = position.astype("float64").fillna(0.0)
    delta = pos.diff().abs().fillna(pos.abs())
    denom = (pos.abs() + pos.shift(1).abs()).fillna(pos.abs())
    turnover = delta / denom.replace(0.0, np.nan)
    return float(turnover.replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
