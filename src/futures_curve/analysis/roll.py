from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollFilterConfig:
    start_threshold: float = 0.25
    end_threshold: float = 0.75
    min_total_volume: float = 1.0
    use_smooth: bool = True


def load_roll_shares_daily(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["frequency"] == "daily"].copy()
    if df.empty:
        raise ValueError("roll_shares parquet has no daily rows")
    df = df.sort_values(["trade_date"]).set_index("trade_date")
    return df


def compute_roll_tradable_mask(
    roll_daily: pd.DataFrame,
    *,
    mode: str,
    cfg: RollFilterConfig,
) -> pd.Series:
    """Return a boolean Series indexed by trade_date for when trading is allowed.

    Modes:
    - "none": always tradable
    - "exclude_roll": tradable when share <= start OR share >= end
    - "pre_roll_only": tradable when share <= start
    - "post_roll_only": tradable when share >= end
    """
    mode = mode.lower()
    if mode not in {"none", "exclude_roll", "pre_roll_only", "post_roll_only"}:
        raise ValueError(f"Unknown roll filter mode: {mode}")

    if mode == "none":
        return pd.Series(True, index=roll_daily.index)

    share_col = "volume_share_smooth" if cfg.use_smooth else "volume_share"
    if share_col not in roll_daily.columns:
        raise ValueError(f"Missing roll share column: {share_col}")

    share = roll_daily[share_col].astype("float64")
    total = roll_daily.get("total_volume", pd.Series(np.nan, index=roll_daily.index)).astype("float64")

    valid = total >= float(cfg.min_total_volume)
    share = share.where(valid)

    start = float(cfg.start_threshold)
    end = float(cfg.end_threshold)

    if mode == "exclude_roll":
        tradable = (share <= start) | (share >= end)
    elif mode == "pre_roll_only":
        tradable = share <= start
    else:  # post_roll_only
        tradable = share >= end

    return tradable.fillna(False)

