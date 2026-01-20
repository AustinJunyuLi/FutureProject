"""Stage 4 data aggregation helpers.

Stage 4 backtests primarily operate on bucket-level panels. Some legacy
analyses (e.g., the Archive repo seasonality work) used a daily US-session
proxy computed from bucket data. This module provides a small, explicit
aggregation step so Stage 4 can test strategies on that daily proxy without
re-ingesting raw data.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Literal

import numpy as np
import pandas as pd


DailyPairPolicy = Literal["all", "mode", "strict"]


def build_us_session_daily_vwap_panel(
    spread_panel: pd.DataFrame,
    *,
    spread_col: str = "S1",
    dte_col: str = "F1_dte_bdays",
    buckets: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    volume_cols: Sequence[str] = ("F1_volume", "F2_volume"),
    pair_policy: DailyPairPolicy = "mode",
) -> pd.DataFrame:
    """Aggregate a bucket-level spread panel into a daily US-session VWAP proxy.

    Parameters
    ----------
    spread_panel:
        Bucket-level panel from Stage 2 (one row per trade_date x bucket).
    spread_col:
        Spread column to aggregate (e.g. ``"S1"``).
    dte_col:
        DTE column to carry to the daily proxy (default: ``"F1_dte_bdays"``).
        This is used by the :class:`~futures_curve.stage4.strategies.PreExpiryStrategy`.
    buckets:
        Buckets defining the US session. Archive used buckets 1..7.
    volume_cols:
        Columns to form VWAP weights. By default, weight = F1_volume + F2_volume.
    pair_policy:
        How to handle days where the spread's contract pair changes intraday.
        - ``"mode"``: choose the (near,far) pair with the highest total weight
          and compute the VWAP using only that pair's buckets (default; safest).
        - ``"strict"``: keep only days with exactly one observed pair.
        - ``"all"``: compute VWAP across all buckets regardless of pair (least strict).

    Returns
    -------
    pd.DataFrame
        Daily panel with one row per trade_date and columns:
        ``trade_date, bucket, {spread_col}, {dte_col}, {spread_col}_near, {spread_col}_far, daily_weight, bucket_obs, pair_obs``.
    """
    if pair_policy not in {"all", "mode", "strict"}:
        raise ValueError("pair_policy must be one of {'all','mode','strict'}")

    df = spread_panel.copy()
    if "trade_date" not in df.columns:
        raise ValueError("spread_panel must contain 'trade_date'")
    if "bucket" not in df.columns:
        raise ValueError("spread_panel must contain 'bucket'")
    if spread_col not in df.columns:
        raise ValueError(f"spread_panel must contain '{spread_col}'")
    if dte_col not in df.columns:
        raise ValueError(f"spread_panel must contain '{dte_col}'")

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df[df["bucket"].isin(list(buckets))].copy()
    df = df[df[spread_col].notna()].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "bucket",
                spread_col,
                dte_col,
                f"{spread_col}_near",
                f"{spread_col}_far",
                "daily_weight",
                "bucket_obs",
                "pair_obs",
            ]
        )

    # VWAP weights (default: F1_volume + F2_volume). Missing columns are treated as 0.
    weights = np.zeros(len(df), dtype="float64")
    for col in volume_cols:
        if col in df.columns:
            weights += pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    df["_w"] = weights

    # If all weights are zero (rare), fall back to equal weights.
    if float(df["_w"].sum()) <= 0.0:
        df["_w"] = 1.0

    near_col = f"{spread_col}_near"
    far_col = f"{spread_col}_far"
    if near_col not in df.columns or far_col not in df.columns:
        # Stage 2 spread panels should have these, but keep failure explicit.
        raise ValueError(f"spread_panel must contain '{near_col}' and '{far_col}'")

    # Build a per-row pair key (keep NaNs as-is, but stable stringification for grouping).
    df["_near"] = df[near_col].astype("string")
    df["_far"] = df[far_col].astype("string")
    df["_pair"] = df["_near"].fillna(pd.NA).astype("string") + "|" + df["_far"].fillna(pd.NA).astype("string")

    # Per-day pair stats (observed pairs within the session).
    pair_weight = df.groupby(["trade_date", "_pair"], dropna=False)["_w"].sum().reset_index()
    pair_obs = pair_weight.groupby("trade_date")["_pair"].nunique(dropna=False)

    if pair_policy == "strict":
        # Keep only days with a single observed pair.
        keep_days = pair_obs[pair_obs == 1].index
        df = df[df["trade_date"].isin(keep_days)].copy()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "bucket",
                    spread_col,
                    dte_col,
                    near_col,
                    far_col,
                    "daily_weight",
                    "bucket_obs",
                    "pair_obs",
                ]
            )
    elif pair_policy == "mode":
        # Choose the pair with max total weight per day and keep only those buckets.
        # Ties are broken deterministically by pair string ordering.
        pair_weight = pair_weight.sort_values(["trade_date", "_w", "_pair"], ascending=[True, False, True])
        mode_pair = pair_weight.groupby("trade_date", as_index=False).head(1)[["trade_date", "_pair"]]
        df = df.merge(mode_pair, on="trade_date", suffixes=("", "_mode"))
        df = df[df["_pair"] == df["_pair_mode"]].copy()

    # Daily VWAP on the selected bucket subset.
    vwap_num = (df[spread_col].astype("float64") * df["_w"]).groupby(df["trade_date"]).sum()
    vwap_den = df["_w"].groupby(df["trade_date"]).sum()
    daily_vwap = vwap_num / vwap_den.replace(0.0, np.nan)

    # Carry DTE + near/far labels (should be constant per day after filtering).
    daily_dte = df.groupby("trade_date")[dte_col].first()
    daily_near = df.groupby("trade_date")["_near"].first()
    daily_far = df.groupby("trade_date")["_far"].first()
    daily_weight = vwap_den
    bucket_obs = df.groupby("trade_date")["bucket"].count()

    out = pd.DataFrame(
        {
            "trade_date": daily_vwap.index,
            "bucket": 1,
            spread_col: daily_vwap.values,
            dte_col: daily_dte.reindex(daily_vwap.index).values,
            near_col: daily_near.reindex(daily_vwap.index).values,
            far_col: daily_far.reindex(daily_vwap.index).values,
            "daily_weight": daily_weight.reindex(daily_vwap.index).values,
            "bucket_obs": bucket_obs.reindex(daily_vwap.index).values,
            "pair_obs": pair_obs.reindex(daily_vwap.index).values,
        }
    ).reset_index(drop=True)

    # Ensure chronological order.
    return out.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

