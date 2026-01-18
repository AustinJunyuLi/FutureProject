"""Roll share + roll event detection.

This module implements roll timing proxies via liquidity migration between
front (F1) and second (F2) contracts, using deterministic expiry-ranked curve
labels (never volume-ranked).

Core metric (volume share):
    s(t) = V2(t) / (V1(t) + V2(t))

Where V1/V2 are the observed bucket (or daily) volumes of the contracts that
are labeled F1/F2 at time t.

Event definitions (EX-ANTE / causal, via persistence):
    roll_start_ts = first time s_smooth(t) > start_threshold for m consecutive units
    roll_peak_ts  = first time s_smooth(t) > peak_threshold  for m consecutive units
    roll_end_ts   = first time s_smooth(t) > end_threshold   for m consecutive units

IMPORTANT:
We define the event timestamp as the *confirmation time* (the m-th consecutive
observation), which is causal. We also store the run start timestamp for
diagnostics.

This module produces:
- roll share panels at bucket and daily frequency
- roll event tables (start/peak/end) per F1 contract cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


RollFrequency = Literal["bucket", "daily"]


@dataclass(frozen=True)
class RollShareConfig:
    """Configuration for roll share computation + event detection."""

    # Share thresholds
    start_threshold: float = 0.25
    peak_threshold: float = 0.50
    end_threshold: float = 0.75

    # Persistence (number of consecutive observations required to confirm)
    persistence: int = 2

    # Causal smoothing (rolling mean over last K observations; K=1 => none)
    smoothing_window: int = 3

    # Minimum total (V1+V2) volume required to compute share; else NaN
    min_total_volume: float = 1.0

    # Use strictly greater-than comparisons (matches prompt "s > threshold")
    strict_gt: bool = True


def _ensure_ts(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    s = df[col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    return s


def _select_daily_rows_from_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """Select one observation per trade_date for daily roll share.

    We use the last available US-session bucket (1-7) per trade_date as the
    daily reference point. If no US buckets exist for that trade_date, the day
    is dropped (not fillable under a US-session proxy convention).
    """
    if "trade_date" not in panel.columns or "bucket" not in panel.columns:
        raise ValueError("panel must include trade_date and bucket")

    us = panel[panel["bucket"].between(1, 7)].copy()
    if us.empty:
        return us

    # Buckets 1-7 are naturally ordered; pick max bucket available per trade_date.
    us = us.sort_values(["trade_date", "bucket"])
    return us.groupby("trade_date", as_index=False).tail(1).reset_index(drop=True)


class RollDetector:
    """Detect roll events from a curve panel (bucket-level deterministic strip)."""

    def __init__(self, config: Optional[RollShareConfig] = None):
        self.config = config or RollShareConfig()

    def build_roll_share_panel(
        self,
        curve_panel: pd.DataFrame,
        frequency: RollFrequency = "bucket",
        exclude_maintenance_bucket: bool = True,
    ) -> pd.DataFrame:
        """Build a roll share panel from a curve panel.

        Args:
            curve_panel: Output of Stage 2 curve construction with at least:
                trade_date, bucket, ts_end_utc,
                F1_contract, F2_contract,
                F1_volume, F2_volume
            frequency: "bucket" or "daily"
            exclude_maintenance_bucket: drop bucket 0 (maintenance) before processing
        """
        required = [
            "trade_date",
            "bucket",
            "ts_end_utc",
            "F1_contract",
            "F2_contract",
            "F1_volume",
            "F2_volume",
        ]
        missing = [c for c in required if c not in curve_panel.columns]
        if missing:
            raise ValueError(f"curve_panel missing required columns: {missing}")

        df = curve_panel.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ts_end_utc"] = _ensure_ts(df, "ts_end_utc")

        if exclude_maintenance_bucket:
            df = df[df["bucket"] != 0].copy()

        # Keep only columns needed for roll share + useful diagnostics.
        keep_cols = [
            "trade_date",
            "bucket",
            "ts_end_utc",
            "F1_contract",
            "F2_contract",
            "F1_volume",
            "F2_volume",
        ]
        if "F1_dte_bdays" in df.columns:
            keep_cols.append("F1_dte_bdays")
        if "F1_dte_hours" in df.columns:
            keep_cols.append("F1_dte_hours")
        df = df[keep_cols].copy()

        # Chronological ordering for any rolling ops.
        df = df.sort_values(["ts_end_utc"]).reset_index(drop=True)

        # Compute volume share with min volume filter.
        v1 = df["F1_volume"].astype("float64")
        v2 = df["F2_volume"].astype("float64")
        total = v1 + v2
        df["total_volume"] = total
        df["volume_share"] = np.where(
            total >= float(self.config.min_total_volume),
            v2 / total,
            np.nan,
        )

        # Optionally downsample to daily frequency.
        if frequency == "daily":
            df = _select_daily_rows_from_buckets(df)
            if df.empty:
                return df.assign(frequency="daily")
            df = df.sort_values(["ts_end_utc"]).reset_index(drop=True)

        # Causal smoothing per F1 contract cycle.
        k = int(self.config.smoothing_window)
        if k <= 1:
            df["volume_share_smooth"] = df["volume_share"]
        else:
            df["volume_share_smooth"] = df.groupby("F1_contract", sort=False)["volume_share"].transform(
                lambda s: s.rolling(window=k, min_periods=1).mean()
            )

        df["frequency"] = frequency
        return df

    def detect_roll_events(
        self,
        roll_share_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Detect roll_start/peak/end events per F1 contract cycle."""
        if roll_share_panel.empty:
            return pd.DataFrame()

        required = ["frequency", "ts_end_utc", "trade_date", "F1_contract", "F2_contract", "volume_share_smooth"]
        missing = [c for c in required if c not in roll_share_panel.columns]
        if missing:
            raise ValueError(f"roll_share_panel missing required columns: {missing}")

        df = roll_share_panel.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ts_end_utc"] = _ensure_ts(df, "ts_end_utc")
        df = df.sort_values(["ts_end_utc"]).reset_index(drop=True)

        events: list[dict] = []
        for (freq, f1_contract), g in df.groupby(["frequency", "F1_contract"], sort=False):
            g = g.sort_values(["ts_end_utc"]).reset_index(drop=True)
            s = g["volume_share_smooth"]

            start = self._first_persistent_crossing(g, s, self.config.start_threshold)
            peak = self._first_persistent_crossing(g, s, self.config.peak_threshold)
            end = self._first_persistent_crossing(g, s, self.config.end_threshold)

            # Diagnostic peak (argmax) — NOT ex-ante; keep for analysis only.
            argmax_idx = s.astype("float64").idxmax() if s.notna().any() else None
            if argmax_idx is not None and not pd.isna(argmax_idx):
                argmax_row = g.loc[int(argmax_idx)]
                argmax_ts = argmax_row["ts_end_utc"]
                argmax_share = float(argmax_row["volume_share_smooth"])
            else:
                argmax_ts = pd.NaT
                argmax_share = np.nan

            event = {
                "frequency": freq,
                "F1_contract": f1_contract,
                "f1_contract": f1_contract,  # backward compatibility for older outputs
                "start_threshold": self.config.start_threshold,
                "peak_threshold": self.config.peak_threshold,
                "end_threshold": self.config.end_threshold,
                "persistence": int(self.config.persistence),
                "smoothing_window": int(self.config.smoothing_window),
                "min_total_volume": float(self.config.min_total_volume),
                "strict_gt": bool(self.config.strict_gt),
                # Ex-ante / causal event timestamps (confirmation times)
                "roll_start_ts_utc": start.confirm_ts if start else pd.NaT,
                "roll_peak_ts_utc": peak.confirm_ts if peak else pd.NaT,
                "roll_end_ts_utc": end.confirm_ts if end else pd.NaT,
                # Diagnostic run starts (not causal in the sense of "known at the time")
                "roll_start_run_start_ts_utc": start.run_start_ts if start else pd.NaT,
                "roll_peak_run_start_ts_utc": peak.run_start_ts if peak else pd.NaT,
                "roll_end_run_start_ts_utc": end.run_start_ts if end else pd.NaT,
                # Diagnostics (argmax)
                "roll_argmax_ts_utc": argmax_ts,
                "roll_argmax_share": argmax_share,
            }

            # Attach DTE metrics at confirm times when available.
            for label, crossing in [("start", start), ("peak", peak), ("end", end)]:
                if crossing is None:
                    event[f"roll_{label}_trade_date"] = pd.NaT
                    event[f"roll_{label}_bucket"] = np.nan
                    event[f"roll_{label}_f2_contract"] = None
                    event[f"roll_{label}_share"] = np.nan
                    event[f"roll_{label}_f1_dte_bdays"] = np.nan
                    event[f"roll_{label}_f1_dte_hours"] = np.nan
                    continue

                row = crossing.confirm_row
                event[f"roll_{label}_trade_date"] = row["trade_date"]
                event[f"roll_{label}_bucket"] = row.get("bucket", np.nan)
                event[f"roll_{label}_f2_contract"] = row.get("F2_contract")
                event[f"roll_{label}_share"] = float(row["volume_share_smooth"]) if pd.notna(row["volume_share_smooth"]) else np.nan
                event[f"roll_{label}_f1_dte_bdays"] = float(row.get("F1_dte_bdays")) if "F1_dte_bdays" in row else np.nan
                event[f"roll_{label}_f1_dte_hours"] = float(row.get("F1_dte_hours")) if "F1_dte_hours" in row else np.nan

            events.append(event)

        return pd.DataFrame(events)

    @dataclass(frozen=True)
    class _Crossing:
        confirm_ts: pd.Timestamp
        run_start_ts: pd.Timestamp
        confirm_row: pd.Series

    def _first_persistent_crossing(
        self,
        group: pd.DataFrame,
        s_smooth: pd.Series,
        threshold: float,
    ) -> Optional[_Crossing]:
        """Find first (causal) persistent crossing confirmation.

        Returns the confirmation timestamp (m-th consecutive > threshold) and the
        run start timestamp for diagnostics.
        """
        m = int(self.config.persistence)
        if m <= 0:
            raise ValueError("persistence must be >= 1")

        gt = s_smooth > threshold if self.config.strict_gt else s_smooth >= threshold

        streak = 0
        run_start_i: Optional[int] = None

        for i, ok in enumerate(gt.fillna(False).to_numpy()):
            if ok:
                if streak == 0:
                    run_start_i = i
                streak += 1
                if streak >= m:
                    confirm_row = group.iloc[i]
                    run_start_row = group.iloc[int(run_start_i)] if run_start_i is not None else confirm_row
                    return self._Crossing(
                        confirm_ts=confirm_row["ts_end_utc"],
                        run_start_ts=run_start_row["ts_end_utc"],
                        confirm_row=confirm_row,
                    )
            else:
                streak = 0
                run_start_i = None

        return None


def build_roll_volume_panel(
    bucket_data: pd.DataFrame,  # kept for backward compatibility; unused now
    curve_panel: pd.DataFrame,
    config: Optional[RollShareConfig] = None,
) -> pd.DataFrame:
    """Backward compatible helper (returns DAILY roll share panel).

    Historically this pipeline computed daily roll share from aggregated contract
    volumes. The deterministic curve panel already contains F1/F2 bucket volumes,
    so we compute the daily roll share directly from it.
    """
    detector = RollDetector(config=config)
    return detector.build_roll_share_panel(curve_panel, frequency="daily")


def detect_rolls(
    roll_volume_panel: pd.DataFrame,
    threshold: float = 0.2,  # legacy arg (maps to start_threshold)
) -> pd.DataFrame:
    """Backward compatible helper for roll detection.

    This maps the legacy single `threshold` to `start_threshold` and keeps the
    default peak/end thresholds at 0.50/0.75.
    """
    cfg = RollShareConfig(start_threshold=float(threshold))
    detector = RollDetector(config=cfg)
    return detector.detect_roll_events(roll_volume_panel)
