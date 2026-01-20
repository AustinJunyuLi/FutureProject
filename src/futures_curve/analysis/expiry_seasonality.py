"""Expiry-anchored seasonality helpers.

These utilities build daily spread series from Stage 2 bucket panels and
align observations by business-day DTE (days-to-expiry), using the near-leg
DTE for the chosen spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


US_SESSION_BUCKETS = (1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class DailySeriesConfig:
    """Configuration for building daily series from bucket panels."""

    source: str = "bucket1"  # "us_vwap" or "bucket1"
    execution_shift_bdays: int = 1  # shift forward (next day) for tradeable entry
    buckets: Sequence[int] = US_SESSION_BUCKETS


def _spread_index(spread: str) -> int:
    if not spread.upper().startswith("S"):
        raise ValueError("spread must be like S1, S2, ...")
    try:
        return int(spread[1:])
    except ValueError as exc:
        raise ValueError("spread must be like S1, S2, ...") from exc


def _volume_cols_for_spread(spread: str) -> tuple[str, str]:
    i = _spread_index(spread)
    return f"F{i}_volume", f"F{i+1}_volume"


def _near_far_cols(spread: str) -> tuple[str, str]:
    return f"{spread}_near", f"{spread}_far"


def _dte_col_for_spread(spread: str, spread_panel: pd.DataFrame) -> str:
    spread = spread.upper()
    spread_dte = f"{spread}_dte_bdays"
    if spread_dte in spread_panel.columns:
        return spread_dte
    if "F1_dte_bdays" in spread_panel.columns:
        return "F1_dte_bdays"
    raise ValueError("spread_panel missing business-day DTE columns")


def build_daily_spread_series(
    spread_panel: pd.DataFrame,
    spread: str,
    *,
    config: DailySeriesConfig | None = None,
) -> pd.DataFrame:
    """Build a daily series for a spread aligned to trade_date.

    Parameters
    ----------
    spread_panel:
        Stage 2 spreads panel (bucket-level).
    spread:
        Spread name (S1, S2, ...).
    config:
        DailySeriesConfig with aggregation and execution shift settings.

    Returns
    -------
    DataFrame with columns:
        trade_date, value, dte_bdays, near_contract, far_contract
    """
    cfg = config or DailySeriesConfig()
    spread = spread.upper()

    if "trade_date" not in spread_panel.columns or "bucket" not in spread_panel.columns:
        raise ValueError("spread_panel must include trade_date and bucket columns")
    if spread not in spread_panel.columns:
        raise ValueError(f"spread_panel missing {spread} column")
    dte_col = _dte_col_for_spread(spread, spread_panel)

    df = spread_panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()

    if cfg.source == "bucket1":
        # Use bucket 1 close as the daily series (tradeable opening bucket).
        df = df[df["bucket"] == 1].copy()
        near_col, far_col = _near_far_cols(spread)
        daily = df[[
            "trade_date",
            spread,
            dte_col,
            near_col,
            far_col,
        ]].copy()
        daily = daily.rename(
            columns={
                spread: "value",
                dte_col: "dte_bdays",
                near_col: "near_contract",
                far_col: "far_contract",
            }
        )
    elif cfg.source == "us_vwap":
        buckets = tuple(int(b) for b in cfg.buckets)
        df = df[df["bucket"].isin(buckets)].copy()
        if df.empty:
            raise ValueError("No rows after filtering to US session buckets")

        vcol1, vcol2 = _volume_cols_for_spread(spread)
        weights = np.zeros(len(df), dtype="float64")
        for col in (vcol1, vcol2):
            if col in df.columns:
                weights += pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        df["_w"] = weights
        if float(df["_w"].sum()) <= 0.0:
            df["_w"] = 1.0

        # Compute VWAP by trade_date
        vwap_num = (df[spread].astype("float64") * df["_w"]).groupby(df["trade_date"]).sum()
        vwap_den = df["_w"].groupby(df["trade_date"]).sum()
        vwap = vwap_num / vwap_den.replace(0.0, np.nan)

        # DTE and contract labels (should be constant per day)
        dte = df.groupby("trade_date")[dte_col].first()
        near_col, far_col = _near_far_cols(spread)
        near = df.groupby("trade_date")[near_col].first()
        far = df.groupby("trade_date")[far_col].first()

        daily = pd.DataFrame({
            "trade_date": vwap.index,
            "value": vwap.values,
            "dte_bdays": dte.reindex(vwap.index).values,
            "near_contract": near.reindex(vwap.index).values,
            "far_contract": far.reindex(vwap.index).values,
        }).reset_index(drop=True)
    else:
        raise ValueError("config.source must be 'us_vwap' or 'bucket1'")

    # Execution shift: align next-day tradeable price to prior day DTE
    if cfg.execution_shift_bdays:
        shift = int(cfg.execution_shift_bdays)
        if shift < 0:
            raise ValueError("execution_shift_bdays must be >= 0")
        daily = daily.sort_values("trade_date").reset_index(drop=True)
        daily["value"] = daily["value"].shift(-shift)
        daily = daily.dropna(subset=["value"]).reset_index(drop=True)

    return daily


def build_daily_spread_panel(
    spread_panel: pd.DataFrame,
    spreads: Iterable[str],
    *,
    config: DailySeriesConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build daily series for multiple spreads."""
    out: Dict[str, pd.DataFrame] = {}
    for spread in spreads:
        out[spread.upper()] = build_daily_spread_series(spread_panel, spread, config=config)
    return out


def align_by_dte(
    daily: pd.DataFrame,
    *,
    dte_col: str = "dte_bdays",
    group_col: str = "near_contract",
    dte_max: int | None = None,
) -> Dict[str, pd.Series]:
    """Align a daily series by DTE per contract cycle.

    Returns a mapping: contract -> Series indexed by DTE (business days).
    """
    if dte_col not in daily.columns or group_col not in daily.columns:
        raise ValueError("daily DataFrame missing DTE or group columns")

    out: Dict[str, pd.Series] = {}
    df = daily.dropna(subset=["value", dte_col]).copy()
    df[dte_col] = pd.to_numeric(df[dte_col], errors="coerce")
    df = df.dropna(subset=[dte_col])

    for contract, g in df.groupby(group_col, sort=False):
        g = g.sort_values(dte_col)
        if dte_max is not None:
            g = g[g[dte_col] <= dte_max]
        # Build series indexed by DTE
        s = pd.Series(g["value"].values, index=g[dte_col].astype(int).values)
        if not s.index.is_unique:
            s = s.groupby(level=0).mean()
        s = s.sort_index()
        out[str(contract)] = s

    return out


def plot_overlay_by_dte(
    series_map: Dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    output_png: Path,
    dte_max: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Determine observed DTE bounds after filtering (avoid empty axis range)
    obs_min = None
    obs_max = None
    for s in series_map.values():
        if s.empty:
            continue
        ss = s.sort_index()
        if dte_max is not None:
            ss = ss[ss.index <= dte_max]
        if ss.empty:
            continue
        smin = int(ss.index.min())
        smax = int(ss.index.max())
        obs_min = smin if obs_min is None else min(obs_min, smin)
        obs_max = smax if obs_max is None else max(obs_max, smax)

    if obs_min is None or obs_max is None:
        raise ValueError("No data available to plot overlay")

    full_idx = pd.Index(range(obs_min, obs_max + 1), dtype=int)

    plt.figure(figsize=(10, 6))
    for label, s in series_map.items():
        if s.empty:
            continue
        ss = s.sort_index()
        if dte_max is not None:
            ss = ss[ss.index <= dte_max]
        # Reindex to observed range to break lines on missing DTEs
        ss = ss.reindex(full_idx)
        plt.plot(ss.index.values, ss.values, linewidth=1.2, alpha=0.8)

    plt.title(title)
    plt.xlabel("Business Days to Expiry (DTE)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    if obs_min == obs_max:
        plt.xlim(obs_max + 1, obs_min - 1)
    else:
        plt.xlim(obs_max, obs_min)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def plot_average_by_dte(
    series_map: Dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    output_png: Path,
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series_map:
        raise ValueError("No series to plot")

    # Build wide frame on DTE index
    frames = []
    for label, s in series_map.items():
        if s.empty:
            continue
        ss = s.copy()
        ss.name = label
        frames.append(ss)
    if not frames:
        raise ValueError("All series are empty")
    wide = pd.concat(frames, axis=1).sort_index()
    if not wide.index.is_unique:
        wide = wide.groupby(level=0).mean()
    mean = wide.mean(axis=1)
    p25 = wide.quantile(0.25, axis=1)
    p75 = wide.quantile(0.75, axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = mean.index.values
    ax.plot(x, mean.values, color="#1f77b4", linewidth=2.0, label="Mean")
    ax.fill_between(x, p25.values, p75.values, color="#1f77b4", alpha=0.15, label="IQR (25-75%)")
    ax.set_title(title)
    ax.set_xlabel("Business Days to Expiry (DTE)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    out = pd.DataFrame({"mean": mean, "p25": p25, "p75": p75})
    out.index.name = "DTE"
    return out
