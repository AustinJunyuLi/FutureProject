"""Strategy discovery utilities for DTE-anchored pre-expiry analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd

from .expiry_seasonality import DailySeriesConfig, build_daily_spread_series, align_by_dte, plot_average_by_dte


HG_TICK_SIZE_USD_PER_LB = 0.0005  # COMEX HG tick size (USD/lb). Source: CME Group.
# Reference: https://www.cmegroup.com/education/articles-and-reports/hedging-with-comex-copper-futures.html
HG_CONTRACT_SIZE_LB = 25_000


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost model in spread price units."""

    tick_size: float = HG_TICK_SIZE_USD_PER_LB
    ticks_per_leg_side: float = 1.0
    legs: int = 2  # F1 + F2
    sides: int = 2  # entry + exit
    contract_size: int = HG_CONTRACT_SIZE_LB

    @property
    def round_trip_spread_cost(self) -> float:
        return float(self.tick_size * self.ticks_per_leg_side * self.legs * self.sides)

    @property
    def round_trip_dollar_cost(self) -> float:
        return float(self.round_trip_spread_cost * self.contract_size)


def _filter_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31 23:59:59")
    return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()


def build_strategy_panel(
    spread_panel: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    config: DailySeriesConfig | None = None,
    s1_config: DailySeriesConfig | None = None,
    s2_config: DailySeriesConfig | None = None,
) -> pd.DataFrame:
    """Build daily S1 panel with optional S2 columns for regime diagnostics."""
    if config is not None and (s1_config is not None or s2_config is not None):
        raise ValueError("Pass either config=... or (s1_config=..., s2_config=...), not both.")

    s1_cfg = config or s1_config or DailySeriesConfig()
    s2_cfg = config or s2_config or s1_cfg

    # S1 daily series (near leg = F1)
    s1_daily = build_daily_spread_series(spread_panel, "S1", config=s1_cfg)
    s1_daily = _filter_year_range(s1_daily, start_year, end_year)
    s1_daily = s1_daily.rename(columns={"value": "s1"})

    # Compute S1 daily change within each near-contract cycle
    s1_daily = s1_daily.sort_values(["near_contract", "trade_date"]).copy()
    s1_daily["ds1"] = s1_daily.groupby("near_contract", sort=False)["s1"].diff()

    # S2 daily series (near leg = F2)
    s2_daily = build_daily_spread_series(spread_panel, "S2", config=s2_cfg)
    s2_daily = _filter_year_range(s2_daily, start_year, end_year)
    s2_daily = s2_daily.rename(
        columns={
            "value": "s2",
            "dte_bdays": "s2_dte_bdays",
            "near_contract": "s2_contract",
        }
    )
    s2_daily = s2_daily.sort_values(["s2_contract", "trade_date"]).copy()
    s2_daily["ds2"] = s2_daily.groupby("s2_contract", sort=False)["s2"].diff()

    # Merge S2 onto S1 panel by trade_date (same curve date)
    panel = s1_daily.merge(
        s2_daily[["trade_date", "s2", "ds2", "s2_dte_bdays", "s2_contract"]],
        on="trade_date",
        how="left",
    )

    panel["dte_bdays"] = pd.to_numeric(panel["dte_bdays"], errors="coerce")
    return panel


def compute_dte_drift_stats(
    panel: pd.DataFrame,
    *,
    dte_max: int,
    output_png: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Compute and plot mean ΔS1 by DTE using IQR bands."""
    df = panel.dropna(subset=["ds1", "dte_bdays", "near_contract"]).copy()
    df = df.rename(columns={"ds1": "value"})
    series_map = align_by_dte(df, dte_max=dte_max)
    # Drop empty series
    cleaned: Dict[str, pd.Series] = {}
    for key, s in series_map.items():
        s = s.dropna()
        if not s.empty:
            cleaned[key] = s
    if not cleaned:
        raise ValueError("No ΔS1 data available for drift stats.")

    avg_df = plot_average_by_dte(
        cleaned,
        title=f"ΔS1 by DTE — {dte_max}-day window (US VWAP, shift=0)",
        ylabel="ΔS1 (daily change)",
        output_png=output_png,
    )
    avg_df.to_csv(output_csv)
    return avg_df


def _pnl_summary(pnls: np.ndarray) -> dict:
    n = len(pnls)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "win_rate": np.nan,
        }
    mean = float(np.mean(pnls))
    median = float(np.median(pnls))
    std = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
    t_stat = float(mean / (std / np.sqrt(n))) if std > 0 and n > 1 else np.nan
    win_rate = float((pnls > 0).mean())
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "t_stat": t_stat,
        "win_rate": win_rate,
    }


def _build_cycle_frames(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = panel.dropna(subset=["s1", "dte_bdays", "near_contract"]).copy()
    df["dte_bdays"] = pd.to_numeric(df["dte_bdays"], errors="coerce")
    df = df.dropna(subset=["dte_bdays"])
    df["dte_bdays"] = df["dte_bdays"].astype(int)

    frames: Dict[str, pd.DataFrame] = {}
    for contract, g in df.groupby("near_contract", sort=False):
        g = g.sort_values("dte_bdays")
        # Average duplicates on same DTE
        agg = g.groupby("dte_bdays")[["s1", "s2"]].mean()
        if not agg.empty:
            frames[str(contract)] = agg
    return frames


def scan_windows(
    panel: pd.DataFrame,
    *,
    entry_dtes: Iterable[int],
    exit_dtes: Iterable[int],
    cost_model: CostModel,
    regimes: Iterable[str] = ("all", "s2_pos", "s2_neg"),
) -> pd.DataFrame:
    """Scan entry/exit DTE windows for long/short S1."""
    cycle_frames = _build_cycle_frames(panel)

    rows = []
    entry_list = sorted(set(int(x) for x in entry_dtes), reverse=True)
    exit_list = sorted(set(int(y) for y in exit_dtes), reverse=True)
    regime_list = [r.strip().lower() for r in regimes if r.strip()]

    for regime in regime_list:
        for direction in ("long", "short"):
            for entry in entry_list:
                for exit in exit_list:
                    if exit >= entry:
                        continue
                    pnls: List[float] = []
                    for _, frame in cycle_frames.items():
                        if entry not in frame.index or exit not in frame.index:
                            continue
                        s2_entry = frame.loc[entry, "s2"] if "s2" in frame.columns else np.nan
                        if regime == "s2_pos":
                            if pd.isna(s2_entry) or float(s2_entry) < 0:
                                continue
                        elif regime == "s2_neg":
                            if pd.isna(s2_entry) or float(s2_entry) >= 0:
                                continue
                        gross = float(frame.loc[exit, "s1"] - frame.loc[entry, "s1"])
                        if direction == "short":
                            gross = -gross
                        pnls.append(gross)
                    pnls = np.array(pnls, dtype=float)
                    gross_stats = _pnl_summary(pnls)
                    net_pnls = pnls - cost_model.round_trip_spread_cost
                    net_stats = _pnl_summary(net_pnls)

                    rows.append(
                        {
                            "regime": regime,
                            "direction": direction,
                            "entry_dte": entry,
                            "exit_dte": exit,
                            "hold_days": entry - exit,
                            "n": gross_stats["n"],
                            "gross_mean": gross_stats["mean"],
                            "gross_median": gross_stats["median"],
                            "gross_std": gross_stats["std"],
                            "gross_t": gross_stats["t_stat"],
                            "gross_win_rate": gross_stats["win_rate"],
                            "net_mean": net_stats["mean"],
                            "net_median": net_stats["median"],
                            "net_std": net_stats["std"],
                            "net_t": net_stats["t_stat"],
                            "net_win_rate": net_stats["win_rate"],
                            "cost_spread": cost_model.round_trip_spread_cost,
                            "cost_usd": cost_model.round_trip_dollar_cost,
                            "gross_mean_usd": float(gross_stats["mean"] * cost_model.contract_size) if not np.isnan(gross_stats["mean"]) else np.nan,
                            "net_mean_usd": float(net_stats["mean"] * cost_model.contract_size) if not np.isnan(net_stats["mean"]) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)
