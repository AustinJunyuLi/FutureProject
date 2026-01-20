"""Robustness utilities for fixed-rule DTE window strategies.

Key design choice:
- Keep *signals* fixed (baseline S2 regime at entry), while allowing *execution*
  assumptions (price source, execution shift, costs) to vary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from futures_curve.utils.month_codes import parse_contract_code

from .event_study import EntryRegimeSpec, entry_day_s2_sign_by_cycle
from .strategy_scan import CostModel, _build_cycle_frames, _pnl_summary, scan_windows


@dataclass(frozen=True)
class FixedRule:
    regime: str
    direction: str  # "long" or "short"
    entry_dte: int
    exit_dte: int


def _expiry_year(contract: str) -> Optional[int]:
    info = parse_contract_code(contract)
    return info.year_full if info is not None else None


def rank_windows(
    window_summary: pd.DataFrame,
    *,
    metric: str = "net_sharpe_like",
    min_trades: int = 50,
    enforce_positive_mean: bool = True,
) -> pd.DataFrame:
    """Rank scanned windows with guardrails and a chosen metric."""
    df = window_summary.copy()
    if df.empty:
        return df
    for col in ("net_mean", "net_std", "n"):
        if col not in df.columns:
            raise ValueError(f"window_summary missing {col}")

    df["net_sharpe_like"] = df["net_mean"] / df["net_std"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["net_sharpe_like"])
    df = df[df["n"] >= int(min_trades)]
    if enforce_positive_mean:
        df = df[df["net_mean"] > 0]

    if df.empty:
        return df

    if metric not in df.columns:
        raise ValueError(f"Unknown rank metric: {metric}")

    return df.sort_values([metric, "net_mean", "n"], ascending=[False, False, False]).reset_index(drop=True)


def select_best_rule(
    panel: pd.DataFrame,
    *,
    entry_dtes: Iterable[int],
    exit_dtes: Iterable[int],
    cost_model: CostModel,
    regimes: Iterable[str] = ("all", "s2_pos", "s2_neg"),
    directions: Iterable[str] = ("long", "short"),
    min_trades: int = 50,
    enforce_positive_mean: bool = True,
    rank_metric: str = "net_sharpe_like",
) -> tuple[FixedRule, pd.DataFrame]:
    """Scan and select the best fixed rule on the provided panel."""
    summary = scan_windows(panel, entry_dtes=entry_dtes, exit_dtes=exit_dtes, cost_model=cost_model, regimes=regimes)
    summary = summary[summary["direction"].isin([d.strip().lower() for d in directions])].copy()
    ranked = rank_windows(summary, metric=rank_metric, min_trades=min_trades, enforce_positive_mean=enforce_positive_mean)
    if ranked.empty:
        raise ValueError("No candidate met constraints for selection.")
    top = ranked.iloc[0]
    rule = FixedRule(
        regime=str(top["regime"]),
        direction=str(top["direction"]),
        entry_dte=int(top["entry_dte"]),
        exit_dte=int(top["exit_dte"]),
    )
    return rule, ranked


def evaluate_fixed_rule(
    execution_panel: pd.DataFrame,
    *,
    rule: FixedRule,
    signal_panel: pd.DataFrame,
    cost_model: CostModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a fixed rule using *execution* prices and *signal* regime membership."""
    cycle_frames = _build_cycle_frames(execution_panel)
    if not cycle_frames:
        raise ValueError("No cycle frames built from execution_panel.")

    # Signal membership at entry (fixed across scenarios)
    entry_sign = entry_day_s2_sign_by_cycle(signal_panel, spec=EntryRegimeSpec(entry_dte=int(rule.entry_dte)))

    cycles = list(cycle_frames.keys())
    if rule.regime.strip().lower() == "s2_pos":
        cycles = [c for c in cycles if c in entry_sign.index and bool(entry_sign.loc[c])]
    elif rule.regime.strip().lower() == "s2_neg":
        cycles = [c for c in cycles if c in entry_sign.index and not bool(entry_sign.loc[c])]

    rows = []
    for c in cycles:
        frame = cycle_frames.get(c)
        if frame is None or frame.empty:
            continue
        if int(rule.entry_dte) not in frame.index or int(rule.exit_dte) not in frame.index:
            continue
        gross = float(frame.loc[int(rule.exit_dte), "s1"] - frame.loc[int(rule.entry_dte), "s1"])
        if rule.direction.strip().lower() == "short":
            gross = -gross
        net = gross - cost_model.round_trip_spread_cost
        rows.append(
            {
                "cycle": c,
                "expiry_year": _expiry_year(c),
                "regime": rule.regime,
                "direction": rule.direction,
                "entry_dte": int(rule.entry_dte),
                "exit_dte": int(rule.exit_dte),
                "hold_days": int(rule.entry_dte) - int(rule.exit_dte),
                "gross_pnl": gross,
                "net_pnl": net,
                "gross_pnl_usd": gross * cost_model.contract_size,
                "net_pnl_usd": net * cost_model.contract_size,
            }
        )

    trades = pd.DataFrame(rows)
    if trades.empty:
        summary = pd.DataFrame(
            [
                {
                    "expiry_year": "pooled",
                    "n": 0,
                    "gross_mean": np.nan,
                    "net_mean": np.nan,
                    "net_t": np.nan,
                    "net_win_rate": np.nan,
                }
            ]
        )
        return trades, summary

    def _summ(net: np.ndarray, gross: np.ndarray) -> dict:
        gross_stats = _pnl_summary(gross)
        net_stats = _pnl_summary(net)
        sharpe_like = float(net_stats["mean"] / net_stats["std"]) if net_stats["std"] and net_stats["std"] > 0 else np.nan
        return {
            "n": net_stats["n"],
            "gross_mean": gross_stats["mean"],
            "gross_std": gross_stats["std"],
            "gross_t": gross_stats["t_stat"],
            "gross_win_rate": gross_stats["win_rate"],
            "net_mean": net_stats["mean"],
            "net_std": net_stats["std"],
            "net_t": net_stats["t_stat"],
            "net_win_rate": net_stats["win_rate"],
            "net_sharpe_like": sharpe_like,
            "net_mean_usd": net_stats["mean"] * cost_model.contract_size if not np.isnan(net_stats["mean"]) else np.nan,
        }

    # By-year summary + pooled row
    summary_rows = []
    for y, g in trades.groupby("expiry_year", dropna=False):
        gross = g["gross_pnl"].to_numpy(dtype=float)
        net = g["net_pnl"].to_numpy(dtype=float)
        row = {"expiry_year": int(y) if y is not None and not pd.isna(y) else "unknown"}
        row.update(_summ(net, gross))
        summary_rows.append(row)

    pooled_gross = trades["gross_pnl"].to_numpy(dtype=float)
    pooled_net = trades["net_pnl"].to_numpy(dtype=float)
    pooled = {"expiry_year": "pooled"}
    pooled.update(_summ(pooled_net, pooled_gross))
    summary_rows.append(pooled)

    summary = pd.DataFrame(summary_rows)
    return trades, summary

