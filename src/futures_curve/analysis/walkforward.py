"""Walk-forward validation utilities for DTE window strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from futures_curve.utils.month_codes import parse_contract_code

from .strategy_scan import CostModel, _build_cycle_frames, _pnl_summary


@dataclass(frozen=True)
class Candidate:
    regime: str
    direction: str  # "long" or "short"
    entry_dte: int
    exit_dte: int


def _expiry_year(contract: str) -> Optional[int]:
    info = parse_contract_code(contract)
    return info.year_full if info is not None else None


def _candidate_pnls(
    cycle_frames: Dict[str, pd.DataFrame],
    cycles: Iterable[str],
    *,
    candidate: Candidate,
    cost_model: CostModel,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    gross_list: List[float] = []
    net_list: List[float] = []
    used: List[str] = []
    for key in cycles:
        frame = cycle_frames.get(key)
        if frame is None or frame.empty:
            continue
        if candidate.entry_dte not in frame.index or candidate.exit_dte not in frame.index:
            continue
        s2_entry = frame.loc[candidate.entry_dte, "s2"] if "s2" in frame.columns else np.nan
        if candidate.regime == "s2_pos":
            if pd.isna(s2_entry) or float(s2_entry) < 0:
                continue
        elif candidate.regime == "s2_neg":
            if pd.isna(s2_entry) or float(s2_entry) >= 0:
                continue
        gross = float(frame.loc[candidate.exit_dte, "s1"] - frame.loc[candidate.entry_dte, "s1"])
        if candidate.direction == "short":
            gross = -gross
        net = gross - cost_model.round_trip_spread_cost
        gross_list.append(gross)
        net_list.append(net)
        used.append(key)
    return np.asarray(gross_list, dtype=float), np.asarray(net_list, dtype=float), used


def _net_sharpe_like(net_pnls: np.ndarray) -> float:
    if net_pnls.size < 2:
        return float("nan")
    std = float(np.std(net_pnls, ddof=1))
    mean = float(np.mean(net_pnls))
    if std <= 0:
        return float("nan")
    return mean / std


def walkforward_validate(
    panel: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    entry_dtes: Iterable[int],
    exit_dtes: Iterable[int],
    regimes: Iterable[str] = ("all", "s2_pos", "s2_neg"),
    directions: Iterable[str] = ("long", "short"),
    min_train_trades: int = 50,
    enforce_positive_mean: bool = True,
    min_train_years: int = 5,
    cost_model: CostModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward selection by expiry year.

    Returns:
      - selected_by_year: chosen candidate per test year with train metrics
      - oos_trades: per-cycle trade PnLs for each test year
      - oos_summary: summary metrics per test year (and pooled row)
    """
    cycle_frames = _build_cycle_frames(panel)
    if not cycle_frames:
        raise ValueError("No cycle frames built from panel.")

    # Group cycles by expiry year using contract code.
    cycles_by_year: Dict[int, List[str]] = {}
    for contract in cycle_frames.keys():
        y = _expiry_year(contract)
        if y is None:
            continue
        cycles_by_year.setdefault(int(y), []).append(contract)

    entry_list = sorted(set(int(x) for x in entry_dtes), reverse=True)
    exit_list = sorted(set(int(y) for y in exit_dtes), reverse=True)
    regime_list = [r.strip().lower() for r in regimes if r.strip()]
    direction_list = [d.strip().lower() for d in directions if d.strip()]

    selected_rows: List[dict] = []
    oos_rows: List[dict] = []
    summary_rows: List[dict] = []

    first_test_year = start_year + int(min_train_years)
    test_years = [y for y in range(first_test_year, end_year + 1) if y in cycles_by_year]

    for test_year in test_years:
        train_cycles = []
        for y in range(start_year, test_year):
            train_cycles.extend(cycles_by_year.get(y, []))
        test_cycles = cycles_by_year.get(test_year, [])

        best_candidate: Optional[Candidate] = None
        best_sharpe = float("-inf")
        best_net_mean = float("-inf")
        best_n = 0
        best_train_stats: dict = {}

        for regime in regime_list:
            for direction in direction_list:
                for entry in entry_list:
                    for exit_dte in exit_list:
                        if exit_dte >= entry:
                            continue
                        cand = Candidate(regime=regime, direction=direction, entry_dte=entry, exit_dte=exit_dte)
                        _, net, _ = _candidate_pnls(cycle_frames, train_cycles, candidate=cand, cost_model=cost_model)
                        if net.size < min_train_trades:
                            continue
                        net_mean = float(np.mean(net))
                        if enforce_positive_mean and not (net_mean > 0):
                            continue
                        net_std = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
                        if net_std <= 0:
                            continue
                        sharpe_like = net_mean / net_std
                        if (
                            sharpe_like > best_sharpe
                            or (np.isclose(sharpe_like, best_sharpe) and net_mean > best_net_mean)
                            or (np.isclose(sharpe_like, best_sharpe) and np.isclose(net_mean, best_net_mean) and net.size > best_n)
                        ):
                            best_candidate = cand
                            best_sharpe = sharpe_like
                            best_net_mean = net_mean
                            best_n = int(net.size)
                            best_train_stats = {
                                "train_net_mean": net_mean,
                                "train_net_std": net_std,
                                "train_net_sharpe_like": sharpe_like,
                                "train_n": int(net.size),
                            }

        if best_candidate is None:
            selected_rows.append(
                {
                    "test_year": test_year,
                    "selected": False,
                    "reason": "No candidate met constraints",
                    "min_train_trades": min_train_trades,
                    "enforce_positive_mean": enforce_positive_mean,
                }
            )
            continue

        # Apply selected candidate to test year
        gross, net, used_cycles = _candidate_pnls(cycle_frames, test_cycles, candidate=best_candidate, cost_model=cost_model)

        # Record per-cycle OOS trades
        for c, g, n in zip(used_cycles, gross.tolist(), net.tolist()):
            oos_rows.append(
                {
                    "test_year": test_year,
                    "cycle": c,
                    "regime": best_candidate.regime,
                    "direction": best_candidate.direction,
                    "entry_dte": best_candidate.entry_dte,
                    "exit_dte": best_candidate.exit_dte,
                    "hold_days": best_candidate.entry_dte - best_candidate.exit_dte,
                    "gross_pnl": g,
                    "net_pnl": n,
                    "gross_pnl_usd": g * cost_model.contract_size,
                    "net_pnl_usd": n * cost_model.contract_size,
                }
            )

        # Year summary
        net_stats = _pnl_summary(net)
        sharpe_like_oos = _net_sharpe_like(net)
        summary_rows.append(
            {
                "test_year": test_year,
                "regime": best_candidate.regime,
                "direction": best_candidate.direction,
                "entry_dte": best_candidate.entry_dte,
                "exit_dte": best_candidate.exit_dte,
                "hold_days": best_candidate.entry_dte - best_candidate.exit_dte,
                "n": net_stats["n"],
                "net_mean": net_stats["mean"],
                "net_std": net_stats["std"],
                "net_t": net_stats["t_stat"],
                "net_win_rate": net_stats["win_rate"],
                "net_sharpe_like": sharpe_like_oos,
                "net_mean_usd": net_stats["mean"] * cost_model.contract_size if not np.isnan(net_stats["mean"]) else np.nan,
                **best_train_stats,
            }
        )

        selected_rows.append(
            {
                "test_year": test_year,
                "selected": True,
                "regime": best_candidate.regime,
                "direction": best_candidate.direction,
                "entry_dte": best_candidate.entry_dte,
                "exit_dte": best_candidate.exit_dte,
                "hold_days": best_candidate.entry_dte - best_candidate.exit_dte,
                **best_train_stats,
            }
        )

    selected_by_year = pd.DataFrame(selected_rows)
    oos_trades = pd.DataFrame(oos_rows)
    oos_summary = pd.DataFrame(summary_rows)

    # Pooled OOS row
    if not oos_trades.empty:
        pooled = oos_trades["net_pnl"].to_numpy(dtype=float)
        pooled_stats = _pnl_summary(pooled)
        pooled_row = {
            "test_year": "pooled",
            "n": pooled_stats["n"],
            "net_mean": pooled_stats["mean"],
            "net_std": pooled_stats["std"],
            "net_t": pooled_stats["t_stat"],
            "net_win_rate": pooled_stats["win_rate"],
            "net_sharpe_like": _net_sharpe_like(pooled),
            "net_mean_usd": pooled_stats["mean"] * cost_model.contract_size if not np.isnan(pooled_stats["mean"]) else np.nan,
        }
        oos_summary = pd.concat([oos_summary, pd.DataFrame([pooled_row])], ignore_index=True)

    return selected_by_year, oos_trades, oos_summary
