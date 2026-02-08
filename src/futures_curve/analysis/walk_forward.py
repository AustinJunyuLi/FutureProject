from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .backtest import run_backtest_single_series
from .config import AnalysisConfig
from .costs import CostModel
from .data import SpreadDailySeries
from .metrics import compute_drawdown_series, compute_sharpe_annualized
from .params import FilterConfig
from .risk import ewma_volatility
from .roll import RollFilterConfig, compute_roll_tradable_mask
from .strategies.base import BaseStrategy


@dataclass(frozen=True)
class FoldResult:
    fold_year: int
    strategy: str
    spread: str
    params: dict[str, object]
    filters: dict[str, object]

    train_sharpe: float
    train_max_dd: float
    train_nonzero_days: int

    test_sharpe: float
    test_max_dd: float
    test_mean_daily: float
    test_turnover: float
    test_n_days: int
    test_nonzero_days: int


@dataclass(frozen=True)
class WalkForwardOutput:
    folds: list[FoldResult]
    oos_pnl: pd.Series
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_sharpe: float
    oos_max_dd: float
    oos_nonzero_days: int
    oos_years: list[int]


def _year_window(df_index: pd.DatetimeIndex, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date(year, 1, 1))
    end = pd.Timestamp(date(year, 12, 31))
    start = max(start, df_index.min())
    end = min(end, df_index.max())
    return start, end


def build_cost_model(series: SpreadDailySeries, config: AnalysisConfig) -> CostModel:
    """Build the standard cost model for a spread series."""
    return CostModel(
        dollars_per_tick=series.dollars_per_tick,
        legs=2,
        ticks_per_leg_per_side=config.ticks_per_leg_per_side,
    )


def _compute_vol_series(
    *,
    price_exec: pd.Series,
    contract_size: float,
    config: AnalysisConfig,
) -> pd.Series:
    delta_price = price_exec.diff()
    pnl_1 = (delta_price * contract_size).fillna(0.0)
    return ewma_volatility(
        pnl_1,
        halflife_days=config.ewma_halflife_days,
        min_periods=config.ewma_min_periods,
    )


def _contango_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    mode = mode.lower()
    if mode not in {"all", "contango_only", "backwardation_only"}:
        raise ValueError(f"Unknown contango mode: {mode}")
    if mode == "all":
        return pd.Series(True, index=df.index)
    s = df["s_signal_pct"].astype("float64")
    if mode == "contango_only":
        return (s > 0).fillna(False)
    return (s < 0).fillna(False)


def _vol_mask(vol: pd.Series, mode: str, threshold: float) -> pd.Series:
    mode = mode.lower()
    if mode not in {"all", "high_vol_only", "low_vol_only"}:
        raise ValueError(f"Unknown vol mode: {mode}")
    if mode == "all":
        return pd.Series(True, index=vol.index)
    if mode == "high_vol_only":
        return (vol >= threshold).fillna(False)
    return (vol < threshold).fillna(False)


def _combined_trade_mask(
    *,
    df: pd.DataFrame,
    roll_daily: pd.DataFrame | None,
    roll_filter_mode: str,
    roll_cfg: RollFilterConfig,
    contango_mode: str,
    vol_mode: str,
    vol_series: pd.Series,
    vol_threshold: float,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if roll_daily is not None:
        roll = compute_roll_tradable_mask(roll_daily, mode=roll_filter_mode, cfg=roll_cfg).reindex(df.index)
        mask &= roll.fillna(False)
    mask &= _contango_mask(df, contango_mode)
    mask &= _vol_mask(vol_series.reindex(df.index), vol_mode, vol_threshold)
    return mask.fillna(False)


def _turnover_holding_gate(turnover_daily: float) -> bool:
    if not np.isfinite(turnover_daily):
        return False
    return (turnover_daily >= 1.0 / 20.0) and (turnover_daily <= 1.0)


def _summarize_oos(config: AnalysisConfig, pnl_net: pd.Series) -> tuple[pd.Series, pd.Series, float, float, int]:
    pnl_net = pnl_net.sort_index().fillna(0.0)
    equity = pnl_net.cumsum().add(config.initial_capital)
    returns = (pnl_net / equity.shift(1).replace(0.0, np.nan)).fillna(0.0)
    sharpe = compute_sharpe_annualized(returns, trading_days=config.trading_days_per_year)
    max_dd = float(compute_drawdown_series(equity).max()) if len(equity) else float("nan")
    nonzero_days = int((pnl_net != 0).sum())
    return returns, equity, float(sharpe), float(max_dd), nonzero_days


def _slice_series(series: SpreadDailySeries, df_subset: pd.DataFrame) -> SpreadDailySeries:
    """Create a SpreadDailySeries with a subset of the original DataFrame."""
    return SpreadDailySeries(
        spread=series.spread,
        df=df_subset,
        contract_size=series.contract_size,
        dollars_per_tick=series.dollars_per_tick,
    )


def regime_pairs(regimes: list[str]) -> list[tuple[str, str]]:
    """Translate a single list into (contango_mode, vol_mode) pairs.

    Avoids a full cross-product while covering the key regime filters.
    """
    pairs: list[tuple[str, str]] = [("all", "all")]
    for r in regimes:
        r = r.strip()
        if not r or r == "all":
            continue
        if r in {"contango_only", "backwardation_only"}:
            pairs.append((r, "all"))
        elif r in {"high_vol_only", "low_vol_only"}:
            pairs.append(("all", r))
        else:
            raise ValueError(f"Unknown regime: {r}")
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def walk_forward(
    *,
    config: AnalysisConfig,
    series: SpreadDailySeries,
    roll_daily: pd.DataFrame | None,
    roll_filter_mode: str,
    roll_cfg: RollFilterConfig,
    contango_vol_filter_pairs: list[tuple[str, str]],
    first_test_year: int,
    last_test_year: int,
    strategy_grids: list[tuple[BaseStrategy, list[object]]],
    min_train_days: int = 252 * 3,
) -> WalkForwardOutput:
    """Yearly expanding-window walk-forward for any set of strategies.

    Args:
        strategy_grids: List of (strategy, param_grid) tuples. All are searched
            jointly per fold; the best (strategy, param, direction, filter) wins.
    """
    df = series.df.copy().sort_index()
    cm = build_cost_model(series, config)

    folds: list[FoldResult] = []
    oos_pnl_pieces: list[pd.Series] = []
    oos_years: list[int] = []

    for year in range(first_test_year, last_test_year + 1):
        train_end = pd.Timestamp(date(year - 1, 12, 31))
        test_start, test_end = _year_window(df.index, year)

        train_df = df.loc[df.index <= train_end]
        if len(train_df) < min_train_days:
            continue

        vol_full = _compute_vol_series(
            price_exec=df.loc[df.index <= test_end, "s_exec"],
            contract_size=series.contract_size,
            config=config,
        )
        vol_threshold = float(vol_full.loc[train_df.index].median())

        train_series = _slice_series(series, train_df)

        # Precompute base (unfiltered) positions for each (strategy, params).
        base_positions: list[tuple[BaseStrategy, object, pd.Series]] = []
        for strategy, param_grid in strategy_grids:
            for params in param_grid:
                sig = strategy.generate_signal(
                    train_series, params,
                    trading_days_per_year=config.trading_days_per_year,
                )
                pos = strategy.positions_from_signal(sig, params).astype("float64")
                base_positions.append((strategy, params, pos))

        best: float | None = None
        best_meta: dict[str, object] | None = None

        for contango_mode, vol_mode in contango_vol_filter_pairs:
            mask_train = _combined_trade_mask(
                df=train_df,
                roll_daily=roll_daily,
                roll_filter_mode=roll_filter_mode,
                roll_cfg=roll_cfg,
                contango_mode=contango_mode,
                vol_mode=vol_mode,
                vol_series=vol_full,
                vol_threshold=vol_threshold,
            )

            for strategy, params, base_pos in base_positions:
                pos = base_pos.where(mask_train, 0.0)
                for direction in (1, -1):
                    signal = (direction * pos).astype("float64")
                    bt = run_backtest_single_series(
                        config=config,
                        price_exec=train_df["s_exec"],
                        signal=signal,
                        contract_size=series.contract_size,
                        cost_model=cm,
                    )
                    nonzero_days = int((bt.position.abs() > 1e-12).sum())
                    if nonzero_days < config.min_train_nonzero_days:
                        continue
                    if not _turnover_holding_gate(bt.turnover_daily):
                        continue
                    if bt.max_drawdown > config.max_drawdown_limit:
                        continue
                    if bt.returns.mean() <= 0:
                        continue

                    score = float(bt.sharpe_annual)
                    if best is None or (np.isfinite(score) and score > best):
                        best = score
                        best_meta = {
                            "strategy": strategy,
                            "params": params,
                            "direction": int(direction),
                            "contango_mode": contango_mode,
                            "vol_mode": vol_mode,
                            "vol_threshold": vol_threshold,
                            "train_max_dd": float(bt.max_drawdown),
                            "train_nonzero_days": nonzero_days,
                        }

        if best_meta is None or best is None:
            continue

        # Test: deploy selected params for the test year using full history up to test_end.
        df_upto = df.loc[df.index <= test_end].copy()
        vol_upto = _compute_vol_series(
            price_exec=df_upto["s_exec"],
            contract_size=series.contract_size,
            config=config,
        )
        mask_upto = _combined_trade_mask(
            df=df_upto,
            roll_daily=roll_daily,
            roll_filter_mode=roll_filter_mode,
            roll_cfg=roll_cfg,
            contango_mode=str(best_meta["contango_mode"]),
            vol_mode=str(best_meta["vol_mode"]),
            vol_series=vol_upto,
            vol_threshold=float(best_meta["vol_threshold"]),
        )

        best_strategy: BaseStrategy = best_meta["strategy"]
        best_params = best_meta["params"]
        upto_series = _slice_series(series, df_upto)

        sig = best_strategy.generate_signal(
            upto_series, best_params,
            trading_days_per_year=config.trading_days_per_year,
        )
        pos = best_strategy.positions_from_signal(sig, best_params)
        pos = pos.where(mask_upto, 0.0).astype("float64") * int(best_meta["direction"])
        pos.loc[pos.index < test_start] = 0.0

        bt = run_backtest_single_series(
            config=config,
            price_exec=df_upto["s_exec"],
            signal=pos,
            contract_size=series.contract_size,
            cost_model=cm,
        )

        win = bt.returns.loc[(bt.returns.index >= test_start) & (bt.returns.index <= test_end)]
        pnl_win = bt.pnl_net.loc[win.index]
        nonzero_days_test = int((bt.position.loc[win.index].abs() > 1e-12).sum())
        if nonzero_days_test < config.min_test_nonzero_days:
            continue

        equity_win = bt.equity.loc[win.index]
        test_sharpe = compute_sharpe_annualized(win, trading_days=config.trading_days_per_year)
        test_max_dd = float(compute_drawdown_series(equity_win).max()) if len(equity_win) else float("nan")

        folds.append(
            FoldResult(
                fold_year=year,
                strategy=best_strategy.name,
                spread=series.spread,
                params=best_strategy.fold_params_dict(best_params, int(best_meta["direction"])),
                filters={
                    "roll_filter": roll_filter_mode,
                    "contango_mode": best_meta["contango_mode"],
                    "vol_mode": best_meta["vol_mode"],
                },
                train_sharpe=float(best),
                train_max_dd=float(best_meta["train_max_dd"]),
                train_nonzero_days=int(best_meta["train_nonzero_days"]),
                test_sharpe=float(test_sharpe),
                test_max_dd=float(test_max_dd),
                test_mean_daily=float(win.mean()),
                test_turnover=float(bt.turnover_daily),
                test_n_days=int(len(win)),
                test_nonzero_days=nonzero_days_test,
            )
        )

        oos_pnl_pieces.append(pnl_win)
        oos_years.append(year)

    if oos_pnl_pieces:
        pnl_all = pd.concat(oos_pnl_pieces).sort_index()
    else:
        pnl_all = pd.Series(dtype="float64")

    oos_returns, oos_equity, oos_sharpe, oos_max_dd, oos_nonzero_days = _summarize_oos(config, pnl_all)

    return WalkForwardOutput(
        folds=folds,
        oos_pnl=pnl_all,
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        oos_sharpe=oos_sharpe,
        oos_max_dd=oos_max_dd,
        oos_nonzero_days=oos_nonzero_days,
        oos_years=oos_years,
    )


# ---------------------------------------------------------------------------
# Legacy wrappers — preserve the old API for backward compatibility.
# These delegate to the unified walk_forward() with the appropriate strategies.
# ---------------------------------------------------------------------------

def walk_forward_carry(
    *,
    config: AnalysisConfig,
    series: SpreadDailySeries,
    roll_daily: pd.DataFrame | None,
    roll_filter_mode: str,
    roll_cfg: RollFilterConfig,
    contango_vol_filter_pairs: list[tuple[str, str]],
    first_test_year: int,
    last_test_year: int,
    thresholds: list[float],
    scales: list[float],
    min_train_days: int = 252 * 3,
) -> WalkForwardOutput:
    """Walk-forward OOS evaluation for carry/roll-down family (legacy wrapper)."""
    from .params import CarryClippedParams, CarrySignParams
    from .strategies.carry import CarryClippedStrategy, CarrySignStrategy

    sign_grid = [CarrySignParams(threshold=t) for t in thresholds]
    clipped_grid = [
        CarryClippedParams(threshold=t, scale=s)
        for t in thresholds
        for s in scales
        if s > 0
    ]

    return walk_forward(
        config=config,
        series=series,
        roll_daily=roll_daily,
        roll_filter_mode=roll_filter_mode,
        roll_cfg=roll_cfg,
        contango_vol_filter_pairs=contango_vol_filter_pairs,
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        strategy_grids=[
            (CarrySignStrategy(), sign_grid),
            (CarryClippedStrategy(), clipped_grid),
        ],
        min_train_days=min_train_days,
    )


def walk_forward_mean_reversion(
    *,
    config: AnalysisConfig,
    series: SpreadDailySeries,
    roll_daily: pd.DataFrame | None,
    roll_filter_mode: str,
    roll_cfg: RollFilterConfig,
    contango_vol_filter_pairs: list[tuple[str, str]],
    first_test_year: int,
    last_test_year: int,
    param_grid: list,
    min_train_days: int = 252 * 3,
) -> WalkForwardOutput:
    """Walk-forward OOS evaluation for mean reversion (legacy wrapper)."""
    from .strategies.mean_reversion import MeanReversionStrategy

    return walk_forward(
        config=config,
        series=series,
        roll_daily=roll_daily,
        roll_filter_mode=roll_filter_mode,
        roll_cfg=roll_cfg,
        contango_vol_filter_pairs=contango_vol_filter_pairs,
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        strategy_grids=[
            (MeanReversionStrategy(), param_grid),
        ],
        min_train_days=min_train_days,
    )
