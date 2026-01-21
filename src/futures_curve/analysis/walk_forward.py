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
from .risk import ewma_volatility
from .roll import RollFilterConfig, compute_roll_tradable_mask
from .strategies.mean_reversion import MeanReversionParams, compute_dte_conditioned_zscore, positions_from_zscore


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


def _cost_model(series: SpreadDailySeries, config: AnalysisConfig) -> CostModel:
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
    # Approx holding period proxy ~ 1 / turnover_daily
    # Enforce 1–20 business days => turnover in [0.05, 1.0]
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
    # Carry search space
    thresholds: list[float],
    scales: list[float],
    min_train_days: int = 252 * 3,
) -> WalkForwardOutput:
    """Walk-forward OOS evaluation for carry/roll-down family.

    Two candidate signal types:
    - sign:    signal = -sign(carry) * 1[|carry|>=thr]
    - clipped: signal = clip(-carry/scale, -1, 1) * 1[|carry|>=thr]
    """
    df = series.df.copy().sort_index()
    cm = _cost_model(series, config)

    folds: list[FoldResult] = []
    oos_pnl_pieces: list[pd.Series] = []
    oos_years: list[int] = []

    for year in range(first_test_year, last_test_year + 1):
        train_end = pd.Timestamp(date(year - 1, 12, 31))
        test_start, test_end = _year_window(df.index, year)

        train_df = df.loc[df.index <= train_end]
        if len(train_df) < min_train_days:
            continue

        # Vol regime threshold computed from training window.
        vol_full = _compute_vol_series(price_exec=df.loc[df.index <= test_end, "s_exec"], contract_size=series.contract_size, config=config)
        vol_threshold = float(vol_full.loc[train_df.index].median())

        # Carry proxy on train (annualized spread pct / spacing years).
        s_pct_train = train_df["s_signal_pct"].astype("float64")
        spacing_train = (train_df["far_dte_bdays"] - train_df["near_dte_bdays"]).astype("float64")
        spacing_years_train = spacing_train / float(config.trading_days_per_year)
        carry_train = (s_pct_train / spacing_years_train.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

        best = None
        best_meta = None

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

            # Candidate 1: sign-based (use smallest threshold list entry as the only sign candidate if provided)
            for thr in thresholds:
                sig = -np.sign(carry_train).where(carry_train.abs() >= float(thr), 0.0).fillna(0.0)
                sig = sig.where(mask_train, 0.0).astype("float64")

                for direction in (1, -1):
                    signal = (direction * sig).astype("float64")
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
                            "type": "sign",
                            "threshold": float(thr),
                            "scale": None,
                            "direction": int(direction),
                            "contango_mode": contango_mode,
                            "vol_mode": vol_mode,
                            "vol_threshold": vol_threshold,
                            "train_max_dd": float(bt.max_drawdown),
                            "train_nonzero_days": nonzero_days,
                        }

            # Candidate 2: clipped continuous
            for thr in thresholds:
                for scale in scales:
                    if scale <= 0:
                        continue
                    raw = -(carry_train / float(scale))
                    sig = raw.clip(lower=-1.0, upper=1.0).where(carry_train.abs() >= float(thr), 0.0).fillna(0.0)
                    sig = sig.where(mask_train, 0.0).astype("float64")

                    for direction in (1, -1):
                        signal = (direction * sig).astype("float64")
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
                                "type": "clipped",
                                "threshold": float(thr),
                                "scale": float(scale),
                                "direction": int(direction),
                                "contango_mode": contango_mode,
                                "vol_mode": vol_mode,
                                "vol_threshold": vol_threshold,
                                "train_max_dd": float(bt.max_drawdown),
                                "train_nonzero_days": nonzero_days,
                            }

        if best_meta is None:
            continue

        # Test: deploy selected params for the test year using full history up to test_end.
        df_upto = df.loc[df.index <= test_end].copy()
        vol_upto = _compute_vol_series(price_exec=df_upto["s_exec"], contract_size=series.contract_size, config=config)
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

        s_pct = df_upto["s_signal_pct"].astype("float64")
        spacing = (df_upto["far_dte_bdays"] - df_upto["near_dte_bdays"]).astype("float64")
        spacing_years = spacing / float(config.trading_days_per_year)
        carry = (s_pct / spacing_years.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

        if best_meta["type"] == "sign":
            sig = -np.sign(carry).where(carry.abs() >= float(best_meta["threshold"]), 0.0).fillna(0.0)
        else:
            raw = -(carry / float(best_meta["scale"]))
            sig = raw.clip(lower=-1.0, upper=1.0).where(carry.abs() >= float(best_meta["threshold"]), 0.0).fillna(0.0)

        sig = sig.where(mask_upto, 0.0).astype("float64") * int(best_meta["direction"])
        sig.loc[sig.index < test_start] = 0.0

        bt = run_backtest_single_series(
            config=config,
            price_exec=df_upto["s_exec"],
            signal=sig,
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
                strategy="carry",
                spread=series.spread,
                params={
                    "type": best_meta["type"],
                    "threshold": best_meta["threshold"],
                    "scale": best_meta["scale"],
                    "direction": best_meta["direction"],
                },
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
    param_grid: list[MeanReversionParams],
    min_train_days: int = 252 * 3,
) -> WalkForwardOutput:
    df = series.df.copy().sort_index()
    cm = _cost_model(series, config)

    folds: list[FoldResult] = []
    oos_pnl_pieces: list[pd.Series] = []
    oos_years: list[int] = []

    for year in range(first_test_year, last_test_year + 1):
        train_end = pd.Timestamp(date(year - 1, 12, 31))
        test_start, test_end = _year_window(df.index, year)

        train_df = df.loc[df.index <= train_end]
        if len(train_df) < min_train_days:
            continue

        vol_full = _compute_vol_series(price_exec=df.loc[df.index <= test_end, "s_exec"], contract_size=series.contract_size, config=config)
        vol_threshold = float(vol_full.loc[train_df.index].median())

        best: float | None = None
        best_meta: dict[str, object] | None = None

        # Precompute base (unfiltered) MR positions for each params once per fold/year.
        base_positions: list[tuple[MeanReversionParams, pd.Series]] = []
        for params in param_grid:
            z = compute_dte_conditioned_zscore(train_df, params=params, value_col="s_signal_pct")
            pos = positions_from_zscore(
                z,
                entry_z=params.entry_z,
                exit_z=params.exit_z,
                max_hold_days=params.max_hold_days,
            ).astype("float64")
            base_positions.append((params, pos))

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

            for params, base_pos in base_positions:
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

        df_upto = df.loc[df.index <= test_end].copy()
        vol_upto = _compute_vol_series(price_exec=df_upto["s_exec"], contract_size=series.contract_size, config=config)
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

        params: MeanReversionParams = best_meta["params"]
        z = compute_dte_conditioned_zscore(df_upto, params=params, value_col="s_signal_pct")
        pos = positions_from_zscore(z, entry_z=params.entry_z, exit_z=params.exit_z, max_hold_days=params.max_hold_days)
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
                strategy="mean_reversion",
                spread=series.spread,
                params={**params.__dict__, "direction": best_meta["direction"]},
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
