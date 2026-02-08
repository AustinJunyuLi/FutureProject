"""Tests for strategy signal generation and position logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_curve.analysis.strategies.carry import (
    CarryClippedStrategy,
    CarrySignStrategy,
    compute_annualized_carry_signal,
    _compute_carry_series,
)
from futures_curve.analysis.strategies.mean_reversion import (
    MeanReversionParams,
    compute_dte_conditioned_zscore,
    positions_from_zscore,
)
from futures_curve.analysis.data import SpreadDailySeries
from futures_curve.analysis.params import CarryClippedParams, CarrySignParams


def _make_spread_df(
    n: int = 20,
    s_signal_pct: float = 0.01,
    near_dte: float = 20.0,
    far_dte: float = 40.0,
    s_exec_base: float = 0.5,
) -> pd.DataFrame:
    """Build a minimal DataFrame matching SpreadDailySeries.df columns."""
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "s_signal_pct": s_signal_pct,
            "near_dte_bdays": near_dte,
            "far_dte_bdays": far_dte,
            "s_exec": s_exec_base + np.random.default_rng(42).normal(0, 0.01, n),
            "fnear_signal": 10.0,
            "ffar_signal": 10.5,
            "s_signal": 0.5,
        },
        index=idx,
    )


def _make_spread_series(df: pd.DataFrame) -> SpreadDailySeries:
    return SpreadDailySeries(spread="S1", df=df, contract_size=25000.0, dollars_per_tick=12.50)


# ---------- compute_annualized_carry_signal ----------

class TestComputeAnnualizedCarrySignal:
    def test_contango_gives_negative_direction(self) -> None:
        df = _make_spread_df(s_signal_pct=0.05, near_dte=20, far_dte=40)
        sig = compute_annualized_carry_signal(df)
        # Positive spread pct (contango) -> direction = -1
        assert (sig == -1.0).all()

    def test_backwardation_gives_positive_direction(self) -> None:
        df = _make_spread_df(s_signal_pct=-0.05, near_dte=20, far_dte=40)
        sig = compute_annualized_carry_signal(df)
        assert (sig == 1.0).all()

    def test_zero_spread_gives_zero(self) -> None:
        df = _make_spread_df(s_signal_pct=0.0, near_dte=20, far_dte=40)
        sig = compute_annualized_carry_signal(df)
        assert (sig == 0.0).all()

    def test_small_spacing_masked(self) -> None:
        df = _make_spread_df(s_signal_pct=0.05, near_dte=18, far_dte=20)
        sig = compute_annualized_carry_signal(df, min_spacing_bdays=5)
        # Spacing is only 2 bdays < 5, so all masked to 0
        assert (sig == 0.0).all()


# ---------- CarrySignStrategy ----------

class TestCarrySignStrategy:
    def test_interface(self) -> None:
        strat = CarrySignStrategy()
        assert strat.name == "carry_sign"
        grid = strat.default_param_grid()
        assert len(grid) > 0
        assert all(isinstance(p, CarrySignParams) for p in grid)

    def test_generate_signal_matches_sign(self) -> None:
        df = _make_spread_df(n=50, s_signal_pct=0.10, near_dte=20, far_dte=40)
        series = _make_spread_series(df)
        strat = CarrySignStrategy()
        sig = strat.generate_signal(series, CarrySignParams(threshold=0.0))
        # All contango -> should be -1
        assert (sig == -1.0).all()

    def test_threshold_filters(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
        s_pct = rng.normal(0.0, 0.02, n)
        df = pd.DataFrame(
            {
                "s_signal_pct": s_pct,
                "near_dte_bdays": 20.0,
                "far_dte_bdays": 40.0,
                "s_exec": 0.5,
                "fnear_signal": 10.0,
                "ffar_signal": 10.5,
                "s_signal": 0.5,
            },
            index=idx,
        )
        series = _make_spread_series(df)
        strat = CarrySignStrategy()
        sig = strat.generate_signal(series, CarrySignParams(threshold=100.0))
        # Threshold so high that nothing passes
        assert (sig == 0.0).all()

    def test_positions_from_signal_is_identity(self) -> None:
        strat = CarrySignStrategy()
        s = pd.Series([1.0, -1.0, 0.0], index=pd.date_range("2024-01-01", periods=3))
        assert (strat.positions_from_signal(s, CarrySignParams()) == s).all()


# ---------- CarryClippedStrategy ----------

class TestCarryClippedStrategy:
    def test_interface(self) -> None:
        strat = CarryClippedStrategy()
        assert strat.name == "carry_clipped"
        grid = strat.default_param_grid()
        assert len(grid) > 0
        assert all(isinstance(p, CarryClippedParams) for p in grid)

    def test_signal_clipped_to_minus_one_one(self) -> None:
        df = _make_spread_df(n=50, s_signal_pct=0.50, near_dte=20, far_dte=40)
        series = _make_spread_series(df)
        strat = CarryClippedStrategy()
        sig = strat.generate_signal(series, CarryClippedParams(threshold=0.0, scale=0.01))
        assert sig.max() <= 1.0
        assert sig.min() >= -1.0


# ---------- MR z-score ----------

class TestMeanReversionZScore:
    def test_zscore_output_shape(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2022-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "s_signal_pct": rng.normal(0.01, 0.005, n),
                "near_dte_bdays": np.tile(np.arange(10, 30), n // 20 + 1)[:n].astype(float),
            },
            index=idx,
        )
        params = MeanReversionParams(dte_bin_size=5, lookback_days=50)
        z = compute_dte_conditioned_zscore(df, params=params, value_col="s_signal_pct")
        assert len(z) == n
        # z should have some NaN at the start (warm-up) then finite values
        assert z.notna().sum() > 0

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"near_dte_bdays": [10.0]}, index=pd.date_range("2024-01-01", periods=1))
        params = MeanReversionParams()
        with pytest.raises(ValueError, match="Missing column"):
            compute_dte_conditioned_zscore(df, params=params, value_col="s_signal_pct")


# ---------- positions_from_zscore ----------

class TestPositionsFromZScore:
    def test_entry_and_exit(self) -> None:
        idx = pd.date_range("2024-01-01", periods=8)
        z = pd.Series([0.0, 1.5, 1.2, 0.3, 0.1, -1.5, -0.3, 0.0], index=idx)
        pos = positions_from_zscore(z, entry_z=1.0, exit_z=0.2)
        # z=1.5 >= 1.0 -> short entry (-1)
        assert pos.iloc[1] == -1.0
        # z=1.2: still short
        assert pos.iloc[2] == -1.0
        # z=0.3: still short (exit_z=0.2, so z=0.3 > 0.2 means NOT exit for short)
        # z=0.1: z <= exit_z -> exit
        assert pos.iloc[4] == 0.0
        # z=-1.5 <= -1.0 -> long entry (+1)
        assert pos.iloc[5] == 1.0

    def test_max_hold_days(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6)
        z = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], index=idx)
        pos = positions_from_zscore(z, entry_z=1.0, exit_z=0.2, max_hold_days=3)
        # Entry on day 0, hold_days increments each day while in position.
        # Day 0: entry, hold_days -> 1 (not yet >= 3)
        # Day 1: hold_days -> 2 (not yet >= 3)
        # Day 2: hold_days -> 3 (>= 3, exit)
        assert pos.iloc[0] == -1.0
        assert pos.iloc[1] == -1.0
        assert pos.iloc[2] == 0.0   # forced exit after 3 holds
        assert pos.iloc[3] == -1.0  # re-entry since z still >= entry_z

    def test_nan_preserves_state(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4)
        z = pd.Series([2.0, np.nan, np.nan, 0.1], index=idx)
        pos = positions_from_zscore(z, entry_z=1.0, exit_z=0.2)
        assert pos.iloc[0] == -1.0
        assert pos.iloc[1] == -1.0  # NaN -> keep state
        assert pos.iloc[2] == -1.0  # NaN -> keep state
        assert pos.iloc[3] == 0.0   # z=0.1 <= exit_z -> exit

    def test_all_zero_zscore(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5)
        z = pd.Series(0.0, index=idx)
        pos = positions_from_zscore(z, entry_z=1.0, exit_z=0.2)
        assert (pos == 0.0).all()
