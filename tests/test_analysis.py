from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.data import build_daily_spread_series
from futures_curve.analysis.metrics import compute_drawdown_series
from futures_curve.analysis.risk import position_from_vol_target


def test_compute_drawdown_series() -> None:
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0], index=pd.date_range("2024-01-01", periods=5))
    dd = compute_drawdown_series(equity)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0
    assert np.isclose(dd.iloc[2], (110.0 - 105.0) / 110.0)
    assert dd.iloc[3] == 0.0
    assert np.isclose(dd.max(), 0.25)


def test_position_from_vol_target_nan_safe() -> None:
    idx = pd.date_range("2024-01-01", periods=4)
    signal = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    vol = pd.Series([np.nan, 2.0, 2.0, 0.0], index=idx)
    pos = position_from_vol_target(
        signal=signal,
        vol_per_contract=vol,
        target_dollar_vol=10.0,
        allow_fractional=True,
    )
    assert pos.iloc[0] == 0.0  # NaN vol -> 0
    assert np.isclose(pos.iloc[1], 5.0)
    assert np.isclose(pos.iloc[2], 5.0)
    assert pos.iloc[3] == 0.0  # zero vol -> 0


def test_build_daily_spread_series_uses_us_session_vwap_and_exec_bucket() -> None:
    # Minimal synthetic spreads_panel-like input for S1 (F1/F2).
    trade_dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    rows: list[dict[str, object]] = []
    for td in trade_dates:
        for b in range(1, 8):  # signal buckets 1..7
            rows.append(
                {
                    "trade_date": td,
                    "bucket": b,
                    "F1_price": 10.0 + 0.1 * b,
                    "F2_price": 10.5 + 0.1 * b,
                    "F1_volume": float(b),
                    "F2_volume": float(2 * b),
                    "F1_contract": "HGH24",
                    "F2_contract": "HGJ24",
                    "F1_dte_bdays": 20.0,
                    "F2_dte_bdays": 40.0,
                }
            )
        # execution bucket (bucket=1) row already included above; keep it consistent.

    panel = pd.DataFrame(rows)
    cfg = AnalysisConfig(symbol="HG", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

    series = build_daily_spread_series(config=cfg, spreads_panel=panel, spread_num=1)
    out = series.df

    assert list(out.index) == trade_dates
    # Spread is constant difference (0.5) for both signal and execution in this synthetic setup.
    assert np.allclose(out["s_signal"].values, 0.5, equal_nan=False)
    assert np.allclose(out["s_exec"].values, 0.5, equal_nan=False)
    assert np.allclose(out["s_signal_pct"].values, 0.5 / out["fnear_signal"].values)

