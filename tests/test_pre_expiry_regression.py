"""End-to-end regression tests for the expiry-anchored pre-expiry strategy.

These tests are designed to prevent regressions in the exact areas that caused
Archive vs refactor divergences:
  - exchange trading calendar vs naive weekday calendars (Thanksgiving session)
  - raw timestamp timezone normalization (vendor ET -> exchange CT)
  - expiry-ranked curve construction and DTE computation
  - pre-expiry entry/exit rule execution + PnL math (net of costs)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from futures_curve.stage0.trading_calendar import TradingCalendar
from futures_curve.stage1.pipeline import Stage1Pipeline
from futures_curve.stage1.timezone_inference import infer_raw_timezone_from_files
from futures_curve.stage2.pipeline import Stage2Pipeline, read_spread_panel
from futures_curve.stage4.aggregation import build_us_session_daily_vwap_panel
from futures_curve.stage4.backtester import run_backtest


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures_pre_expiry"


def test_trading_calendar_includes_thanksgiving_2020() -> None:
    cal = TradingCalendar("CMEGlobex_Metals")
    bdays = cal.get_business_days(date(2020, 11, 20), date(2020, 11, 30))
    assert pd.Timestamp("2020-11-26") in pd.to_datetime(bdays)


def test_pre_expiry_end_to_end_fixture_nov_2020(tmp_path: Path) -> None:
    fixture_root = _fixture_root()
    raw_root = fixture_root  # contains copper/...

    # Confirm raw timezone inference on this fixture.
    tz = infer_raw_timezone_from_files(
        [
            raw_root / "copper" / "HG_X20_1min.txt",
            raw_root / "copper" / "HG_Z20_1min.txt",
            raw_root / "copper" / "HG_F21_1min.txt",
        ],
        max_files=10,
        max_rows_per_file=5000,
    ).raw_timezone
    assert tz == "US/Eastern"

    data_dir = tmp_path / "data_parquet"

    # Stage 1: ingest fixture raw -> buckets/daily parquet
    s1 = Stage1Pipeline(raw_data_dir=raw_root, output_dir=data_dir, chunk_size=50_000)
    s1_stats = s1.process_symbol("HG", start_year=2020, end_year=2021, verbose=False)
    assert s1_stats["status"] == "success"
    assert s1_stats["contracts_processed"] == 3

    # Stage 2: deterministic curve + spreads (includes F1_dte_bdays)
    s2 = Stage2Pipeline(data_dir=data_dir)
    s2_stats = s2.process_symbol("HG", verbose=False)
    assert s2_stats["status"] == "success"

    spread_panel = read_spread_panel(data_dir, "HG")
    assert not spread_panel.empty
    assert "F1_dte_bdays" in spread_panel.columns

    # Daily proxy aggregation sanity check (used for comparison vs Archive-style reports).
    daily = build_us_session_daily_vwap_panel(
        spread_panel,
        spread_col="S1",
        dte_col="F1_dte_bdays",
        buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        pair_policy="mode",
    )
    assert not daily.empty
    assert daily["trade_date"].is_unique
    assert (daily["bucket"].astype(int) == 1).all()
    assert daily["S1"].notna().any()

    # Stage 4: run expiry-anchored backtest. We intentionally search in the
    # "last few business days" window, but keep the test deterministic by fixing
    # parameters on this small fixture.
    entry_dte = 3
    exit_dte = 1
    res = run_backtest(
        data_dir=data_dir,
        symbol="HG",
        strategy_name="pre_expiry",
        strategy_params={"entry_dte": entry_dte, "exit_dte": exit_dte},
        execution_config={
            "slippage_ticks": 1.0,
            "tick_size": 0.0005,
            "tick_value": 12.50,
            "commission_per_contract": 2.50,
            "initial_capital": 100000.0,
        },
        stop_loss_usd=None,
        max_holding_bdays=None,
        allow_same_bucket_execution=False,
        auto_roll_on_contract_change=True,
    )
    assert res["status"] == "success"

    trades = res["trades"]
    assert len(trades) >= 1
    t = trades.iloc[0]

    # Validate the entry/exit are in a pre-expiry DTE window.
    entry_row = spread_panel.loc[
        (pd.to_datetime(spread_panel["trade_date"]) == pd.Timestamp(t["entry_date"]))
        & (spread_panel["bucket"].astype(int) == int(t["entry_bucket"]))
    ]
    exit_row = spread_panel.loc[
        (pd.to_datetime(spread_panel["trade_date"]) == pd.Timestamp(t["exit_date"]))
        & (spread_panel["bucket"].astype(int) == int(t["exit_bucket"]))
    ]
    assert len(entry_row) == 1
    assert len(exit_row) == 1

    entry_dte_observed = int(entry_row.iloc[0]["F1_dte_bdays"])
    exit_dte_observed = int(exit_row.iloc[0]["F1_dte_bdays"])

    assert exit_dte < entry_dte_observed <= entry_dte
    assert exit_dte_observed <= exit_dte

    # Gross PnL (USD) = Δspread -> ticks -> USD, then subtract costs
    delta_ticks = (float(t["exit_price"]) - float(t["entry_price"])) / 0.0005
    gross = delta_ticks * 12.50 * float(t["contracts"]) * float(t["direction"])
    expected_net = gross - (float(t["slippage_cost"]) + float(t["commission_cost"]))
    assert abs(float(t["pnl"]) - expected_net) < 1e-6

    # Daily US-session proxy backtest should also run end-to-end without errors.
    res_daily = run_backtest(
        data_dir=data_dir,
        symbol="HG",
        strategy_name="pre_expiry",
        strategy_params={"entry_dte": entry_dte, "exit_dte": exit_dte},
        execution_config={
            "slippage_ticks": 1.0,
            "tick_size": 0.0005,
            "tick_value": 12.50,
            "commission_per_contract": 2.50,
            "initial_capital": 100000.0,
        },
        data_frequency="daily_us_vwap",
        daily_agg_config={
            "spread_col": "S1",
            "dte_col": "F1_dte_bdays",
            "buckets": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            "pair_policy": "mode",
        },
        stop_loss_usd=None,
        max_holding_bdays=None,
        allow_same_bucket_execution=False,
        auto_roll_on_contract_change=True,
    )
    assert res_daily["status"] in {"success", "no_signals"}
