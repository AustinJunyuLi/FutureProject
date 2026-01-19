"""End-to-end regression tests for the EOM strategy on a tiny deterministic slice.

These tests are designed to prevent regressions in the exact areas that caused
Archive vs refactor divergences:
  - exchange trading calendar vs naive weekday calendars (Thanksgiving session)
  - raw timestamp timezone normalization (vendor ET -> exchange CT)
  - expiry edge cases when Good Friday lands in March (handled in test_stage0)
  - EOM offsets/signals/trade generation + PnL math
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from futures_curve.stage0.trading_calendar import TradingCalendar
from futures_curve.stage1.pipeline import Stage1Pipeline
from futures_curve.stage1.timezone_inference import infer_raw_timezone_from_files
from futures_curve.stage2.pipeline import Stage2Pipeline, read_spread_panel
from futures_curve.stage3.eom_seasonality import build_eom_daily_dataset
from futures_curve.stage4.backtester import run_backtest
from futures_curve.stage4.strategies import EOMStrategy


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures_eom"


def test_trading_calendar_includes_thanksgiving_2020() -> None:
    cal = TradingCalendar("CMEGlobex_Metals")
    bdays = cal.get_business_days(date(2020, 11, 20), date(2020, 11, 30))
    assert pd.Timestamp("2020-11-26") in pd.to_datetime(bdays)


def test_eom_end_to_end_fixture_nov_2020(tmp_path: Path) -> None:
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

    # Stage 2: deterministic curve + spreads
    s2 = Stage2Pipeline(data_dir=data_dir)
    s2_stats = s2.process_symbol("HG", verbose=False)
    assert s2_stats["status"] == "success"

    spread_panel = read_spread_panel(data_dir, "HG")
    assert not spread_panel.empty

    # Stage 3 (subset): daily US-session S1 proxy + EOM offsets
    daily = build_eom_daily_dataset(spread_panel, spread_col="S1")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])

    # EOM offsets for the last business days of Nov 2020.
    expected_offsets = {
        pd.Timestamp("2020-11-24"): 4,
        pd.Timestamp("2020-11-25"): 3,
        pd.Timestamp("2020-11-26"): 2,
        pd.Timestamp("2020-11-27"): 1,
        pd.Timestamp("2020-11-30"): 0,
    }
    for td, off in expected_offsets.items():
        row = daily.loc[daily["trade_date"] == td].iloc[0]
        assert int(row["eom_offset"]) == off

    # Key S1 levels (from the fixture construction)
    s1_1124 = float(daily.loc[daily["trade_date"] == pd.Timestamp("2020-11-24"), "S1"].iloc[0])
    s1_1125 = float(daily.loc[daily["trade_date"] == pd.Timestamp("2020-11-25"), "S1"].iloc[0])
    s1_1127 = float(daily.loc[daily["trade_date"] == pd.Timestamp("2020-11-27"), "S1"].iloc[0])
    assert abs(s1_1124 - (-0.0060)) < 1e-12
    assert abs(s1_1125 - (-0.0020)) < 1e-12
    # Weighted daily average across the included buckets
    assert abs(s1_1127 - 0.00704) < 5e-5

    # Signal series (causal alignment under next-bucket execution)
    strat = EOMStrategy(entry_offset=3, exit_offset=1, execution_delay_bdays=1)
    signals = strat.generate_signals(daily)
    assert any(s.metadata.get("action") == "entry" and int(s.metadata.get("eom_offset")) == 4 for s in signals)
    assert any(s.metadata.get("action") == "exit" and int(s.metadata.get("eom_offset")) == 2 for s in signals)

    # Stage 4: backtest (EOM only) and assert trade + PnL math
    res = run_backtest(
        data_dir=data_dir,
        symbol="HG",
        strategy_name="eom",
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
    assert len(trades) == 1
    t = trades.iloc[0]

    assert str(t["entry_date"]) == "2020-11-25"
    assert str(t["exit_date"]) == "2020-11-27"
    assert t["direction"] == 1

    # Gross PnL (USD) = Δspread -> ticks -> USD, then subtract costs
    delta_ticks = (float(t["exit_price"]) - float(t["entry_price"])) / 0.0005
    gross = delta_ticks * 12.50 * float(t["contracts"]) * float(t["direction"])
    expected_net = gross - (float(t["slippage_cost"]) + float(t["commission_cost"]))
    assert abs(float(t["pnl"]) - expected_net) < 1e-6
