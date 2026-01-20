"""Regression tests for report generation (HG).

These tests ensure the analysis report remains:
- syntactically valid (no f-string brace regressions)
- structurally complete (required sections present)
- robust to missing optional artifacts (figures skipped in CI)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from futures_curve.reporting import build_analysis_report


def test_build_analysis_report_tex_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_parquet"
    research_dir = tmp_path / "research_outputs"

    # Minimal inputs consumed by build_analysis_report()
    (data_dir / "qc").mkdir(parents=True, exist_ok=True)
    (data_dir / "trades" / "HG").mkdir(parents=True, exist_ok=True)
    (data_dir / "spreads" / "HG").mkdir(parents=True, exist_ok=True)
    (research_dir / "tables").mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"contracts_processed": 3, "total_minute_rows": 12345}]).to_parquet(
        data_dir / "qc" / "HG_qc.parquet", index=False
    )

    # Minimal Stage 2 spread panel for coverage text (optional in report)
    pd.DataFrame(
        [
            {"trade_date": "2020-11-24", "bucket": 1, "S1": -0.01},
            {"trade_date": "2020-11-30", "bucket": 1, "S1": 0.02},
        ]
    ).to_parquet(data_dir / "spreads" / "HG" / "spreads_panel.parquet", index=False)

    # Minimal Stage 3 tables (optional in report)
    pd.DataFrame(
        [
            {"dte_bin": "0-5", "pct_contango": 60.0, "mean": 0.001, "median": 0.0008},
            {"dte_bin": "6-10", "pct_contango": 55.0, "mean": 0.0005, "median": 0.0004},
        ]
    ).to_parquet(research_dir / "tables" / "HG_dte_profile.parquet", index=False)

    # Backtest summary + matching trade logs (so the report can compute gross vs net).
    pd.DataFrame(
        [
            {
                "symbol": "HG",
                "strategy": "pre_expiry",
                "total_trades": 1,
                "win_rate": 100.0,
                "total_pnl": 1000.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -2.0,
                "profit_factor": 2.0,
            },
        ]
    ).to_parquet(data_dir / "trades" / "HG" / "summary.parquet", index=False)

    pd.DataFrame(
        [
            {
                "status": "closed",
                "pnl": 1000.0,
                "slippage_cost": 50.0,
                "commission_cost": 10.0,
            }
        ]
    ).to_parquet(data_dir / "trades" / "HG" / "pre_expiry_trades.parquet", index=False)

    # Optional: pre-expiry sweep table (used in Strategy Backtesting section).
    pd.DataFrame(
        [
            {
                "symbol": "HG",
                "strategy": "pre_expiry",
                "entry_dte": 5,
                "exit_dte": 1,
                "total_trades": 1,
                "win_rate": 100.0,
                "total_pnl": 1000.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -2.0,
            }
        ]
    ).to_parquet(data_dir / "trades" / "HG" / "pre_expiry_sweep.parquet", index=False)

    # Optional: stop-loss sensitivity tables (used in the robustness section).
    pd.DataFrame(
        [
            {
                "stop_loss_usd": None,
                "total_trades": 1,
                "win_rate": 100.0,
                "gross_pnl": 1060.0,
                "total_costs": 60.0,
                "total_pnl": 1000.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -2.0,
            },
            {
                "stop_loss_usd": 100.0,
                "total_trades": 1,
                "win_rate": 100.0,
                "gross_pnl": 1060.0,
                "total_costs": 60.0,
                "total_pnl": 1000.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -2.0,
            },
        ]
    ).to_parquet(data_dir / "trades" / "HG" / "pre_expiry_stop_loss_sensitivity.parquet", index=False)

    tex_path = build_analysis_report(
        symbol="HG",
        data_dir=data_dir,
        research_dir=research_dir,
        compile_pdf=False,
        generate_figures=False,
    )

    assert tex_path.exists()
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\section{Executive Summary}" in tex
    assert "\\section{Data and Preprocessing}" in tex
    assert "\\section{Strategy Backtesting}" in tex
    assert "\\section{Sensitivity and Robustness}" in tex
    assert "Stop-Loss Sensitivity" in tex
    assert "\\texttt{pre\\_expiry}" in tex
