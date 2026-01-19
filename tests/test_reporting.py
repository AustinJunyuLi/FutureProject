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
    (research_dir / "tables").mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"contracts_processed": 3, "total_minute_rows": 12345}]).to_parquet(
        data_dir / "qc" / "HG_qc.parquet", index=False
    )
    pd.DataFrame([{"trade_date": "2020-11-24"}, {"trade_date": "2020-11-30"}]).to_parquet(
        research_dir / "tables" / "HG_eom_daily.parquet", index=False
    )

    # Backtest summary + matching trade logs (so the report can compute gross vs net).
    pd.DataFrame(
        [
            {
                "symbol": "HG",
                "strategy": "eom",
                "total_trades": 1,
                "win_rate": 100.0,
                "total_pnl": 1000.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -2.0,
                "profit_factor": 2.0,
            },
            {
                "symbol": "HG",
                "strategy": "dte",
                "total_trades": 1,
                "win_rate": 0.0,
                "total_pnl": -500.0,
                "sharpe_ratio": -1.0,
                "max_drawdown_pct": -5.0,
                "profit_factor": 0.5,
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
    ).to_parquet(data_dir / "trades" / "HG" / "eom_trades.parquet", index=False)

    pd.DataFrame(
        [
            {
                "status": "closed",
                "pnl": -500.0,
                "slippage_cost": 50.0,
                "commission_cost": 10.0,
            }
        ]
    ).to_parquet(data_dir / "trades" / "HG" / "dte_trades.parquet", index=False)

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
    ).to_parquet(data_dir / "trades" / "HG" / "eom_stop_loss_sensitivity.parquet", index=False)

    pd.DataFrame(
        [
            {
                "stop_loss_usd": None,
                "total_trades": 1,
                "win_rate": 0.0,
                "gross_pnl": -440.0,
                "total_costs": 60.0,
                "total_pnl": -500.0,
                "sharpe_ratio": -1.0,
                "max_drawdown_pct": -5.0,
            }
        ]
    ).to_parquet(data_dir / "trades" / "HG" / "dte_stop_loss_sensitivity.parquet", index=False)

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
    assert "EOM is profitable in this sample" in tex
