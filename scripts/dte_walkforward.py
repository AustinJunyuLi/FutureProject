from __future__ import annotations

"""Walk-forward validation for DTE window strategies on S1 (with S2 regimes).

This uses the same inputs as the window scan, but performs an expanding
walk-forward selection by expiry year:

- Train on cycles with expiry_year < Y
- Select best candidate by net Sharpe-like (net_mean / net_std) with guardrails
- Test on cycles with expiry_year == Y
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from futures_curve.stage2.pipeline import read_spread_panel
from futures_curve.analysis.expiry_seasonality import DailySeriesConfig
from futures_curve.analysis.strategy_scan import CostModel, build_strategy_panel
from futures_curve.analysis.walkforward import walkforward_validate


LOGGER = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward validation for DTE window strategies (S1).")
    ap.add_argument("--data-dir", default="data_parquet", help="Stage 2 output directory")
    ap.add_argument("--symbol", default="HG", help="Commodity symbol")
    ap.add_argument("--start-year", type=int, default=2008)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--source", choices=["us_vwap", "bucket1"], default="us_vwap")
    ap.add_argument("--execution-shift", type=int, default=0)
    ap.add_argument("--entry-min", type=int, default=5)
    ap.add_argument("--entry-max", type=int, default=20)
    ap.add_argument("--exit-max", type=int, default=4)
    ap.add_argument("--min-train-trades", type=int, default=50)
    ap.add_argument("--min-train-years", type=int, default=5)
    ap.add_argument("--cost-ticks", type=float, default=1.0, help="Ticks per leg per side")
    ap.add_argument("--outdir", default="output/walkforward")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    outdir = _ensure_dir(Path(args.outdir))

    spread_panel = read_spread_panel(Path(args.data_dir), args.symbol)
    spread_panel["trade_date"] = pd.to_datetime(spread_panel["trade_date"])

    cfg = DailySeriesConfig(source=args.source, execution_shift_bdays=args.execution_shift)
    panel = build_strategy_panel(spread_panel, start_year=args.start_year, end_year=args.end_year, config=cfg)

    entry_dtes = range(args.entry_min, args.entry_max + 1)
    exit_dtes = range(0, args.exit_max + 1)
    cost_model = CostModel(ticks_per_leg_side=float(args.cost_ticks))

    selected, oos_trades, oos_summary = walkforward_validate(
        panel,
        start_year=args.start_year,
        end_year=args.end_year,
        entry_dtes=entry_dtes,
        exit_dtes=exit_dtes,
        min_train_trades=int(args.min_train_trades),
        enforce_positive_mean=True,
        min_train_years=int(args.min_train_years),
        cost_model=cost_model,
    )

    selected_path = outdir / "walkforward_selected_by_year.csv"
    trades_path = outdir / "walkforward_oos_trades.csv"
    summary_path = outdir / "walkforward_oos_summary.csv"

    selected.to_csv(selected_path, index=False)
    oos_trades.to_csv(trades_path, index=False)
    oos_summary.to_csv(summary_path, index=False)

    LOGGER.info("Wrote %s", selected_path)
    LOGGER.info("Wrote %s", trades_path)
    LOGGER.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
