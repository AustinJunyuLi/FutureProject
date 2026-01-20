from __future__ import annotations

"""Regime-stratified DTE event studies for S1 (expiry-anchored).

Outputs are written under the repo-level `output/` folder (gitignored).
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from futures_curve.analysis.event_study import (
    EntryRegimeSpec,
    cumulative_drift_by_regime,
    dte_drift_by_regime,
    plot_regime_frames,
)
from futures_curve.analysis.expiry_seasonality import DailySeriesConfig
from futures_curve.analysis.strategy_scan import build_strategy_panel
from futures_curve.stage2.pipeline import read_spread_panel


LOGGER = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Regime-stratified DTE event studies for S1.")
    ap.add_argument("--data-dir", default="data_parquet", help="Stage 2 output directory")
    ap.add_argument("--symbol", default="HG", help="Commodity symbol")
    ap.add_argument("--start-year", type=int, default=2008)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--dte-max", type=int, default=20)
    ap.add_argument("--entry-dte", type=int, required=True, help="Entry DTE used to define S2 regime (signal day).")

    ap.add_argument("--source", choices=["us_vwap", "bucket1"], default="us_vwap", help="S1 execution price source")
    ap.add_argument("--execution-shift", type=int, default=0, help="S1 execution shift (business days)")

    ap.add_argument("--s2-source", choices=["us_vwap", "bucket1"], default=None, help="S2 signal source (defaults to --source)")
    ap.add_argument("--s2-shift", type=int, default=0, help="S2 signal shift (business days)")

    ap.add_argument("--baseline-dte", type=int, default=20, help="Baseline DTE for cumulative drift normalization")
    ap.add_argument("--scenario", default=None, help="Output subfolder name under output/event_study/")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    scenario = (
        args.scenario
        or f"s1_{args.source}_shift{args.execution_shift}__s2_{args.s2_source or args.source}_shift{args.s2_shift}__entry{args.entry_dte}"
    )
    outdir = _ensure_dir(Path(args.outdir) / "event_study" / scenario)

    spread_panel = read_spread_panel(Path(args.data_dir), args.symbol)
    spread_panel["trade_date"] = pd.to_datetime(spread_panel["trade_date"])

    s2_source = args.s2_source or args.source
    s1_cfg = DailySeriesConfig(source=args.source, execution_shift_bdays=int(args.execution_shift))
    s2_cfg = DailySeriesConfig(source=s2_source, execution_shift_bdays=int(args.s2_shift))

    panel = build_strategy_panel(
        spread_panel,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        s1_config=s1_cfg,
        s2_config=s2_cfg,
    )

    regime_spec = EntryRegimeSpec(entry_dte=int(args.entry_dte), s2_col="s2", dte_col="dte_bdays", cycle_col="near_contract")

    # ΔS1 drift
    drift_frames = dte_drift_by_regime(panel, dte_max=int(args.dte_max), value_col="ds1", regime_spec=regime_spec)
    drift_dir = _ensure_dir(outdir / "dte_drift_ds1")
    for regime, df in drift_frames.items():
        df.to_csv(drift_dir / f"dte_drift_ds1_{regime}.csv")
    plot_regime_frames(
        drift_frames,
        title_prefix=f"ΔS1 by DTE — {args.start_year}-{args.end_year} (entry S2 regime @ DTE={args.entry_dte})",
        ylabel="ΔS1 (daily change)",
        output_dir=drift_dir,
        filename_prefix="dte_drift_ds1",
    )
    LOGGER.info("Wrote %s", drift_dir)

    # Cumulative drift path (S1 - S1@baseline_dte)
    cum_frames = cumulative_drift_by_regime(
        panel,
        dte_max=int(args.dte_max),
        baseline_dte=int(args.baseline_dte),
        level_col="s1",
        regime_spec=regime_spec,
    )
    cum_dir = _ensure_dir(outdir / "cumulative_s1")
    for regime, df in cum_frames.items():
        df.to_csv(cum_dir / f"cumulative_s1_{regime}.csv")
    plot_regime_frames(
        cum_frames,
        title_prefix=f"S1 - S1@DTE={args.baseline_dte} by DTE — {args.start_year}-{args.end_year} (entry S2 regime @ DTE={args.entry_dte})",
        ylabel=f"S1 - S1@DTE={args.baseline_dte}",
        output_dir=cum_dir,
        filename_prefix=f"cumulative_s1_from_dte{args.baseline_dte}",
    )
    LOGGER.info("Wrote %s", cum_dir)

    readme = f"""DTE Event Study (Regime-stratified)
================================

Symbol: {args.symbol}
Years: {args.start_year}-{args.end_year}
DTE cap: <= {args.dte_max}

Regime definition (entry-day S2):
- Entry DTE: {args.entry_dte}
- s2_pos: include cycles with S2(entry) >= 0
- s2_neg: include cycles with S2(entry) < 0

Series:
- S1 source: {args.source}
- S1 execution shift: {args.execution_shift}
- S2 source (signal): {s2_source}
- S2 shift (signal): {args.s2_shift}

Outputs:
- dte_drift_ds1/: mean/IQR/`n` for ΔS1 by DTE, by regime
- cumulative_s1/: mean/IQR/`n` for (S1 - S1@DTE={args.baseline_dte}) by DTE, by regime
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    LOGGER.info("Wrote %s", outdir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
