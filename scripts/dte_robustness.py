from __future__ import annotations

"""Fixed-rule robustness runs for DTE window strategies (S1).

This script:
1) Selects a single baseline rule on the baseline scenario (by net Sharpe-like).
2) Evaluates the *same* rule under alternate execution assumptions without re-selection.

Outputs are written under the repo-level `output/robustness/` folder (gitignored).
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from futures_curve.analysis.expiry_seasonality import DailySeriesConfig
from futures_curve.analysis.robustness import evaluate_fixed_rule, select_best_rule
from futures_curve.analysis.strategy_scan import CostModel, build_strategy_panel
from futures_curve.stage2.pipeline import read_spread_panel


LOGGER = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Scenario:
    name: str
    s1_source: str
    s1_shift: int
    cost_ticks: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed-rule robustness runs for DTE window strategies (S1).")
    ap.add_argument("--data-dir", default="data_parquet", help="Stage 2 output directory")
    ap.add_argument("--symbol", default="HG", help="Commodity symbol")
    ap.add_argument("--start-year", type=int, default=2008)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--entry-min", type=int, default=5)
    ap.add_argument("--entry-max", type=int, default=20)
    ap.add_argument("--exit-max", type=int, default=4)
    ap.add_argument("--min-trades", type=int, default=50)
    ap.add_argument("--rank-top-n", type=int, default=50)
    ap.add_argument(
        "--baseline-s1-source",
        choices=["rest_vwap", "us_vwap", "bucket1"],
        default="rest_vwap",
        help="Baseline S1 execution source (default: rest_vwap).",
    )
    ap.add_argument(
        "--baseline-s2-source",
        choices=["bucket1", "rest_vwap", "us_vwap"],
        default="bucket1",
        help="Baseline S2 signal source (default: bucket1).",
    )
    ap.add_argument("--baseline-cost-ticks", type=float, default=1.0, help="Ticks per leg per side (baseline)")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    out_root = _ensure_dir(Path(args.outdir) / "robustness")

    spread_panel = read_spread_panel(Path(args.data_dir), args.symbol)
    spread_panel["trade_date"] = pd.to_datetime(spread_panel["trade_date"])

    # Fixed signal panel (baseline): used for S2 regime membership at entry.
    s1_baseline_cfg = DailySeriesConfig(source=args.baseline_s1_source, execution_shift_bdays=0)
    s2_baseline_cfg = DailySeriesConfig(source=args.baseline_s2_source, execution_shift_bdays=0)
    signal_panel = build_strategy_panel(
        spread_panel,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        s1_config=s1_baseline_cfg,
        s2_config=s2_baseline_cfg,
    )

    entry_dtes = range(int(args.entry_min), int(args.entry_max) + 1)
    exit_dtes = range(0, int(args.exit_max) + 1)
    baseline_cost_model = CostModel(ticks_per_leg_side=float(args.baseline_cost_ticks))

    # Select baseline rule on baseline scenario (by net Sharpe-like, with guardrails).
    rule, ranked = select_best_rule(
        signal_panel,
        entry_dtes=entry_dtes,
        exit_dtes=exit_dtes,
        cost_model=baseline_cost_model,
        min_trades=int(args.min_trades),
        enforce_positive_mean=True,
        rank_metric="net_sharpe_like",
    )

    ranked_path = out_root / "baseline_ranked_windows.csv"
    ranked.head(int(args.rank_top_n)).to_csv(ranked_path, index=False)
    (out_root / "baseline_rule.json").write_text(json.dumps(asdict(rule), indent=2), encoding="utf-8")
    LOGGER.info("Selected baseline rule: %s", rule)
    LOGGER.info("Wrote %s", ranked_path)

    scenarios = [
        Scenario(name="baseline", s1_source=args.baseline_s1_source, s1_shift=0, cost_ticks=float(args.baseline_cost_ticks)),
        Scenario(name="exec_bucket1", s1_source="bucket1", s1_shift=0, cost_ticks=float(args.baseline_cost_ticks)),
        Scenario(name="exec_us_vwap", s1_source="us_vwap", s1_shift=0, cost_ticks=float(args.baseline_cost_ticks)),
        Scenario(name="baseline_cost2", s1_source=args.baseline_s1_source, s1_shift=0, cost_ticks=float(args.baseline_cost_ticks) * 2.0),
    ]

    summary_rows = []
    for scenario in scenarios:
        scenario_dir = _ensure_dir(out_root / scenario.name)

        exec_cfg = DailySeriesConfig(source=scenario.s1_source, execution_shift_bdays=int(scenario.s1_shift))
        # Keep S2 on the baseline signal definition so regime membership stays fixed.
        execution_panel = build_strategy_panel(
            spread_panel,
            start_year=int(args.start_year),
            end_year=int(args.end_year),
            s1_config=exec_cfg,
            s2_config=s2_baseline_cfg,
        )

        cost_model = CostModel(ticks_per_leg_side=float(scenario.cost_ticks))
        trades, by_year = evaluate_fixed_rule(execution_panel, rule=rule, signal_panel=signal_panel, cost_model=cost_model)

        trades_path = scenario_dir / "trades.csv"
        by_year_path = scenario_dir / "summary_by_expiry_year.csv"
        trades.to_csv(trades_path, index=False)
        by_year.to_csv(by_year_path, index=False)
        LOGGER.info("Wrote %s", trades_path)
        LOGGER.info("Wrote %s", by_year_path)

        pooled = by_year[by_year["expiry_year"] == "pooled"].copy()
        pooled_row = pooled.iloc[0].to_dict() if not pooled.empty else {}
        summary_rows.append(
            {
                "scenario": scenario.name,
                "s1_source": scenario.s1_source,
                "s1_shift": scenario.s1_shift,
                "cost_ticks_per_leg_side": scenario.cost_ticks,
                **asdict(rule),
                **pooled_row,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Wrote %s", summary_path)

    readme = f"""Robustness (Fixed Rule)
=====================

Signal is held fixed:
- S2 regime membership is defined using baseline signal panel:
  - S2 source={args.baseline_s2_source}
  - shift=0

Rule selection:
- Objective: net Sharpe-like (net_mean / net_std)
- Guardrails: n>={args.min_trades}, net_mean>0
- Candidate search: entry_dte={args.entry_min}..{args.entry_max}, exit_dte=0..{args.exit_max}, both directions, regimes={{all,s2_pos,s2_neg}}

Execution scenarios:
- baseline: {args.baseline_s1_source}, shift=0, cost={args.baseline_cost_ticks} ticks/leg/side
- exec_bucket1: bucket1 close, shift=0, cost={args.baseline_cost_ticks}
- exec_us_vwap: us_vwap (buckets 1-7), shift=0, cost={args.baseline_cost_ticks}
- baseline_cost2: {args.baseline_s1_source}, shift=0, cost doubled

Files:
- baseline_rule.json: selected rule parameters
- baseline_ranked_windows.csv: top ranked windows (truncated to --rank-top-n)
- <scenario>/: trades.csv + summary_by_expiry_year.csv
- summary.csv: pooled comparison across scenarios
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")
    LOGGER.info("Wrote %s", out_root / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
