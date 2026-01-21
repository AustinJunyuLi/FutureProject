from __future__ import annotations

"""
DTE strategy discovery scan for S1 (with S2 context).

Generates:
- strategy_panel.csv (daily, DTE-anchored, S1 + S2)
- dte_drift_s1_average.png + CSV
- window_scan_summary.csv
- heatmaps for mean net PnL and t-stat (long/short)
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_curve.stage2.pipeline import read_spread_panel
from futures_curve.analysis.expiry_seasonality import DailySeriesConfig
from futures_curve.analysis.strategy_scan import (
    CostModel,
    build_strategy_panel,
    compute_dte_drift_stats,
    scan_windows,
)


LOGGER = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_heatmap(
    summary: pd.DataFrame,
    *,
    direction: str,
    metric: str,
    title: str,
    output_png: Path,
    entry_min: int,
    entry_max: int,
    exit_max: int,
) -> None:
    sub = summary[summary["direction"] == direction].copy()
    if sub.empty:
        LOGGER.warning("No summary rows for %s %s; skipping heatmap.", direction, metric)
        return

    # Build grid
    entry_vals = list(range(entry_max, entry_min - 1, -1))
    exit_vals = list(range(exit_max, -1, -1))

    pivot = sub.pivot(index="entry_dte", columns="exit_dte", values=metric)
    grid = np.full((len(entry_vals), len(exit_vals)), np.nan, dtype=float)
    for i, entry in enumerate(entry_vals):
        for j, exit_dte in enumerate(exit_vals):
            if entry in pivot.index and exit_dte in pivot.columns:
                grid[i, j] = pivot.loc[entry, exit_dte]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="coolwarm", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Exit DTE")
    ax.set_ylabel("Entry DTE")
    ax.set_xticks(range(len(exit_vals)))
    ax.set_xticklabels(exit_vals)
    ax.set_yticks(range(len(entry_vals)))
    ax.set_yticklabels(entry_vals)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def _rank_top(
    summary: pd.DataFrame,
    *,
    metric: str,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for regime in sorted(summary["regime"].unique()):
        for direction in sorted(summary["direction"].unique()):
            sub = summary[(summary["regime"] == regime) & (summary["direction"] == direction)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(metric, ascending=False).head(top_n).copy()
            sub["rank_metric"] = metric
            rows.append(sub)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description="DTE window scan for S1 strategy discovery.")
    ap.add_argument("--data-dir", default="data_parquet", help="Stage 2 output directory")
    ap.add_argument("--symbol", default="HG", help="Commodity symbol")
    ap.add_argument("--start-year", type=int, default=2008)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument(
        "--s1-source",
        choices=["rest_vwap", "us_vwap", "bucket1"],
        default="rest_vwap",
        help="S1 execution price source (baseline: rest_vwap = VWAP over buckets 2-7).",
    )
    ap.add_argument(
        "--s2-source",
        choices=["bucket1", "rest_vwap", "us_vwap"],
        default="bucket1",
        help="S2 signal source (baseline: bucket1 close).",
    )
    ap.add_argument("--s1-shift", type=int, default=0, help="Business-day shift for S1 execution price")
    ap.add_argument("--s2-shift", type=int, default=0, help="Business-day shift for S2 signal")
    ap.add_argument("--dte-max", type=int, default=20)
    ap.add_argument("--entry-min", type=int, default=5)
    ap.add_argument("--entry-max", type=int, default=20)
    ap.add_argument("--exit-max", type=int, default=4)
    ap.add_argument("--cost-ticks", type=float, default=1.0, help="Ticks per leg per side")
    ap.add_argument("--top-n", type=int, default=10, help="Top N windows per regime/direction")
    ap.add_argument("--outdir", default="output/strategy_scan")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    outdir = _ensure_dir(Path(args.outdir))
    panel_out = outdir / "strategy_panel.csv"

    spread_panel = read_spread_panel(Path(args.data_dir), args.symbol)
    spread_panel["trade_date"] = pd.to_datetime(spread_panel["trade_date"])

    s1_cfg = DailySeriesConfig(source=args.s1_source, execution_shift_bdays=args.s1_shift)
    s2_cfg = DailySeriesConfig(source=args.s2_source, execution_shift_bdays=args.s2_shift)
    panel = build_strategy_panel(
        spread_panel,
        start_year=args.start_year,
        end_year=args.end_year,
        s1_config=s1_cfg,
        s2_config=s2_cfg,
    )
    panel.to_csv(panel_out, index=False)
    LOGGER.info("Wrote %s", panel_out)

    # DTE drift stats for ΔS1
    drift_png = outdir / "dte_drift_s1_average.png"
    drift_csv = outdir / "dte_drift_s1_average.csv"
    title = f"ΔS1 by DTE — {args.dte_max}-day window (S1={args.s1_source}, S2={args.s2_source})"
    compute_dte_drift_stats(panel, dte_max=args.dte_max, output_png=drift_png, output_csv=drift_csv, title=title)
    LOGGER.info("Wrote %s", drift_png)
    LOGGER.info("Wrote %s", drift_csv)

    # Window scan
    entry_dtes = range(args.entry_min, args.entry_max + 1)
    exit_dtes = range(0, args.exit_max + 1)
    cost_model = CostModel(ticks_per_leg_side=float(args.cost_ticks))

    summary = scan_windows(panel, entry_dtes=entry_dtes, exit_dtes=exit_dtes, cost_model=cost_model)
    summary_path = outdir / "window_scan_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Wrote %s", summary_path)

    # Rank top N windows by net mean and net t-stat (per regime + direction)
    top_mean = _rank_top(summary, metric="net_mean", top_n=args.top_n)
    top_t = _rank_top(summary, metric="net_t", top_n=args.top_n)
    top_out = outdir / "window_scan_topN.csv"
    if not top_mean.empty or not top_t.empty:
        top_all = pd.concat([top_mean, top_t], ignore_index=True)
        top_all.to_csv(top_out, index=False)
        LOGGER.info("Wrote %s", top_out)

    # Write a short README with assumptions
    readme = f"""
DTE Strategy Scan (S1)
======================

Inputs
- Symbol: {args.symbol}
- Years: {args.start_year}–{args.end_year}
- S1 execution series: {args.s1_source} (shift={args.s1_shift})
- S2 signal series: {args.s2_source} (shift={args.s2_shift})
- DTE window: entry {args.entry_min}–{args.entry_max}, exit 0–{args.exit_max}

Cost model
- Tick size (HG): $0.0005 per lb (COMEX copper)
- Contract size: 25,000 lb
- Ticks per leg per side: {args.cost_ticks}
- Round-trip spread cost: {cost_model.round_trip_spread_cost:.4f} $/lb
- Round-trip cost per contract: ${cost_model.round_trip_dollar_cost:,.2f}
- Source: CME Group (tick size / contract size): https://www.cmegroup.com/articles/faqs/micro-copper-futures-faq.html

This scan reports gross and net metrics (after the round-trip cost).
Regimes:
- all: no S2 filter
- s2_pos: S2 at entry >= 0
- s2_neg: S2 at entry < 0
"""
    (outdir / "README.md").write_text(readme.strip(), encoding="utf-8")

    # Heatmaps: net mean and t-stat (per regime)
    for regime in sorted(summary["regime"].unique()):
        sub = summary[summary["regime"] == regime].copy()
        if sub.empty:
            continue
        label = regime.replace(" ", "_")
        _plot_heatmap(
            sub,
            direction="long",
            metric="net_mean",
            title=f"Long S1 — Net Mean PnL ({regime})",
            output_png=outdir / f"heatmap_{label}_long_net_mean.png",
            entry_min=args.entry_min,
            entry_max=args.entry_max,
            exit_max=args.exit_max,
        )
        _plot_heatmap(
            sub,
            direction="short",
            metric="net_mean",
            title=f"Short S1 — Net Mean PnL ({regime})",
            output_png=outdir / f"heatmap_{label}_short_net_mean.png",
            entry_min=args.entry_min,
            entry_max=args.entry_max,
            exit_max=args.exit_max,
        )
        _plot_heatmap(
            sub,
            direction="long",
            metric="net_t",
            title=f"Long S1 — Net t-stat ({regime})",
            output_png=outdir / f"heatmap_{label}_long_net_t.png",
            entry_min=args.entry_min,
            entry_max=args.entry_max,
            exit_max=args.exit_max,
        )
        _plot_heatmap(
            sub,
            direction="short",
            metric="net_t",
            title=f"Short S1 — Net t-stat ({regime})",
            output_png=outdir / f"heatmap_{label}_short_net_t.png",
            entry_min=args.entry_min,
            entry_max=args.entry_max,
            exit_max=args.exit_max,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
