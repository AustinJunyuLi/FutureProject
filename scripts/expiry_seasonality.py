from __future__ import annotations

"""Generate expiry-anchored seasonality plots for S1/S2.

This script reads Stage 2 spreads_panel.parquet and builds daily series
aligned by business-day DTE (near-leg DTE for the spread). It then produces
overlay and average/IQR plots by DTE.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from futures_curve.stage2.pipeline import read_spread_panel
from futures_curve.analysis.expiry_seasonality import (
    DailySeriesConfig,
    build_daily_spread_series,
    align_by_dte,
    plot_overlay_by_dte,
    plot_average_by_dte,
)

LOGGER = logging.getLogger(__name__)


def _filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:59:59")
    return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()


def _parse_spreads(spreads: str) -> list[str]:
    return [s.strip().upper() for s in spreads.split(",") if s.strip()]


def _filter_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31 23:59:59")
    return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()


def _run_one(
    *,
    panel: pd.DataFrame,
    year: int,
    spread: str,
    metric: str,
    cfg: DailySeriesConfig,
    dte_max: int,
    outdir: Path,
) -> None:
    daily = build_daily_spread_series(panel, spread, config=cfg)
    daily = _filter_year(daily, year)

    if metric == "diff":
        # Daily change within each near-leg contract cycle
        daily = daily.sort_values(["near_contract", "trade_date"]).copy()
        daily["value"] = daily.groupby("near_contract", sort=False)["value"].diff()
        daily = daily.dropna(subset=["value"]).reset_index(drop=True)

    series_map = align_by_dte(daily, dte_max=dte_max)
    # Drop empty/NaN-only series (can happen for tight DTE windows)
    cleaned = {}
    for key, series in series_map.items():
        s = series.dropna()
        if not s.empty:
            cleaned[key] = s
    series_map = cleaned
    if not series_map:
        LOGGER.warning("No data for %s %s %s (dte_max=%s). Skipping.", spread, metric, year, dte_max)
        return

    suffix = f"{spread.lower()}_{metric}_{year}_{cfg.source}_shift{cfg.execution_shift_bdays}"
    overlay_png = outdir / f"{suffix}_overlay.png"
    avg_png = outdir / f"{suffix}_average.png"
    avg_csv = outdir / f"{suffix}_average.csv"

    ylabel = f"{spread} level" if metric == "level" else f"Δ{spread} (daily change)"
    title = f"{spread} by DTE — {year} ({metric}, {cfg.source}, shift={cfg.execution_shift_bdays})"

    plot_overlay_by_dte(series_map, title=title, ylabel=ylabel, output_png=overlay_png, dte_max=dte_max)
    avg_df = plot_average_by_dte(series_map, title=title, ylabel=ylabel, output_png=avg_png)
    avg_df.to_csv(avg_csv)

    LOGGER.info("Wrote %s", overlay_png)
    LOGGER.info("Wrote %s", avg_png)
    LOGGER.info("Wrote %s", avg_csv)


def _run_overall_s1(
    *,
    panel: pd.DataFrame,
    start_year: int,
    end_year: int,
    metric: str,
    cfg: DailySeriesConfig,
    dte_max: int,
    outdir: Path,
) -> None:
    spread = "S1"
    daily = build_daily_spread_series(panel, spread, config=cfg)
    daily = _filter_year_range(daily, start_year, end_year)

    if metric == "diff":
        daily = daily.sort_values(["near_contract", "trade_date"]).copy()
        daily["value"] = daily.groupby("near_contract", sort=False)["value"].diff()
        daily = daily.dropna(subset=["value"]).reset_index(drop=True)

    series_map = align_by_dte(daily, dte_max=dte_max)
    cleaned = {}
    for key, series in series_map.items():
        s = series.dropna()
        if not s.empty:
            cleaned[key] = s
    series_map = cleaned
    if not series_map:
        LOGGER.warning(
            "No data for overall %s %s (%s-%s, dte_max=%s). Skipping.",
            spread,
            metric,
            start_year,
            end_year,
            dte_max,
        )
        return

    suffix = f"s1_{metric}_{start_year}_{end_year}_{cfg.source}_shift{cfg.execution_shift_bdays}_overall"
    avg_png = outdir / f"{suffix}_average.png"
    avg_csv = outdir / f"{suffix}_average.csv"

    ylabel = "S1 level" if metric == "level" else "ΔS1 (daily change)"
    title = f"S1 by DTE — {start_year}-{end_year} overall ({metric}, {cfg.source}, shift={cfg.execution_shift_bdays})"

    avg_df = plot_average_by_dte(series_map, title=title, ylabel=ylabel, output_png=avg_png)
    avg_df.to_csv(avg_csv)

    LOGGER.info("Wrote %s", avg_png)
    LOGGER.info("Wrote %s", avg_csv)


def main() -> int:
    ap = argparse.ArgumentParser(description="Expiry-anchored seasonality plots by DTE.")
    ap.add_argument("--data-dir", default="data_parquet", help="Stage 2 output directory")
    ap.add_argument("--symbol", default="HG", help="Commodity symbol")
    ap.add_argument("--year", type=int, help="Target year (e.g., 2024)")
    ap.add_argument("--start-year", type=int, help="Start year for batch mode")
    ap.add_argument("--end-year", type=int, help="End year for batch mode")
    ap.add_argument("--spread", default="S1", help="Spread name for single-year mode (S1 or S2)")
    ap.add_argument("--spreads", default="S1,S2", help="Comma-separated spreads for batch mode")
    ap.add_argument("--metric", choices=["level", "diff"], default="level")
    ap.add_argument(
        "--source",
        choices=["us_vwap", "bucket1"],
        default="us_vwap",
        help="Daily price source: US-session VWAP or bucket1 close",
    )
    ap.add_argument(
        "--execution-shift",
        type=int,
        default=0,
        help="Shift prices forward by N business days (align next-day execution to signal day)",
    )
    ap.add_argument("--dte-max", type=int, default=30, help="Max DTE to include in per-year plots")
    ap.add_argument("--overall-dte-max", type=int, default=20, help="Max DTE to include in overall S1 averages")
    ap.add_argument("--outdir", default="output/expiry_seasonality", help="Output directory")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel = read_spread_panel(data_dir, args.symbol)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    cfg = DailySeriesConfig(source=args.source, execution_shift_bdays=args.execution_shift)

    if args.start_year is not None or args.end_year is not None:
        if args.start_year is None or args.end_year is None:
            raise SystemExit("Both --start-year and --end-year are required for batch mode.")
        years = list(range(args.start_year, args.end_year + 1))
        spreads = _parse_spreads(args.spreads)
        for year in years:
            year_dir = outdir / f"{year}"
            for spread in spreads:
                spread_dir = year_dir / spread.lower()
                spread_dir.mkdir(parents=True, exist_ok=True)
                for metric in ("level", "diff"):
                    _run_one(
                        panel=panel,
                        year=year,
                        spread=spread,
                        metric=metric,
                        cfg=cfg,
                        dte_max=args.dte_max,
                        outdir=spread_dir,
                    )

        # Overall S1 averages (level + diff) across the full year range
        for metric in ("level", "diff"):
            _run_overall_s1(
                panel=panel,
                start_year=args.start_year,
                end_year=args.end_year,
                metric=metric,
                cfg=cfg,
                dte_max=args.overall_dte_max,
                outdir=outdir,
            )
    else:
        if args.year is None:
            raise SystemExit("--year is required when not running in batch mode.")
        spread = args.spread.upper()
        _run_one(
            panel=panel,
            year=args.year,
            spread=spread,
            metric=args.metric,
            cfg=cfg,
            dte_max=args.dte_max,
            outdir=outdir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
