from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date

import pandas as pd

from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.data import load_spreads_panel, build_daily_spread_series
from futures_curve.analysis.roll import RollFilterConfig, load_roll_shares_daily
from futures_curve.analysis.strategies.mean_reversion import MeanReversionParams
from futures_curve.analysis.walk_forward import walk_forward_carry, walk_forward_mean_reversion


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward OOS test for carry + DTE MR on HG spreads (S1–S4).")
    p.add_argument("--symbol", default="HG")
    p.add_argument("--out", default="output/analysis")
    p.add_argument("--first-test-year", type=int, default=2013)
    p.add_argument("--last-test-year", type=int, default=2024)

    p.add_argument("--max-dd", type=float, default=0.15)
    p.add_argument("--vol-target", type=float, default=0.10)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--min-train-nonzero-days", type=int, default=252)
    p.add_argument("--min-test-nonzero-days", type=int, default=20)

    # Roll / liquidity filter (applied on signal day)
    p.add_argument(
        "--roll-filter",
        default="exclude_roll",
        choices=["none", "exclude_roll", "pre_roll_only", "post_roll_only"],
    )
    p.add_argument("--roll-start", type=float, default=0.25)
    p.add_argument("--roll-end", type=float, default=0.75)
    p.add_argument("--roll-min-volume", type=float, default=1.0)

    # Regime filters (keep simple: 5 combinations, not full cross-product)
    p.add_argument(
        "--regimes",
        default="all,contango_only,backwardation_only,high_vol_only,low_vol_only",
        help="Comma list among: all,contango_only,backwardation_only,high_vol_only,low_vol_only",
    )

    # Carry search space (widened but controlled)
    p.add_argument("--carry-thresholds", default="0.02,0.05,0.08")
    p.add_argument("--carry-scales", default="0.05,0.10,0.20")

    # MR search space (widened but controlled)
    p.add_argument("--mr-dte-bin-sizes", default="5")
    p.add_argument("--mr-lookbacks", default="504,756")
    p.add_argument("--mr-entry-z", default="1.0,1.25,1.5,1.75,2.0")
    p.add_argument("--mr-exit-z", default="0.2,0.3")
    p.add_argument("--mr-max-hold", default="10,20")
    return p.parse_args()


def _regime_pairs(regimes: list[str]) -> list[tuple[str, str]]:
    # Translate a single list into (contango_mode, vol_mode) pairs while avoiding a full cross-product.
    pairs: list[tuple[str, str]] = [("all", "all")]
    for r in regimes:
        r = r.strip()
        if not r or r == "all":
            continue
        if r in {"contango_only", "backwardation_only"}:
            pairs.append((r, "all"))
        elif r in {"high_vol_only", "low_vol_only"}:
            pairs.append(("all", r))
        else:
            raise ValueError(f"Unknown regime: {r}")
    # Deduplicate while preserving order
    out: list[tuple[str, str]] = []
    seen = set()
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AnalysisConfig(
        symbol=args.symbol,
        max_drawdown_limit=args.max_dd,
        vol_target_annual=args.vol_target,
        initial_capital=args.capital,
        spreads=(1, 2, 3, 4),
        min_train_nonzero_days=args.min_train_nonzero_days,
        min_test_nonzero_days=args.min_test_nonzero_days,
    )

    spreads_path = Path("data_parquet") / "spreads" / args.symbol / "spreads_panel.parquet"
    roll_path = Path("data_parquet") / "roll_events" / args.symbol / "roll_shares.parquet"

    panel = load_spreads_panel(spreads_path)
    roll_daily = load_roll_shares_daily(roll_path)

    roll_cfg = RollFilterConfig(
        start_threshold=args.roll_start,
        end_threshold=args.roll_end,
        min_total_volume=args.roll_min_volume,
        use_smooth=True,
    )

    regime_pairs = _regime_pairs(args.regimes.split(","))

    carry_thresholds = [float(x) for x in args.carry_thresholds.split(",") if x.strip()]
    carry_scales = [float(x) for x in args.carry_scales.split(",") if x.strip()]

    dte_bins = [int(x) for x in args.mr_dte_bin_sizes.split(",") if x.strip()]
    lookbacks = [int(x) for x in args.mr_lookbacks.split(",") if x.strip()]
    entry_zs = [float(x) for x in args.mr_entry_z.split(",") if x.strip()]
    exit_zs = [float(x) for x in args.mr_exit_z.split(",") if x.strip()]
    max_holds = [int(x) for x in args.mr_max_hold.split(",") if x.strip()]

    mr_grid: list[MeanReversionParams] = []
    for b in dte_bins:
        for lb in lookbacks:
            for ez in entry_zs:
                for xz in exit_zs:
                    for mh in max_holds:
                        mr_grid.append(
                            MeanReversionParams(
                                dte_bin_size=b,
                                lookback_days=lb,
                                entry_z=ez,
                                exit_z=xz,
                                max_hold_days=mh,
                            )
                        )

    carry_fold_rows: list[dict[str, object]] = []
    mr_fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    carry_by_spread: dict[str, object] = {}

    for s in cfg.spreads:
        series = build_daily_spread_series(config=cfg, spreads_panel=panel, spread_num=s)

        carry_out = walk_forward_carry(
            config=cfg,
            series=series,
            roll_daily=roll_daily,
            roll_filter_mode=args.roll_filter,
            roll_cfg=roll_cfg,
            contango_vol_filter_pairs=regime_pairs,
            first_test_year=args.first_test_year,
            last_test_year=args.last_test_year,
            thresholds=carry_thresholds,
            scales=carry_scales,
        )
        carry_fold_rows.extend([r.__dict__ for r in carry_out.folds])
        carry_by_spread[series.spread] = carry_out
        summary_rows.append(
            {
                "strategy": "carry",
                "spread": series.spread,
                "roll_filter": args.roll_filter,
                "oos_years": ",".join(str(y) for y in carry_out.oos_years),
                "oos_sharpe": carry_out.oos_sharpe,
                "oos_max_dd": carry_out.oos_max_dd,
                "oos_nonzero_days": carry_out.oos_nonzero_days,
                "oos_n_days": int(len(carry_out.oos_returns)),
            }
        )

        mr_out = walk_forward_mean_reversion(
            config=cfg,
            series=series,
            roll_daily=roll_daily,
            roll_filter_mode=args.roll_filter,
            roll_cfg=roll_cfg,
            contango_vol_filter_pairs=regime_pairs,
            first_test_year=args.first_test_year,
            last_test_year=args.last_test_year,
            param_grid=mr_grid,
        )
        mr_fold_rows.extend([r.__dict__ for r in mr_out.folds])
        summary_rows.append(
            {
                "strategy": "mean_reversion",
                "spread": series.spread,
                "roll_filter": args.roll_filter,
                "oos_years": ",".join(str(y) for y in mr_out.oos_years),
                "oos_sharpe": mr_out.oos_sharpe,
                "oos_max_dd": mr_out.oos_max_dd,
                "oos_nonzero_days": mr_out.oos_nonzero_days,
                "oos_n_days": int(len(mr_out.oos_returns)),
            }
        )

    if carry_fold_rows:
        carry_df = pd.DataFrame(carry_fold_rows).sort_values(["spread", "fold_year"])
        carry_df.to_csv(out_dir / "walk_forward_carry.csv", index=False)
        print("\nCarry folds (head):")
        print(carry_df.head(20).to_string(index=False))

    if mr_fold_rows:
        mr_df = pd.DataFrame(mr_fold_rows).sort_values(["spread", "fold_year"])
        mr_df.to_csv(out_dir / "walk_forward_mean_reversion.csv", index=False)
        print("\nMean reversion folds (head):")
        print(mr_df.head(20).to_string(index=False))

    # Best-of carry portfolio: select the best spread each year by TRAIN Sharpe only.
    best_rows: list[dict[str, object]] = []
    pnl_pieces: list[pd.Series] = []
    for year in range(args.first_test_year, args.last_test_year + 1):
        best_choice = None
        best_train = None
        for spread, out in carry_by_spread.items():
            folds = [f for f in out.folds if f.fold_year == year]
            if not folds:
                continue
            # One fold per year per spread by construction
            f = folds[0]
            if best_train is None or (pd.notna(f.train_sharpe) and f.train_sharpe > best_train):
                best_train = f.train_sharpe
                best_choice = (spread, out, f)

        if best_choice is None:
            continue

        spread, out, f = best_choice
        year_start = pd.Timestamp(date(year, 1, 1))
        year_end = pd.Timestamp(date(year, 12, 31))
        pnl_year = out.oos_pnl.loc[(out.oos_pnl.index >= year_start) & (out.oos_pnl.index <= year_end)]
        if pnl_year.empty:
            continue
        pnl_pieces.append(pnl_year)
        best_rows.append(
            {
                "fold_year": year,
                "selected_spread": spread,
                "train_sharpe": f.train_sharpe,
                "train_max_dd": f.train_max_dd,
                "train_nonzero_days": f.train_nonzero_days,
                "test_sharpe": f.test_sharpe,
                "test_max_dd": f.test_max_dd,
                "test_nonzero_days": f.test_nonzero_days,
                "params": f.params,
                "filters": f.filters,
            }
        )

    if best_rows and pnl_pieces:
        best_df = pd.DataFrame(best_rows).sort_values(["fold_year"])
        best_df.to_csv(out_dir / "walk_forward_best_of_carry.csv", index=False)
        pnl_all = pd.concat(pnl_pieces).sort_index()
        # Convert to returns/equity for reporting.
        equity = pnl_all.cumsum().add(cfg.initial_capital)
        rets = (pnl_all / equity.shift(1).replace(0.0, pd.NA)).fillna(0.0)
        sharpe = float((rets.mean() / rets.std(ddof=0)) * (cfg.trading_days_per_year ** 0.5)) if rets.std(ddof=0) != 0 else float("nan")
        dd = (equity.cummax() - equity) / equity.cummax()
        max_dd = float(dd.fillna(0.0).max())
        print("\nBest-of carry (selected spread per year):")
        print(best_df.to_string(index=False))
        print(f"\nBest-of carry stitched OOS: sharpe={sharpe:.3f}, maxDD={max_dd:.3f}, years={best_df['fold_year'].min()}–{best_df['fold_year'].max()}, nonzero_days={(pnl_all!=0).sum()}")
    else:
        print("\nBest-of carry: no years produced eligible folds.")

    summary_df = pd.DataFrame(summary_rows).sort_values(["strategy", "spread"])
    summary_df.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    print("\nOOS stitched summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
