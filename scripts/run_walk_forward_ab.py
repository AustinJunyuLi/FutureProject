from __future__ import annotations

import argparse
from pathlib import Path

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

    summary_df = pd.DataFrame(summary_rows).sort_values(["strategy", "spread"])
    summary_df.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    print("\nOOS stitched summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

