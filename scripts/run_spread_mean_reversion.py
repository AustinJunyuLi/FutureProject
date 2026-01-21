from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.data import load_spreads_panel, build_daily_spread_series
from futures_curve.analysis.scan import scan_mean_reversion_strategy
from futures_curve.analysis.strategies.mean_reversion import MeanReversionParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan DTE-conditioned mean reversion on HG spreads (S1–S4).")
    p.add_argument("--symbol", default="HG")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--out", default="output/analysis")
    p.add_argument("--max-dd", type=float, default=0.15)
    p.add_argument("--vol-target", type=float, default=0.10)
    p.add_argument("--capital", type=float, default=1_000_000.0)

    # MR params (comma lists)
    p.add_argument("--dte-bin-sizes", default="5")
    p.add_argument("--lookbacks", default="756")
    p.add_argument("--entry-z", default="0.75,1.0,1.25,1.5")
    p.add_argument("--exit-z", default="0.2")
    p.add_argument("--max-hold", default="10,20")
    return p.parse_args()


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
    panel = load_spreads_panel(spreads_path)

    dte_bins = [int(x) for x in args.dte_bin_sizes.split(",") if x.strip()]
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    entry_zs = [float(x) for x in args.entry_z.split(",") if x.strip()]
    exit_zs = [float(x) for x in args.exit_z.split(",") if x.strip()]
    max_holds = [int(x) for x in args.max_hold.split(",") if x.strip()]

    grid: list[MeanReversionParams] = []
    for b in dte_bins:
        for lb in lookbacks:
            for ez in entry_zs:
                for xz in exit_zs:
                    for mh in max_holds:
                        grid.append(
                            MeanReversionParams(
                                dte_bin_size=b,
                                lookback_days=lb,
                                entry_z=ez,
                                exit_z=xz,
                                max_hold_days=mh,
                            )
                        )

    all_results: list[dict[str, object]] = []
    for s in cfg.spreads:
        series = build_daily_spread_series(config=cfg, spreads_panel=panel, spread_num=s)
        results = scan_mean_reversion_strategy(config=cfg, series=series, param_grid=grid)
        all_results.extend([r.__dict__ for r in results])

    if not all_results:
        print("No strategies passed the filters (net_mean>0 and maxDD gate).")
        return

    res_df = pd.DataFrame(all_results).sort_values(["net_sharpe_annual"], ascending=False)
    res_df.to_csv(out_dir / "mean_reversion_results.csv", index=False)

    print(res_df.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()

