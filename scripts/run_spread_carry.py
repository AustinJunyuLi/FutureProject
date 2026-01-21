from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.data import load_spreads_panel, build_daily_spread_series
from futures_curve.analysis.scan import scan_carry_strategy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan carry strategy on HG spreads (S1–S4).")
    p.add_argument("--symbol", default="HG")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--out", default="output/analysis")
    p.add_argument("--thresholds", default="0,0.01,0.02,0.03,0.05")
    p.add_argument("--max-dd", type=float, default=0.15)
    p.add_argument("--vol-target", type=float, default=0.10)
    p.add_argument("--capital", type=float, default=1_000_000.0)
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

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    all_results: list[dict[str, object]] = []
    for s in cfg.spreads:
        series = build_daily_spread_series(config=cfg, spreads_panel=panel, spread_num=s)
        results = scan_carry_strategy(config=cfg, series=series, thresholds=thresholds)
        all_results.extend([r.__dict__ for r in results])

    if not all_results:
        print("No strategies passed the filters (net_mean>0 and maxDD gate).")
        return

    res_df = pd.DataFrame(all_results).sort_values(["net_sharpe_annual"], ascending=False)
    res_df.to_csv(out_dir / "carry_results.csv", index=False)

    print(res_df.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()

