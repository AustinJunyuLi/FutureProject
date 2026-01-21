from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_curve.analysis.config import AnalysisConfig
from futures_curve.analysis.data import SpreadDailySeries, build_daily_spread_series, load_spreads_panel
from futures_curve.analysis.roll import RollFilterConfig, load_roll_shares_daily
from futures_curve.analysis.strategies.mean_reversion import MeanReversionParams
from futures_curve.analysis.walk_forward import WalkForwardOutput, walk_forward_carry, walk_forward_mean_reversion
from futures_curve.stage0.contract_specs import get_contract_spec


@dataclass(frozen=True)
class ScenarioOutput:
    roll_filter: str
    carry_by_spread: dict[str, WalkForwardOutput]
    mr_by_spread: dict[str, WalkForwardOutput]
    best_of_carry_df: pd.DataFrame
    best_of_carry_equity: pd.Series
    best_of_carry_sharpe: float
    best_of_carry_max_dd: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate figures + LaTeX table snippets for HG spread strategy search report."
    )
    p.add_argument("--symbol", default="HG")
    p.add_argument("--out-root", default="output/report/hg_spread_strategy_search")
    p.add_argument("--first-test-year", type=int, default=2013)
    p.add_argument("--last-test-year", type=int, default=2024)

    # Evaluation constraints (match scripts/run_walk_forward_ab.py defaults)
    p.add_argument("--max-dd", type=float, default=0.15)
    p.add_argument("--vol-target", type=float, default=0.10)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--min-train-nonzero-days", type=int, default=120)
    p.add_argument("--min-test-nonzero-days", type=int, default=10)

    # Roll filter scenarios (comma list among: none,exclude_roll,pre_roll_only,post_roll_only)
    p.add_argument("--roll-filters", default="none,exclude_roll")
    p.add_argument("--roll-start", type=float, default=0.25)
    p.add_argument("--roll-end", type=float, default=0.75)
    p.add_argument("--roll-min-volume", type=float, default=1.0)

    # Regime filters (keep simple: avoid full cross-product)
    p.add_argument(
        "--regimes",
        default="all,contango_only,backwardation_only,high_vol_only,low_vol_only",
        help="Comma list among: all,contango_only,backwardation_only,high_vol_only,low_vol_only",
    )

    # Carry search space
    p.add_argument("--carry-thresholds", default="0.02,0.05,0.08")
    p.add_argument("--carry-scales", default="0.05,0.10,0.20")

    # Mean reversion search space
    p.add_argument("--mr-dte-bin-sizes", default="5")
    p.add_argument("--mr-lookbacks", default="504,756")
    p.add_argument("--mr-entry-z", default="1.0,1.25,1.5,1.75,2.0")
    p.add_argument("--mr-exit-z", default="0.2,0.3")
    p.add_argument("--mr-max-hold", default="10,20")
    return p.parse_args()


def _regime_pairs(regimes: list[str]) -> list[tuple[str, str]]:
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

    out: list[tuple[str, str]] = []
    seen = set()
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _year_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(date(year, 1, 1)), pd.Timestamp(date(year, 12, 31))


def _format_years(years: list[int]) -> str:
    if not years:
        return "--"
    years = sorted(set(int(y) for y in years))
    return f"{years[0]}--{years[-1]} ({len(years)})"


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{float(x):.3f}"


def _params_short(params: dict[str, object]) -> str:
    typ = str(params.get("type", ""))
    if typ == "sign":
        thr = params.get("threshold")
        direction = int(params.get("direction", 0))
        return f"sign thr={thr:.2f} dir={direction:+d}"
    if typ == "clipped":
        thr = params.get("threshold")
        scale = params.get("scale")
        direction = int(params.get("direction", 0))
        return f"clip thr={thr:.2f} sc={scale:.2f} dir={direction:+d}"
    # MR params
    direction = int(params.get("direction", 0))
    lb = params.get("lookback_days")
    b = params.get("dte_bin_size")
    ez = params.get("entry_z")
    xz = params.get("exit_z")
    mh = params.get("max_hold_days")
    parts = [f"mr b={b}", f"lb={lb}", f"e={ez}", f"x={xz}", f"mh={mh}", f"dir={direction:+d}"]
    return " ".join(str(p) for p in parts if p is not None)


def _filters_short(filters: dict[str, object]) -> str:
    c = str(filters.get("contango_mode", "all"))
    v = str(filters.get("vol_mode", "all"))
    cont = {"all": "all", "contango_only": "cont", "backwardation_only": "bwd"}.get(c, c)
    vol = {"all": "all", "high_vol_only": "hiVol", "low_vol_only": "loVol"}.get(v, v)
    return f"{cont},{vol}"


def _write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _latex_escape(text: str) -> str:
    # Minimal escaping for table cells.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _make_oos_summary_table(
    *,
    roll_filter: str,
    carry_by_spread: dict[str, WalkForwardOutput],
    mr_by_spread: dict[str, WalkForwardOutput],
    out_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for spread, out in carry_by_spread.items():
        rows.append(
            {
                "strategy": "carry",
                "spread": spread,
                "years": _format_years(out.oos_years),
                "sharpe": out.oos_sharpe,
                "max_dd": out.oos_max_dd,
                "nonzero": out.oos_nonzero_days,
                "n_days": int(len(out.oos_returns)),
            }
        )
    for spread, out in mr_by_spread.items():
        rows.append(
            {
                "strategy": "mean reversion",
                "spread": spread,
                "years": _format_years(out.oos_years),
                "sharpe": out.oos_sharpe,
                "max_dd": out.oos_max_dd,
                "nonzero": out.oos_nonzero_days,
                "n_days": int(len(out.oos_returns)),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["strategy", "spread"])

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{Stitched out-of-sample summary by spread (roll filter = { _latex_escape(roll_filter) }).}}")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("Strategy & Spread & Years & Sharpe & MaxDD & Nonzero days & OOS days \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(r["strategy"])),
                    _latex_escape(str(r["spread"])),
                    _latex_escape(str(r["years"])),
                    _fmt(float(r["sharpe"])) if pd.notna(r["sharpe"]) else "--",
                    _fmt(float(r["max_dd"])) if pd.notna(r["max_dd"]) else "--",
                    str(int(r["nonzero"])),
                    str(int(r["n_days"])),
                ]
            )
            + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    _write_tex(out_path, "\n".join(lines) + "\n")


def _plot_equity_curves(
    *,
    title: str,
    series_by_name: dict[str, pd.Series],
    initial_capital: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 4.5))
    for name, eq in series_by_name.items():
        if eq is None or eq.empty:
            continue
        y = (eq / float(initial_capital)).astype("float64")
        plt.plot(y.index, y.values, label=name, linewidth=1.5)
    plt.title(title)
    plt.ylabel("Equity (normalized)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_execution_bucket_distribution(
    *,
    daily_series: list[SpreadDailySeries],
    out_path: Path,
) -> None:
    buckets = list(range(1, 8))
    counts = pd.DataFrame(index=[s.spread for s in daily_series], columns=buckets, data=0.0)
    for s in daily_series:
        vc = s.df["exec_bucket"].value_counts().reindex(buckets).fillna(0.0)
        counts.loc[s.spread, buckets] = vc.values

    frac = counts.div(counts.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    plt.figure(figsize=(9, 4.2))
    bottom = np.zeros(len(frac))
    x = np.arange(len(frac.index))
    for b in buckets:
        vals = frac[b].to_numpy(dtype="float64")
        plt.bar(x, vals, bottom=bottom, label=f"{b}")
        bottom = bottom + vals
    plt.xticks(x, frac.index.tolist())
    plt.xlabel("Spread")
    plt.ylabel("Fraction of exec days")
    plt.title("Execution bucket distribution (earliest available US bucket in 1–7)")
    plt.legend(title="Bucket", ncol=7, fontsize=8)
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _best_of_carry(
    *,
    carry_by_spread: dict[str, WalkForwardOutput],
    first_test_year: int,
    last_test_year: int,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.Series, float, float]:
    best_rows: list[dict[str, object]] = []
    pnl_pieces: list[pd.Series] = []

    for year in range(first_test_year, last_test_year + 1):
        best_choice = None
        best_train = None
        for spread, out in carry_by_spread.items():
            folds = [f for f in out.folds if f.fold_year == year]
            if not folds:
                continue
            f = folds[0]
            if best_train is None or (pd.notna(f.train_sharpe) and f.train_sharpe > best_train):
                best_train = f.train_sharpe
                best_choice = (spread, out, f)

        if best_choice is None:
            continue

        spread, out, f = best_choice
        year_start, year_end = _year_window(year)
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

    best_df = pd.DataFrame(best_rows).sort_values(["fold_year"]) if best_rows else pd.DataFrame()
    pnl_all = pd.concat(pnl_pieces).sort_index() if pnl_pieces else pd.Series(dtype="float64")

    if pnl_all.empty:
        return best_df, pd.Series(dtype="float64"), float("nan"), float("nan")

    equity = pnl_all.cumsum().add(float(initial_capital))
    returns = (pnl_all / equity.shift(1).replace(0.0, np.nan)).fillna(0.0)
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252)) if returns.std(ddof=0) != 0 else float("nan")
    dd = (equity.cummax() - equity) / equity.cummax()
    max_dd = float(dd.fillna(0.0).max())
    return best_df, equity, sharpe, max_dd


def _make_best_of_table(
    *,
    roll_filter: str,
    best_df: pd.DataFrame,
    stitched_sharpe: float,
    stitched_max_dd: float,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    cap = (
        "Best-of carry (select spread by highest train Sharpe each year). "
        f"Stitched OOS Sharpe={_fmt(stitched_sharpe)}, MaxDD={_fmt(stitched_max_dd)} "
        f"(roll filter = {_latex_escape(roll_filter)})."
    )
    lines.append(f"\\caption{{{cap}}}")
    lines.append("\\begin{tabular}{llrrrrp{0.42\\textwidth}}")
    lines.append("\\toprule")
    lines.append("Year & Spread & Train Sh & Test Sh & Test MaxDD & Test days & Rule (params; filters) \\\\")
    lines.append("\\midrule")

    if best_df.empty:
        lines.append("\\multicolumn{7}{c}{No eligible folds under current gating.} \\\\")
    else:
        for _, r in best_df.iterrows():
            params = r.get("params", {})
            filters = r.get("filters", {})
            rule = f"{_params_short(params)}; {_filters_short(filters)}"
            lines.append(
                " & ".join(
                    [
                        str(int(r["fold_year"])),
                        _latex_escape(str(r["selected_spread"])),
                        _fmt(float(r["train_sharpe"])) if pd.notna(r["train_sharpe"]) else "--",
                        _fmt(float(r["test_sharpe"])) if pd.notna(r["test_sharpe"]) else "--",
                        _fmt(float(r["test_max_dd"])) if pd.notna(r["test_max_dd"]) else "--",
                        str(int(r["test_nonzero_days"])),
                        _latex_escape(rule),
                    ]
                )
                + " \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    _write_tex(out_path, "\n".join(lines) + "\n")


def _run_scenario(
    *,
    config: AnalysisConfig,
    daily_series: list[SpreadDailySeries],
    roll_daily: pd.DataFrame,
    roll_cfg: RollFilterConfig,
    roll_filter: str,
    regime_pairs: list[tuple[str, str]],
    first_test_year: int,
    last_test_year: int,
    carry_thresholds: list[float],
    carry_scales: list[float],
    mr_grid: list[MeanReversionParams],
) -> ScenarioOutput:
    carry_by_spread: dict[str, WalkForwardOutput] = {}
    mr_by_spread: dict[str, WalkForwardOutput] = {}

    for series in daily_series:
        carry_by_spread[series.spread] = walk_forward_carry(
            config=config,
            series=series,
            roll_daily=roll_daily,
            roll_filter_mode=roll_filter,
            roll_cfg=roll_cfg,
            contango_vol_filter_pairs=regime_pairs,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            thresholds=carry_thresholds,
            scales=carry_scales,
        )
        mr_by_spread[series.spread] = walk_forward_mean_reversion(
            config=config,
            series=series,
            roll_daily=roll_daily,
            roll_filter_mode=roll_filter,
            roll_cfg=roll_cfg,
            contango_vol_filter_pairs=regime_pairs,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            param_grid=mr_grid,
        )

    best_df, best_equity, best_sharpe, best_max_dd = _best_of_carry(
        carry_by_spread=carry_by_spread,
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        initial_capital=config.initial_capital,
    )

    return ScenarioOutput(
        roll_filter=roll_filter,
        carry_by_spread=carry_by_spread,
        mr_by_spread=mr_by_spread,
        best_of_carry_df=best_df,
        best_of_carry_equity=best_equity,
        best_of_carry_sharpe=best_sharpe,
        best_of_carry_max_dd=best_max_dd,
    )


def main() -> None:
    args = _parse_args()

    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    cfg = AnalysisConfig(
        symbol=args.symbol,
        spreads=(1, 2, 3, 4),
        max_drawdown_limit=args.max_dd,
        vol_target_annual=args.vol_target,
        initial_capital=args.capital,
        min_train_nonzero_days=args.min_train_nonzero_days,
        min_test_nonzero_days=args.min_test_nonzero_days,
    )

    spreads_path = Path("data_parquet") / "spreads" / args.symbol / "spreads_panel.parquet"
    roll_path = Path("data_parquet") / "roll_events" / args.symbol / "roll_shares.parquet"

    panel = load_spreads_panel(spreads_path)
    roll_daily = load_roll_shares_daily(roll_path)

    # Build daily series upfront (also used for exec bucket distribution).
    daily_series: list[SpreadDailySeries] = []
    for s in cfg.spreads:
        daily_series.append(build_daily_spread_series(config=cfg, spreads_panel=panel, spread_num=s))

    _plot_execution_bucket_distribution(daily_series=daily_series, out_path=fig_dir / "execution_bucket_distribution.png")

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

    roll_filters = [x.strip() for x in args.roll_filters.split(",") if x.strip()]
    scenarios: list[ScenarioOutput] = []

    for rf in roll_filters:
        print(f"\n=== Running scenario: roll_filter={rf} ===")
        scenarios.append(
            _run_scenario(
                config=cfg,
                daily_series=daily_series,
                roll_daily=roll_daily,
                roll_cfg=roll_cfg,
                roll_filter=rf,
                regime_pairs=regime_pairs,
                first_test_year=args.first_test_year,
                last_test_year=args.last_test_year,
                carry_thresholds=carry_thresholds,
                carry_scales=carry_scales,
                mr_grid=mr_grid,
            )
        )

    # Write tables + figures per scenario.
    for sc in scenarios:
        rf = sc.roll_filter
        suffix = "none" if rf == "none" else rf

        _make_oos_summary_table(
            roll_filter=rf,
            carry_by_spread=sc.carry_by_spread,
            mr_by_spread=sc.mr_by_spread,
            out_path=tab_dir / f"oos_summary_{suffix}.tex",
        )

        _make_best_of_table(
            roll_filter=rf,
            best_df=sc.best_of_carry_df,
            stitched_sharpe=sc.best_of_carry_sharpe,
            stitched_max_dd=sc.best_of_carry_max_dd,
            out_path=tab_dir / f"best_of_carry_{suffix}.tex",
        )

        _plot_equity_curves(
            title=f"Carry family — stitched OOS equity (roll_filter={rf})",
            series_by_name={k: v.oos_equity for k, v in sc.carry_by_spread.items()},
            initial_capital=cfg.initial_capital,
            out_path=fig_dir / f"oos_equity_carry_{suffix}.png",
        )
        _plot_equity_curves(
            title=f"Mean reversion family — stitched OOS equity (roll_filter={rf})",
            series_by_name={k: v.oos_equity for k, v in sc.mr_by_spread.items()},
            initial_capital=cfg.initial_capital,
            out_path=fig_dir / f"oos_equity_mean_reversion_{suffix}.png",
        )

        if sc.best_of_carry_equity is not None and not sc.best_of_carry_equity.empty:
            _plot_equity_curves(
                title=f"Best-of carry — stitched OOS equity (roll_filter={rf})",
                series_by_name={"best_of": sc.best_of_carry_equity},
                initial_capital=cfg.initial_capital,
                out_path=fig_dir / f"oos_equity_best_of_carry_{suffix}.png",
            )

    # Write a tiny metadata note (for traceability from the report).
    spec = get_contract_spec(cfg.symbol)
    if spec is not None:
        meta = (
            "HG contract spec (used for dollars/tick and contract size):\n"
            f"- contract_size={spec.contract_size}\n"
            f"- tick_size={spec.tick_size}\n"
            f"- point_value={spec.point_value}\n"
        )
        (out_root / "meta.txt").write_text(meta, encoding="utf-8")

    print(f"\nWrote report assets under: {out_root}")


if __name__ == "__main__":
    main()
