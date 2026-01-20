"""Command-line interface for the futures curve pipeline.

Usage:
    futures-curve run --config config/default.yaml --commodities HG
    futures-curve stage1 --symbol HG
    futures-curve stage2 --symbol HG
    futures-curve stage3 --symbol HG
    futures-curve stage4 --symbol HG
"""

from pathlib import Path
from typing import Optional, List
import typer
import yaml

app = typer.Typer(
    name="futures-curve",
    help="Futures curve analysis pipeline",
    add_completion=False,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.command()
def run(
    config: str = typer.Option(
        "config/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    commodities: Optional[str] = typer.Option(
        None,
        "--commodities",
        help="Comma-separated list of commodities (e.g., HG,GC)",
    ),
    stages: Optional[str] = typer.Option(
        None,
        "--stages",
        help="Comma-separated list of stages to run (e.g., 1,2,3,4)",
    ),
):
    """Run the complete pipeline."""
    from .stage0 import build_expiry_table, save_expiry_table, save_contract_specs
    from .stage1 import run_stage1
    from .stage2 import run_stage2
    from .stage3 import run_stage3
    from .stage4 import run_stage4

    # Load config
    cfg = load_config(config)

    # Get commodities
    if commodities:
        symbols = [s.strip() for s in commodities.split(",")]
    else:
        symbols = cfg.get("commodities", ["HG"])

    # Get stages
    if stages:
        stage_list = [int(s.strip()) for s in stages.split(",")]
    else:
        stage_list = [0, 1, 2, 3, 4]

    raw_data_dir = cfg["paths"]["raw_data"]
    output_dir = cfg["paths"]["output_parquet"]

    typer.echo(f"Running pipeline for: {', '.join(symbols)}")
    typer.echo(f"Stages: {stage_list}")
    typer.echo(f"Output: {output_dir}")

    # Stage 0: Metadata
    if 0 in stage_list:
        typer.echo("\n" + "=" * 60)
        typer.echo("Stage 0: Building metadata")
        typer.echo("=" * 60)

        metadata_dir = Path(cfg["paths"]["metadata"])
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for symbol in symbols:
            typer.echo(f"Building expiry table for {symbol}...")
            expiry_df = build_expiry_table(symbol, 2008, 2025)
            save_expiry_table(expiry_df, str(metadata_dir / f"{symbol}_expiry.parquet"))

        save_contract_specs(str(metadata_dir / "contract_specs.json"))

    # Stage 1: Ingestion
    if 1 in stage_list:
        typer.echo("\n" + "=" * 60)
        typer.echo("Stage 1: Ingestion and bucket aggregation")
        typer.echo("=" * 60)

        run_stage1(
            raw_data_dir,
            output_dir,
            symbols,
            chunk_size=cfg.get("ingestion", {}).get("chunk_size", 100000),
        )

    # Stage 2: Curve construction
    if 2 in stage_list:
        typer.echo("\n" + "=" * 60)
        typer.echo("Stage 2: Curve construction and spreads")
        typer.echo("=" * 60)

        run_stage2(
            output_dir,
            symbols,
            zscore_window=cfg.get("curve", {}).get("zscore_window", 20),
            roll_threshold=cfg.get("roll", {}).get("start_threshold", 0.25),
        )

    # Stage 3: Analysis
    if 3 in stage_list:
        typer.echo("\n" + "=" * 60)
        typer.echo("Stage 3: Analysis")
        typer.echo("=" * 60)

        run_stage3(
            output_dir,
            symbols,
            research_dir=cfg.get("paths", {}).get("research_outputs"),
        )

    # Stage 4: Backtesting
    if 4 in stage_list:
        typer.echo("\n" + "=" * 60)
        typer.echo("Stage 4: Backtesting")
        typer.echo("=" * 60)

        run_stage4(
            output_dir,
            symbols,
            execution_config=cfg.get("backtest", {}),
        )

    typer.echo("\nPipeline complete!")


@app.command()
def stage0(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    output_dir: str = typer.Option("metadata", "--output", "-o", help="Output directory"),
    start_year: int = typer.Option(2008, help="Start year"),
    end_year: int = typer.Option(2025, help="End year"),
):
    """Run Stage 0: Build metadata (expiry tables, calendars)."""
    from .stage0 import build_expiry_table, save_expiry_table, build_trading_calendar, save_trading_calendar

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Building metadata for {symbol}")

    # Expiry table
    typer.echo("  Building expiry table...")
    expiry_df = build_expiry_table(symbol, start_year, end_year)
    save_expiry_table(expiry_df, str(output_path / f"{symbol}_expiry.parquet"))

    # Trading calendar
    typer.echo("  Building trading calendar...")
    save_trading_calendar(
        "CMEGlobex_Metals",
        str(output_path / "trading_calendar.parquet"),
        start_year=start_year,
        end_year=end_year,
    )

    typer.echo("Done!")


@app.command()
def stage1(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    raw_data_dir: str = typer.Option(
        "/home/austinli/futures_data/organized_data",
        "--input", "-i",
        help="Raw data directory",
    ),
    output_dir: str = typer.Option(
        "data_parquet",
        "--output", "-o",
        help="Output directory",
    ),
):
    """Run Stage 1: Ingest raw data and create bucket aggregates."""
    from .stage1 import run_stage1

    typer.echo(f"Running Stage 1 for {symbol}")
    results = run_stage1(raw_data_dir, output_dir, [symbol])

    stats = results.get(symbol, {})
    if stats.get("status") == "success":
        typer.echo(f"\nProcessed {stats.get('contracts_processed', 0)} contracts")
        typer.echo(f"Total rows: {stats.get('total_minute_rows', 0):,}")
        typer.echo(f"Bucket rows: {stats.get('bucket_rows', 0):,}")


@app.command()
def stage2(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    data_dir: str = typer.Option("data_parquet", "--data", "-d", help="Data directory"),
):
    """Run Stage 2: Build curves, spreads, and detect rolls."""
    from .stage2 import run_stage2

    typer.echo(f"Running Stage 2 for {symbol}")
    results = run_stage2(data_dir, [symbol])

    stats = results.get(symbol, {})
    if stats.get("status") == "success":
        typer.echo(f"\nCurve records: {stats.get('curve_records', 0):,}")
        typer.echo(f"Spread observations: {stats.get('spread_observations', 0):,}")
        typer.echo(f"Roll events: {stats.get('roll_events', 0)}")


@app.command()
def stage3(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    data_dir: str = typer.Option("data_parquet", "--data", "-d", help="Data directory"),
):
    """Run Stage 3: Analysis (seasonality, lifecycle, diagnostics)."""
    from .stage3 import run_stage3

    typer.echo(f"Running Stage 3 for {symbol}")
    results = run_stage3(data_dir, [symbol])

    stats = results.get(symbol, {})
    lifecycle = stats.get("lifecycle", {}) or {}
    diagnostics = stats.get("diagnostics", {}) or {}

    if lifecycle:
        typer.echo(f"\nLifecycle:")
        if "dte_bins" in lifecycle:
            typer.echo(f"  DTE bins: {lifecycle.get('dte_bins')}")
        if "roll_events_studied" in lifecycle:
            typer.echo(f"  Roll events studied: {lifecycle.get('roll_events_studied')}")

    if diagnostics:
        typer.echo(f"\nDiagnostics:")
        for k in ["ohlc_issues_count", "expiry_violations_count", "spread_discrepancies_count", "data_gaps_count"]:
            if k in diagnostics:
                typer.echo(f"  {k}: {diagnostics.get(k)}")


@app.command()
def stage4(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    data_dir: str = typer.Option("data_parquet", "--data", "-d", help="Data directory"),
    strategies: str = typer.Option("pre_expiry", help="Comma-separated strategies"),
    frequency: str = typer.Option(
        "bucket",
        "--frequency",
        "-f",
        help="Data frequency: 'bucket' or 'daily_us_vwap' (daily US-session VWAP proxy).",
    ),
    daily_pair_policy: str = typer.Option(
        "mode",
        "--daily-pair-policy",
        help="Daily aggregation contract-pair policy: 'mode', 'strict', or 'all'.",
    ),
):
    """Run Stage 4: Backtesting."""
    from .stage4 import run_stage4

    strategy_list = [s.strip() for s in strategies.split(",")]

    typer.echo(f"Running Stage 4 for {symbol}")
    typer.echo(f"Strategies: {', '.join(strategy_list)}")

    # Use defaults from config/default.yaml if present
    cfg = load_config("config/default.yaml") if Path("config/default.yaml").exists() else {}
    daily_agg_config = {"pair_policy": daily_pair_policy} if str(frequency).lower() in {"daily_us_vwap", "us_session_daily_vwap"} else None
    results = run_stage4(
        data_dir,
        [symbol],
        strategies=strategy_list,
        execution_config=cfg.get("backtest", {}),
        data_frequency=frequency,
        daily_agg_config=daily_agg_config,
    )


@app.command()
def pre_expiry_sweep(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    data_dir: str = typer.Option("data_parquet", "--data", "-d", help="Data directory"),
    entry_dte_min: int = typer.Option(2, help="Minimum entry DTE (bdays to expiry)"),
    entry_dte_max: int = typer.Option(10, help="Maximum entry DTE (bdays to expiry)"),
    exit_dte_min: int = typer.Option(0, help="Minimum exit DTE (bdays to expiry)"),
    exit_dte_max: int = typer.Option(3, help="Maximum exit DTE (bdays to expiry)"),
    min_trades: int = typer.Option(1, help="Minimum trades required to keep a parameter row"),
    frequency: str = typer.Option(
        "bucket",
        "--frequency",
        "-f",
        help="Data frequency: 'bucket' or 'daily_us_vwap' (daily US-session VWAP proxy).",
    ),
    daily_pair_policy: str = typer.Option(
        "mode",
        "--daily-pair-policy",
        help="Daily aggregation contract-pair policy: 'mode', 'strict', or 'all'.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print progress for each parameter combo"),
):
    """Search for the best-performing pre-expiry (DTE) entry/exit window.

    This sweeps entry/exit DTE thresholds and ranks combinations by net P&L.
    Results are written to `data_parquet/trades/{symbol}/pre_expiry_sweep_<frequency>.parquet`.
    """
    from dataclasses import fields

    from .stage4.execution import ExecutionConfig
    from .stage4.pipeline import Stage4Pipeline

    cfg = load_config("config/default.yaml") if Path("config/default.yaml").exists() else {}
    backtest_cfg = (cfg.get("backtest") or {}) if isinstance(cfg, dict) else {}

    allowed = {f.name for f in fields(ExecutionConfig)}
    exec_cfg = {k: v for k, v in backtest_cfg.items() if k in allowed}
    risk_keys = {"stop_loss_usd", "max_holding_bdays", "auto_roll_on_contract_change", "allow_same_bucket_execution"}
    risk_cfg = {k: v for k, v in backtest_cfg.items() if k in risk_keys}

    pipeline = Stage4Pipeline(data_dir, config=ExecutionConfig(**exec_cfg), risk_config=risk_cfg)
    daily_agg_config = {"pair_policy": daily_pair_policy} if str(frequency).lower() in {"daily_us_vwap", "us_session_daily_vwap"} else None
    df = pipeline.run_pre_expiry_sweep(
        symbol,
        entry_dte_min=entry_dte_min,
        entry_dte_max=entry_dte_max,
        exit_dte_min=exit_dte_min,
        exit_dte_max=exit_dte_max,
        min_trades=min_trades,
        data_frequency=frequency,
        daily_agg_config=daily_agg_config,
        verbose=verbose,
    )

    if df is None or df.empty:
        typer.echo("No parameter combinations produced any trades (or none met min_trades).")
    else:
        best = df.iloc[0]
        typer.echo(
            f"Best: entry_dte={int(best['entry_dte'])}, exit_dte={int(best['exit_dte'])} "
            f"| trades={int(best['total_trades'])} | net P&L=${float(best['total_pnl']):,.2f}"
        )


@app.command()
def info(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    raw_data_dir: str = typer.Option(
        "/home/austinli/futures_data/organized_data",
        "--input", "-i",
        help="Raw data directory",
    ),
):
    """Show information about available data."""
    from .stage1 import FileScanner

    scanner = FileScanner(raw_data_dir)
    stats = scanner.get_file_stats(symbol)

    typer.echo(f"\nData for {symbol}:")
    typer.echo(f"  Files: {stats.get('file_count', 0)}")
    typer.echo(f"  Size: {stats.get('total_size_mb', 0):.1f} MB")

    year_range = stats.get("year_range")
    if year_range:
        typer.echo(f"  Years: {year_range[0]} - {year_range[1]}")

    contracts = stats.get("contracts", [])
    if contracts:
        typer.echo(f"  Contracts: {len(contracts)}")
        typer.echo(f"  First: {contracts[0]}, Last: {contracts[-1]}")


@app.command()
def report(
    symbol: str = typer.Option("HG", "--symbol", "-s", help="Commodity symbol"),
    data_dir: str = typer.Option("data_parquet", "--data", "-d", help="Data directory"),
    research_dir: str = typer.Option("research_outputs", "--out", "-o", help="Research outputs directory"),
    report_type: str = typer.Option(
        "both",
        "--report-type", "-t",
        help="Report type: 'technical', 'analysis', or 'both'",
    ),
):
    """Build the LaTeX/PDF pipeline report(s) for a symbol.

    Report types:
    - technical: Technical Implementation Report (explains the codebase)
    - analysis: Analysis Results Report (presents findings with visualizations)
    - both: Generate both reports (default)
    """
    from .reporting import build_pipeline_report

    result = build_pipeline_report(
        symbol,
        data_dir=data_dir,
        research_dir=research_dir,
        compile_pdf=True,
        report_type=report_type,
    )

    if report_type == "both":
        tech_path, analysis_path = result
        typer.echo(f"Wrote Technical Report: {tech_path}")
        typer.echo(f"Wrote Analysis Report: {analysis_path}")
    else:
        typer.echo(f"Wrote report: {result}")


def main() -> None:
    """Entrypoint for the console script."""
    app()


if __name__ == "__main__":
    main()
