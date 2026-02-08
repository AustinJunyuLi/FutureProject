"""Command-line interface for the futures curve pipeline.

Usage:
    futures-curve run --config config/default.yaml --commodities HG
    futures-curve stage1 --symbol HG
    futures-curve stage2 --symbol HG
"""

from pathlib import Path
from typing import Optional, List
import typer
import yaml

app = typer.Typer(
    name="futures-curve",
    help="Futures curve pipeline (Stages 0–2)",
    add_completion=False,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict, symbols: list[str]) -> None:
    """Validate pipeline configuration.

    Checks that required keys are present and that each requested symbol
    has an expiry rule configured.  Raises ``typer.BadParameter`` with a
    clear message on failure.
    """
    # Required top-level keys
    paths = cfg.get("paths")
    if not paths:
        raise typer.BadParameter("Config missing required 'paths' section")
    for key in ("raw_data", "output_parquet", "metadata"):
        if key not in paths:
            raise typer.BadParameter(f"Config 'paths' section missing required key '{key}'")

    # Validate symbols have expiry rules
    from .stage0.expiry_schedule import _load_expiry_rules
    rules = _load_expiry_rules()
    missing = [s for s in symbols if s.upper() not in rules]
    if missing:
        raise typer.BadParameter(
            f"No expiry rule configured for symbol(s): {', '.join(missing)}. "
            f"Add them to config/expiry_rules.yaml. "
            f"Configured symbols: {sorted(rules.keys())}"
        )

    # Validate symbols have directory mapping
    from .stage1.file_scanner import _load_dir_mapping
    dir_map = _load_dir_mapping()
    missing_dirs = [s for s in symbols if s.upper() not in dir_map]
    if missing_dirs:
        raise typer.BadParameter(
            f"No directory mapping for symbol(s): {', '.join(missing_dirs)}. "
            f"Add them to config/commodities.yaml. "
            f"Configured symbols: {sorted(dir_map.keys())}"
        )


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
        help="Comma-separated list of stages to run (e.g., 0,1,2)",
    ),
):
    """Run the pipeline (Stages 0–2)."""
    from .stage0 import build_expiry_table, save_expiry_table, save_contract_specs
    from .stage1 import run_stage1
    from .stage2 import run_stage2

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
        stage_list = [0, 1, 2]

    # Validate config and symbols before starting
    validate_config(cfg, symbols)

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


def main() -> None:
    """Entrypoint for the console script."""
    app()


if __name__ == "__main__":
    main()
