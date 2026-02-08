"""Stage 1 Pipeline: Ingestion and bucket aggregation.

Processes raw minute CSV files into partitioned Parquet bucket data.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from .file_scanner import FileScanner, DataFile
from .streaming_reader import StreamingReader, load_contract_data
from .bucket_aggregator import BucketAggregator, process_contract_to_buckets
from .parquet_writer import ParquetWriter
from .timezone_inference import infer_raw_timezone_from_files


class Stage1Pipeline:
    """Complete Stage 1 processing pipeline."""

    def __init__(
        self,
        raw_data_dir: str | Path,
        output_dir: str | Path,
        chunk_size: int = 100000,
    ):
        """Initialize pipeline.

        Args:
            raw_data_dir: Root directory with raw CSV data
            output_dir: Output directory for Parquet files
            chunk_size: Chunk size for streaming reads
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size

        self.scanner = FileScanner(self.raw_data_dir)
        self.reader = StreamingReader(chunk_size=chunk_size)
        self.aggregator = BucketAggregator()
        self.writer = ParquetWriter(self.output_dir)

    def process_symbol(
        self,
        symbol: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """Process all contracts for a symbol.

        Args:
            symbol: Commodity symbol (e.g., "HG")
            start_year: Start year filter
            end_year: End year filter
            verbose: Print progress

        Returns:
            Dictionary with processing statistics
        """
        start_time = datetime.now()

        # Discover files
        files = list(self.scanner.iter_files(symbol, start_year, end_year))
        if verbose:
            print(f"Found {len(files)} files for {symbol}")

        if not files:
            return {"symbol": symbol, "status": "no_files"}

        # Infer raw timezone once per symbol from sampled files
        tz_result = infer_raw_timezone_from_files([f.path for f in files])
        raw_tz = tz_result.raw_timezone
        if verbose:
            print(f"Inferred raw timezone for {symbol}: {raw_tz} "
                  f"(maintenance leakage={tz_result.maintenance_leakage_pct.get(raw_tz, float('nan')):.3f}%)")

        # Process each contract
        all_bucket_data = []
        all_daily_data = []
        total_rows = 0
        contracts_processed = 0
        invalid_rows = 0

        for i, data_file in enumerate(files):
            if verbose:
                print(f"  [{i+1}/{len(files)}] Processing {data_file.contract}...")

            try:
                # Load and validate
                df = load_contract_data(data_file, chunk_size=self.chunk_size)
                total_rows += len(df)

                # Count invalid rows
                if "ohlc_valid" in df.columns:
                    invalid_rows += (~df["ohlc_valid"]).sum()

                # Convert to exchange time + assign trade_date/bucket using inferred raw timezone
                df = self.aggregator.add_bucket_columns(df, raw_timezone=raw_tz)

                # Aggregate to buckets
                bucket_df = self.aggregator.aggregate_to_buckets(df)
                all_bucket_data.append(bucket_df)

                # Aggregate to daily
                daily_df = self.aggregator.aggregate_to_daily(df)
                all_daily_data.append(daily_df)

                contracts_processed += 1

            except (ValueError, KeyError, pd.errors.EmptyDataError, OSError) as e:
                print(f"    Error processing {data_file.contract}: {e}")
                continue

        # Combine all data
        if all_bucket_data:
            combined_buckets = pd.concat(all_bucket_data, ignore_index=True)
            combined_buckets = combined_buckets.sort_values(
                ["trade_date", "bucket", "contract"]
            ).reset_index(drop=True)

            # Write bucket data
            self.writer.write_bucket_data(combined_buckets, symbol)

        if all_daily_data:
            combined_daily = pd.concat(all_daily_data, ignore_index=True)
            combined_daily = combined_daily.sort_values(
                ["trade_date", "contract"]
            ).reset_index(drop=True)

            # Write daily data
            self.writer.write_daily_data(combined_daily, symbol)

        # Compute statistics
        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            "symbol": symbol,
            "status": "success",
            "contracts_processed": contracts_processed,
            "total_minute_rows": total_rows,
            "invalid_ohlc_rows": invalid_rows,
            "bucket_rows": len(combined_buckets) if all_bucket_data else 0,
            "daily_rows": len(combined_daily) if all_daily_data else 0,
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_second": round(total_rows / elapsed, 0) if elapsed > 0 else 0,
        }

        # Write QC report
        self.writer.write_qc_report(stats, symbol)

        if verbose:
            print(f"Completed {symbol}: {contracts_processed} contracts, "
                  f"{total_rows:,} rows in {elapsed:.1f}s")

        return stats

    def get_coverage_report(self, symbol: str) -> pd.DataFrame:
        """Get data coverage report for a symbol.

        Args:
            symbol: Commodity symbol

        Returns:
            DataFrame with coverage by contract and date
        """
        files = self.scanner.scan_symbol(symbol)

        records = []
        for f in files:
            stats = self.reader.get_file_stats(f.path)
            records.append({
                "contract": f.contract,
                "year": f.year,
                "month": f.month,
                "row_count": stats["row_count"],
                "min_date": stats["min_timestamp"],
                "max_date": stats["max_timestamp"],
                "file_size_mb": stats["file_size_mb"],
            })

        return pd.DataFrame(records)


def run_stage1(
    raw_data_dir: str,
    output_dir: str,
    symbols: list[str],
    chunk_size: int = 100000,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> dict:
    """Run Stage 1 pipeline for specified symbols.

    Args:
        raw_data_dir: Root directory with raw CSV data
        output_dir: Output directory for Parquet files
        symbols: List of commodity symbols to process
        start_year: Start year filter
        end_year: End year filter

    Returns:
        Dictionary with processing statistics per symbol
    """
    pipeline = Stage1Pipeline(raw_data_dir, output_dir, chunk_size=chunk_size)

    results = {}
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        stats = pipeline.process_symbol(
            symbol,
            start_year=start_year,
            end_year=end_year,
        )
        results[symbol] = stats

    return results
