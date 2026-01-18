"""Stage 1: Streaming ingestion and 10-bucket aggregation."""

from .file_scanner import FileScanner, DataFile, discover_all_commodities
from .streaming_reader import (
    StreamingReader,
    load_contract_data,
    stream_all_contracts,
)
from .bucket_aggregator import (
    BucketAggregator,
    process_contract_to_buckets,
    build_bucket_panel,
)
from .parquet_writer import (
    ParquetWriter,
    read_bucket_data,
    read_daily_data,
)
from .pipeline import Stage1Pipeline, run_stage1

__all__ = [
    "FileScanner",
    "DataFile",
    "discover_all_commodities",
    "StreamingReader",
    "load_contract_data",
    "stream_all_contracts",
    "BucketAggregator",
    "process_contract_to_buckets",
    "build_bucket_panel",
    "ParquetWriter",
    "read_bucket_data",
    "read_daily_data",
    "Stage1Pipeline",
    "run_stage1",
]
