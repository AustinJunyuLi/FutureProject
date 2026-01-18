"""Streaming CSV reader for minute-level futures data.

Provides memory-efficient chunked reading of large CSV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional
import pandas as pd
import numpy as np

from .file_scanner import DataFile


# Column names for raw data (timestamp, O, H, L, C, V)
RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
RAW_DTYPES = {
    "timestamp": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}


class StreamingReader:
    """Memory-efficient reader for minute-level CSV data."""

    def __init__(self, chunk_size: int = 100000):
        """Initialize reader.

        Args:
            chunk_size: Number of rows per chunk
        """
        self.chunk_size = chunk_size

    def read_file(
        self,
        path: str | Path,
    ) -> pd.DataFrame:
        """Read entire file into DataFrame.

        Args:
            path: File path
            parse_dates: Parse timestamp column as datetime

        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(path, names=RAW_COLUMNS, dtype=RAW_DTYPES)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        return df.dropna(subset=["timestamp"])

    def iter_chunks(
        self,
        path: str | Path,
    ) -> Iterator[pd.DataFrame]:
        """Iterate over file in chunks.

        Args:
            path: File path
            parse_dates: Parse timestamp column as datetime

        Yields:
            DataFrame chunks
        """
        reader = pd.read_csv(path, names=RAW_COLUMNS, dtype=RAW_DTYPES, chunksize=self.chunk_size)

        for chunk in reader:
            chunk["timestamp"] = pd.to_datetime(
                chunk["timestamp"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            chunk = chunk.dropna(subset=["timestamp"])
            yield chunk

    def read_data_file(self, data_file: DataFile) -> pd.DataFrame:
        """Read a DataFile object.

        Args:
            data_file: DataFile with path and metadata

        Returns:
            DataFrame with contract column added
        """
        df = self.read_file(data_file.path)
        df["contract"] = data_file.contract
        df["symbol"] = data_file.symbol
        return df

    def validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data integrity.

        Checks:
        - high >= low
        - high >= open and close
        - low <= open and close

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with validation column added
        """
        df = df.copy()

        # OHLC integrity checks
        valid_hl = df["high"] >= df["low"]
        valid_ho = df["high"] >= df["open"]
        valid_hc = df["high"] >= df["close"]
        valid_lo = df["low"] <= df["open"]
        valid_lc = df["low"] <= df["close"]

        df["ohlc_valid"] = valid_hl & valid_ho & valid_hc & valid_lo & valid_lc

        return df

    def get_file_stats(self, path: str | Path) -> dict:
        """Get statistics for a file without loading fully.

        Args:
            path: File path

        Returns:
            Dictionary with file statistics
        """
        path = Path(path)

        # Count rows efficiently
        row_count = 0
        min_ts = None
        max_ts = None

        for chunk in self.iter_chunks(path):
            row_count += len(chunk)

            chunk_min = chunk["timestamp"].min()
            chunk_max = chunk["timestamp"].max()

            if min_ts is None or chunk_min < min_ts:
                min_ts = chunk_min
            if max_ts is None or chunk_max > max_ts:
                max_ts = chunk_max

        return {
            "path": str(path),
            "row_count": row_count,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }


def load_contract_data(
    data_file: DataFile,
    chunk_size: int = 100000,
) -> pd.DataFrame:
    """Load and validate contract data.

    Args:
        data_file: DataFile object
        chunk_size: Chunk size for reading

    Returns:
        Validated DataFrame
    """
    reader = StreamingReader(chunk_size=chunk_size)
    df = reader.read_data_file(data_file)
    df = reader.validate_ohlc(df)
    return df


def stream_all_contracts(
    data_files: list[DataFile],
    chunk_size: int = 100000,
) -> Iterator[pd.DataFrame]:
    """Stream data from multiple contracts.

    Args:
        data_files: List of DataFile objects
        chunk_size: Chunk size for reading

    Yields:
        DataFrame chunks with contract metadata
    """
    reader = StreamingReader(chunk_size=chunk_size)

    for data_file in data_files:
        for chunk in reader.iter_chunks(data_file.path):
            chunk["contract"] = data_file.contract
            chunk["symbol"] = data_file.symbol
            yield chunk
