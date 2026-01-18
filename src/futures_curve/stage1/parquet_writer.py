"""Parquet IO helpers.

The pipeline uses Parquet as its primary on-disk contract between stages.

Implementation note
-------------------
Parquet support in pandas requires an optional engine (`pyarrow` or
`fastparquet`). The original repository did not declare those dependencies,
which means a fresh environment may fail at import time.

To improve robustness (especially for unit tests that do not actually read/write
parquet), this module defers hard failures until an IO method is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:  # Optional dependency
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pa = None
    pq = None


def _require_parquet_engine() -> None:
    """Raise a helpful error if no Parquet engine is available."""
    # pandas supports pyarrow or fastparquet; this project uses pyarrow for
    # partitioned datasets, so we prefer it.
    if pa is None or pq is None:
        raise ImportError(
            "Parquet support requires an optional dependency. "
            "Install `pyarrow` (recommended) or `fastparquet` to read/write parquet files."
        )


class ParquetWriter:
    """Write DataFrames to Parquet files."""

    def __init__(self, base_dir: str | Path):
        """Initialize writer.

        Args:
            base_dir: Base directory for output
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_bucket_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        partition_by: Optional[list[str]] = None,
    ) -> Path:
        """Write bucket data to a partitioned Parquet dataset."""
        _require_parquet_engine()

        output_dir = self.base_dir / "buckets" / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add year column for partitioning if not present
        if "year" not in df.columns and "trade_date" in df.columns:
            df = df.copy()
            df["year"] = pd.to_datetime(df["trade_date"]).dt.year

        if partition_by is None:
            # Default partitioning for resumable per-contract ingestion
            # (avoids rewriting entire year partitions for every contract).
            partition_by = ["contract", "year"]

        table = pa.Table.from_pandas(df)

        pq.write_to_dataset(
            table,
            root_path=str(output_dir),
            partition_cols=partition_by,
            existing_data_behavior="delete_matching",
        )

        return output_dir

    def write_daily_data(self, df: pd.DataFrame, symbol: str) -> Path:
        """Write daily OHLCV data to Parquet."""
        # pandas will pick up any available engine.
        output_dir = self.base_dir / "daily" / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{symbol}_daily.parquet"
        df.to_parquet(output_path, index=False)

        return output_path

    def write_qc_report(self, stats: dict, symbol: str) -> Path:
        """Write QC statistics report."""
        output_dir = self.base_dir / "qc"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{symbol}_qc.parquet"
        df = pd.DataFrame([stats])
        df.to_parquet(output_path, index=False)

        return output_path

    def append_to_parquet(self, df: pd.DataFrame, output_path: Path) -> None:
        """Append data to an existing Parquet file (read/concat/write)."""
        if output_path.exists():
            existing = pd.read_parquet(output_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(output_path, index=False)


def read_bucket_data(
    base_dir: str | Path,
    symbol: str,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Read bucket data from a partitioned Parquet dataset."""
    base_dir = Path(base_dir)
    bucket_dir = base_dir / "buckets" / symbol

    if not bucket_dir.exists():
        raise FileNotFoundError(f"Bucket data not found: {bucket_dir}")

    # This call requires a parquet engine.
    try:
        if years is not None:
            filters = [("year", "in", years)]
            return pd.read_parquet(bucket_dir, filters=filters)
        return pd.read_parquet(bucket_dir)
    except ImportError as e:
        raise ImportError(
            "Reading parquet requires an optional engine. Install `pyarrow` or `fastparquet`."
        ) from e


def read_daily_data(base_dir: str | Path, symbol: str) -> pd.DataFrame:
    """Read daily data from Parquet."""
    base_dir = Path(base_dir)
    daily_path = base_dir / "daily" / symbol / f"{symbol}_daily.parquet"

    if not daily_path.exists():
        raise FileNotFoundError(f"Daily data not found: {daily_path}")

    try:
        return pd.read_parquet(daily_path)
    except ImportError as e:
        raise ImportError(
            "Reading parquet requires an optional engine. Install `pyarrow` or `fastparquet`."
        ) from e
