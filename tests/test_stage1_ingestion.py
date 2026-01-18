import pandas as pd

from futures_curve.stage1.file_scanner import FileScanner
from futures_curve.stage1.streaming_reader import StreamingReader
from futures_curve.stage1.bucket_aggregator import BucketAggregator


def test_stage1_parse_and_bucket_assignment_fixture():
    """Parse a small raw 1-min excerpt and validate trade_date + bucket logic.

    This is a deterministic regression test for:
      - raw file schema compatibility
      - contract code parsing from filename
      - CT trade_date cutoff (17:00 CT)
      - bucket mapping for key boundary hours
    """

    root_dir = "tests/fixtures"
    scanner = FileScanner(root_dir=root_dir)
    files = scanner.scan_symbol("HG")

    # Should discover the fixture file under tests/fixtures/copper
    assert len(files) == 1
    meta = files[0]
    assert meta.contract == "HGG22"

    reader = StreamingReader()
    df = reader.read_file(meta.path)

    # Basic schema
    assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert df["timestamp"].isna().sum() == 0

    agg = BucketAggregator()
    df2 = agg.add_bucket_columns(df, raw_timezone="US/Central")

    # Helper to pull a single row by timestamp (naive string in file, localized by code)
    def row(ts: str) -> pd.Series:
        mask = df2["timestamp"].dt.tz_convert("US/Central").dt.strftime("%Y-%m-%d %H:%M:%S") == ts
        out = df2.loc[mask]
        assert len(out) == 1
        return out.iloc[0]

    r = row("2020-12-16 16:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-16").date()
    assert r["bucket"] == 0

    r = row("2020-12-16 17:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 8

    r = row("2020-12-16 21:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 9

    r = row("2020-12-17 02:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 9

    r = row("2020-12-17 03:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 10

    r = row("2020-12-17 09:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 1

    r = row("2020-12-17 15:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 7

    r = row("2020-12-17 16:30:00")
    assert r["trade_date"] == pd.Timestamp("2020-12-17").date()
    assert r["bucket"] == 0
