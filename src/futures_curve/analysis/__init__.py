"""Research analysis utilities (out-of-pipeline).

This package is intentionally separate from the Stage 0–2 pipeline. It provides
lightweight, reproducible research helpers (strategy prototypes, backtests, and
diagnostics) built on top of Stage2 parquet outputs.
"""

from .config import AnalysisConfig

