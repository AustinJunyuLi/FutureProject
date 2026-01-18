"""Stage 3: Analysis layer - seasonality, lifecycle, diagnostics."""

from .eom_seasonality import (
    EOMSeasonality,
    build_eom_daily_dataset,
)
from .lifecycle_analysis import (
    LifecycleAnalyzer,
    build_lifecycle_dataset,
)
from .diagnostics import (
    DataDiagnostics,
    SpreadDiagnostics,
    run_full_diagnostics,
)
from .pipeline import (
    Stage3Pipeline,
    run_stage3,
)

__all__ = [
    "EOMSeasonality",
    "build_eom_daily_dataset",
    "LifecycleAnalyzer",
    "build_lifecycle_dataset",
    "DataDiagnostics",
    "SpreadDiagnostics",
    "run_full_diagnostics",
    "Stage3Pipeline",
    "run_stage3",
]
