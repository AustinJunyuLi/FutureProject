"""Stage 4: Strategy backtesting."""

from .execution import (
    ExecutionEngine,
    ExecutionConfig,
    Trade,
    get_execution_price,
)
from .strategies import (
    Strategy,
    Signal,
    DTEStrategy,
    PreExpiryStrategy,
    LiquidityTriggerStrategy,
    HybridStrategy,
    get_strategy,
)
from .performance import (
    PerformanceAnalyzer,
    print_performance_report,
)
from .backtester import (
    Backtester,
    run_backtest,
)
from .pipeline import (
    Stage4Pipeline,
    run_stage4,
)

__all__ = [
    "ExecutionEngine",
    "ExecutionConfig",
    "Trade",
    "get_execution_price",
    "Strategy",
    "Signal",
    "DTEStrategy",
    "PreExpiryStrategy",
    "LiquidityTriggerStrategy",
    "HybridStrategy",
    "get_strategy",
    "PerformanceAnalyzer",
    "print_performance_report",
    "Backtester",
    "run_backtest",
    "Stage4Pipeline",
    "run_stage4",
]
