"""Strategy templates for research (analysis-only)."""

from .base import BaseStrategy
from .carry import CarryClippedStrategy, CarrySignStrategy
from .mean_reversion import MeanReversionStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "carry_sign": CarrySignStrategy,
    "carry_clipped": CarryClippedStrategy,
    "mean_reversion": MeanReversionStrategy,
}

__all__ = [
    "BaseStrategy",
    "CarryClippedStrategy",
    "CarrySignStrategy",
    "MeanReversionStrategy",
    "STRATEGY_REGISTRY",
]
