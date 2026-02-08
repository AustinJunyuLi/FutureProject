from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarrySignParams:
    """Parameters for sign-based carry strategy."""

    threshold: float = 0.02


@dataclass(frozen=True)
class CarryClippedParams:
    """Parameters for clipped continuous carry strategy."""

    threshold: float = 0.02
    scale: float = 0.10


@dataclass(frozen=True)
class FilterConfig:
    """Regime filter configuration for walk-forward search."""

    contango_mode: str = "all"
    vol_mode: str = "all"
