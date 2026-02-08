from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..data import SpreadDailySeries


class BaseStrategy(ABC):
    """Abstract base for all spread trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (e.g. 'carry_sign')."""

    @abstractmethod
    def generate_signal(
        self, series: SpreadDailySeries, params: object
    ) -> pd.Series:
        """Return a daily signal Series (index=trade_date).

        The signal encodes direction and magnitude in [-1, 1].
        """

    @abstractmethod
    def default_param_grid(self) -> list[object]:
        """Return the default parameter grid for search."""

    @abstractmethod
    def positions_from_signal(
        self, signal: pd.Series, params: object
    ) -> pd.Series:
        """Convert signal to discrete positions.

        For strategies where signal == position (e.g. carry), this is identity.
        For strategies with hysteresis (e.g. MR), this applies state logic.
        """

    @abstractmethod
    def fold_params_dict(self, params: object, direction: int) -> dict[str, object]:
        """Convert params + direction into a dict for FoldResult.params."""
