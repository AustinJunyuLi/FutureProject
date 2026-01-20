"""Trading strategies for spread backtesting.

Implements various entry/exit rules:
- Liquidity Trigger: Enter when volume share crosses threshold
- Hybrid: DTE window + liquidity trigger
- Pre-expiry window: Enter/exit at business-day DTE thresholds
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Trading signal."""

    date: date
    bucket: int
    direction: int  # 1 = long, -1 = short
    strength: float = 1.0
    reason: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str):
        """Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> List[Signal]:
        """Generate trading signals.

        Args:
            data: Market data DataFrame
            **kwargs: Strategy-specific parameters

        Returns:
            List of Signal objects
        """
        pass


class DTEStrategy(Strategy):
    """Backward-compatible alias for :class:`PreExpiryStrategy`."""

    def __init__(
        self,
        dte_entry: int = 15,
        dte_exit: int = 5,
    ):
        super().__init__("DTE_Rule")
        self.strategy = PreExpiryStrategy(entry_dte=dte_entry, exit_dte=dte_exit)

    def generate_signals(
        self,
        data: pd.DataFrame,
        dte_col: str = "F1_dte_bdays",
        **kwargs,
    ) -> List[Signal]:
        return self.strategy.generate_signals(data, dte_col=dte_col, **kwargs)


class PreExpiryStrategy(Strategy):
    """Expiry-anchored spread strategy (pre-expiry window).

    Enter a long S1 spread when F1 has <= `entry_dte` business days remaining
    until expiry, and exit when F1 has <= `exit_dte` business days remaining.

    Notes
    -----
    - This strategy is anchored to contract expiry (via `F1_dte_bdays`), not
      calendar month-end.
    - `F1_dte_bdays` is defined on the trade-date axis (business days).
    """

    def __init__(self, entry_dte: int = 5, exit_dte: int = 1, execution: str = "next"):
        super().__init__("pre_expiry")
        self.entry_dte = int(entry_dte)
        self.exit_dte = int(exit_dte)
        self.execution = str(execution).lower()
        if self.entry_dte <= self.exit_dte:
            raise ValueError("entry_dte must be > exit_dte")
        if self.exit_dte < 0:
            raise ValueError("exit_dte must be >= 0")
        if self.execution not in {"next", "same"}:
            raise ValueError("execution must be 'next' or 'same'")

    def generate_signals(
        self,
        data: pd.DataFrame,
        dte_col: str = "F1_dte_bdays",
        **kwargs,
    ) -> List[Signal]:
        signals: List[Signal] = []

        # Chronological ordering (prefer ts_end_utc for bucket panels)
        if "ts_end_utc" in data.columns:
            data = data.sort_values("ts_end_utc").reset_index(drop=True)
        else:
            data = data.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

        in_position = False

        for _, row in data.iterrows():
            dte = row.get(dte_col)
            if pd.isna(dte):
                continue

            trade_date = row["trade_date"]
            bucket = int(row.get("bucket", 1))

            if not in_position:
                # Enter when within the pre-expiry window (closer to expiry than entry_dte,
                # but not yet at/through the exit threshold).
                if dte <= self.entry_dte and dte > self.exit_dte:
                    signals.append(
                        Signal(
                            date=trade_date,
                            bucket=bucket,
                            direction=1,
                            reason=f"pre_expiry_entry_dte={self.entry_dte}",
                            metadata={
                                "dte": float(dte),
                                "action": "entry",
                                **({"execution": self.execution} if self.execution != "next" else {}),
                            },
                        )
                    )
                    in_position = True
            else:
                if dte <= self.exit_dte:
                    signals.append(
                        Signal(
                            date=trade_date,
                            bucket=bucket,
                            direction=0,
                            reason=f"pre_expiry_exit_dte={self.exit_dte}",
                            metadata={
                                "dte": float(dte),
                                "action": "exit",
                                **({"execution": self.execution} if self.execution != "next" else {}),
                            },
                        )
                    )
                    in_position = False

        return signals


class LiquidityTriggerStrategy(Strategy):
    """Entry when volume share crosses threshold.

    Long spread when s(t) = V2/(V1+V2) first crosses threshold.
    Exit when s(t) drops below exit threshold.
    """

    def __init__(
        self,
        entry_threshold: float = 0.2,
        exit_threshold: float = 0.1,
    ):
        """Initialize liquidity trigger strategy.

        Args:
            entry_threshold: Volume share entry threshold
            exit_threshold: Volume share exit threshold
        """
        super().__init__("Liquidity_Trigger")
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(
        self,
        data: pd.DataFrame,
        volume_share_col: str = "volume_share_smooth",
        **kwargs,
    ) -> List[Signal]:
        """Generate signals based on volume share.

        Args:
            data: DataFrame with volume share column
            volume_share_col: Name of volume share column

        Returns:
            List of entry/exit signals
        """
        signals = []
        if "ts_end_utc" in data.columns:
            data = data.sort_values("ts_end_utc").reset_index(drop=True)
        else:
            data = data.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

        in_position = False
        prev_share = None

        for idx, row in data.iterrows():
            share = row.get(volume_share_col)
            if pd.isna(share):
                prev_share = share
                continue

            trade_date = row["trade_date"]
            bucket = row.get("bucket", 1)

            if not in_position:
                # Check for entry - cross above threshold
                if prev_share is not None and prev_share < self.entry_threshold and share >= self.entry_threshold:
                    signals.append(Signal(
                        date=trade_date,
                        bucket=bucket,
                        direction=1,
                        reason=f"Volume share crossed {self.entry_threshold}",
                        metadata={"volume_share": share, "action": "entry"},
                    ))
                    in_position = True
            else:
                # Check for exit - drop below threshold
                if share < self.exit_threshold:
                    signals.append(Signal(
                        date=trade_date,
                        bucket=bucket,
                        direction=0,
                        reason=f"Volume share dropped below {self.exit_threshold}",
                        metadata={"volume_share": share, "action": "exit"},
                    ))
                    in_position = False

            prev_share = share

        return signals


class HybridStrategy(Strategy):
    """Combines DTE window with liquidity trigger.

    Entry: DTE within window AND volume share crosses threshold
    Exit: DTE below threshold OR volume share drops
    """

    def __init__(
        self,
        dte_max: int = 20,
        dte_min: int = 5,
        volume_threshold: float = 0.15,
    ):
        """Initialize hybrid strategy.

        Args:
            dte_max: Maximum DTE for entry window
            dte_min: Minimum DTE for exit
            volume_threshold: Volume share threshold
        """
        super().__init__("Hybrid")
        self.dte_max = dte_max
        self.dte_min = dte_min
        self.volume_threshold = volume_threshold

    def generate_signals(
        self,
        data: pd.DataFrame,
        dte_col: str = "F1_dte_bdays",
        volume_share_col: str = "volume_share_smooth",
        **kwargs,
    ) -> List[Signal]:
        """Generate signals combining DTE and volume.

        Args:
            data: DataFrame with DTE and volume share
            dte_col: DTE column name
            volume_share_col: Volume share column name

        Returns:
            List of signals
        """
        signals = []
        if "ts_end_utc" in data.columns:
            data = data.sort_values("ts_end_utc").reset_index(drop=True)
        else:
            data = data.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

        in_position = False

        for idx, row in data.iterrows():
            dte = row.get(dte_col)
            vol_share = row.get(volume_share_col)

            if pd.isna(dte) or pd.isna(vol_share):
                continue

            trade_date = row["trade_date"]
            bucket = row.get("bucket", 1)

            if not in_position:
                # Entry: DTE in window AND volume trigger
                in_window = self.dte_min < dte <= self.dte_max
                vol_triggered = vol_share >= self.volume_threshold

                if in_window and vol_triggered:
                    signals.append(Signal(
                        date=trade_date,
                        bucket=bucket,
                        direction=1,
                        reason=f"DTE={dte} in window, vol_share={vol_share:.2f}",
                        metadata={"dte": dte, "volume_share": vol_share, "action": "entry"},
                    ))
                    in_position = True
            else:
                # Exit: DTE below min
                if dte <= self.dte_min:
                    signals.append(Signal(
                        date=trade_date,
                        bucket=bucket,
                        direction=0,
                        reason=f"DTE={dte} <= {self.dte_min}",
                        metadata={"dte": dte, "action": "exit"},
                    ))
                    in_position = False

        return signals


def get_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to get strategy by name.

    Args:
        name: Strategy name
        **kwargs: Strategy parameters

    Returns:
        Strategy instance
    """
    strategies = {
        "liquidity": LiquidityTriggerStrategy,
        "hybrid": HybridStrategy,
        "pre_expiry": PreExpiryStrategy,
        # Backward-compatible alias
        "dte": DTEStrategy,
    }

    name_lower = name.lower()
    if name_lower not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name_lower](**kwargs)
