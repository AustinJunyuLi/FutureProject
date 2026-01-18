"""Trading strategies for spread backtesting.

Implements various entry/exit rules:
- DTE Rule: Entry at L business days before roll peak
- Liquidity Trigger: Enter when volume share crosses threshold
- Hybrid: DTE window + liquidity trigger
- EOM Baseline: Long S1 entry EOM-3, exit EOM-1
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
    """Entry based on days-to-expiry before roll peak.

    Long spread at DTE_entry days before expected roll peak.
    Exit at DTE_exit or roll completion.
    """

    def __init__(
        self,
        dte_entry: int = 15,
        dte_exit: int = 5,
    ):
        """Initialize DTE strategy.

        Args:
            dte_entry: Entry at this many DTE
            dte_exit: Exit at this many DTE
        """
        super().__init__("DTE_Rule")
        self.dte_entry = dte_entry
        self.dte_exit = dte_exit

    def generate_signals(
        self,
        data: pd.DataFrame,
        dte_col: str = "F1_dte_bdays",
        **kwargs,
    ) -> List[Signal]:
        """Generate signals based on DTE.

        Args:
            data: DataFrame with DTE column
            dte_col: Name of DTE column

        Returns:
            List of entry/exit signals
        """
        signals = []
        # Chronological ordering (prefer ts_end_utc for bucket panels)
        if "ts_end_utc" in data.columns:
            data = data.sort_values("ts_end_utc").reset_index(drop=True)
        else:
            data = data.sort_values(["trade_date", "bucket"]).reset_index(drop=True)

        in_position = False
        entry_signal = None

        for idx, row in data.iterrows():
            dte = row.get(dte_col)
            if pd.isna(dte):
                continue

            trade_date = row["trade_date"]
            bucket = row["bucket"]

            if not in_position:
                # Check for entry
                if dte <= self.dte_entry and dte > self.dte_exit:
                    signals.append(Signal(
                        date=trade_date,
                        bucket=bucket,
                        direction=1,  # Long spread
                        reason=f"DTE={dte} <= {self.dte_entry}",
                        metadata={"dte": dte, "action": "entry"},
                    ))
                    in_position = True
            else:
                # Check for exit
                if dte <= self.dte_exit:
                    signals.append(Signal(
                        date=trade_date,
                        bucket=bucket,
                        direction=0,  # Exit
                        reason=f"DTE={dte} <= {self.dte_exit}",
                        metadata={"dte": dte, "action": "exit"},
                    ))
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


class EOMStrategy(Strategy):
    """End-of-month spread strategy.

    Design intent
    -------------
    The research narrative specifies "Enter EOM-3, exit EOM-1" *with next-bucket
    execution* (no same-bucket fills).

    Given next-bucket execution, achieving an executed entry on EOM-N requires
    generating the **signal** on EOM-(N+delay).

    Example (default delay=1 business day):
        - desired entry execution on EOM-3  -> signal on EOM-4
        - desired exit execution on  EOM-1  -> signal on EOM-2

    This keeps the executed window aligned to the documented offsets while
    remaining causal.
    """

    def __init__(
        self,
        entry_offset: int = 3,
        exit_offset: int = 1,
        execution_delay_bdays: int = 1,
    ):
        """Initialize EOM strategy.

        Args:
            entry_offset: Desired executed entry at EOM-N (business days before month end)
            exit_offset: Desired executed exit at EOM-N
            execution_delay_bdays: Execution delay in business days (default 1 = next observation)
        """
        super().__init__("EOM_Baseline")
        self.entry_offset = int(entry_offset)
        self.exit_offset = int(exit_offset)
        self.execution_delay_bdays = int(execution_delay_bdays)
        if self.execution_delay_bdays < 1:
            raise ValueError("execution_delay_bdays must be >= 1 to avoid same-bucket execution")

    def generate_signals(
        self,
        data: pd.DataFrame,
        eom_offset_col: str = "eom_offset",
        **kwargs,
    ) -> List[Signal]:
        """Generate signals based on EOM offset.

        Args:
            data: DataFrame with EOM offset column
            eom_offset_col: EOM offset column name

        Returns:
            List of signals
        """
        signals: List[Signal] = []
        data = data.sort_values(["trade_date"]).reset_index(drop=True)

        in_position = False

        # Convert desired *executed* offsets into *signal* offsets under next-bucket execution.
        entry_signal_offset = self.entry_offset + self.execution_delay_bdays
        exit_signal_offset = self.exit_offset + self.execution_delay_bdays

        for idx, row in data.iterrows():
            offset = row.get(eom_offset_col)
            if pd.isna(offset):
                continue

            trade_date = row["trade_date"]
            bucket = row.get("bucket", 1)

            if not in_position:
                # Signal on EOM-(entry_offset + delay) so execution occurs at EOM-entry_offset.
                if int(offset) == int(entry_signal_offset):
                    signals.append(
                        Signal(
                            date=trade_date,
                            bucket=bucket,
                            direction=1,
                            reason=f"EOM-signal-for-{self.entry_offset}",
                            metadata={
                                "eom_offset": int(offset),
                                "target_eom_offset": int(self.entry_offset),
                                "action": "entry",
                                "execution": "next",
                            },
                        )
                    )
                    in_position = True
            else:
                # Signal on EOM-(exit_offset + delay) so execution occurs at EOM-exit_offset.
                if int(offset) == int(exit_signal_offset):
                    signals.append(
                        Signal(
                            date=trade_date,
                            bucket=bucket,
                            direction=0,
                            reason=f"EOM-signal-for-{self.exit_offset}",
                            metadata={
                                "eom_offset": int(offset),
                                "target_eom_offset": int(self.exit_offset),
                                "action": "exit",
                                "execution": "next",
                            },
                        )
                    )
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
        "dte": DTEStrategy,
        "liquidity": LiquidityTriggerStrategy,
        "hybrid": HybridStrategy,
        "eom": EOMStrategy,
    }

    name_lower = name.lower()
    if name_lower not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name_lower](**kwargs)
