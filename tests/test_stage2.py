"""Tests for Stage 2: Curve construction and spreads."""

import pytest
from datetime import date
import pandas as pd
import numpy as np

from futures_curve.stage2.contract_ranker import ContractRanker
from futures_curve.stage2.spread_calculator import SpreadCalculator, build_spread_panel
from futures_curve.stage2.roll_detector import RollDetector, RollShareConfig


class TestContractRanker:
    """Tests for contract ranking."""

    def test_rank_contracts_basic(self):
        """Test basic contract ranking by expiry."""
        ranker = ContractRanker()

        contracts = ["HGF24", "HGG24", "HGH24", "HGJ24"]
        as_of = date(2024, 1, 1)

        ranks = ranker.rank_contracts(contracts, as_of)

        # F24 (Jan) should be F1, G24 (Feb) should be F2, etc.
        assert ranks.get("HGF24") == 1
        assert ranks.get("HGG24") == 2
        assert ranks.get("HGH24") == 3

    def test_rank_excludes_expired(self):
        """Test that expired contracts are excluded."""
        ranker = ContractRanker()

        contracts = ["HGF24", "HGG24", "HGH24"]
        # After Feb 2024, F24 and G24 should be expired
        as_of = date(2024, 3, 1)

        ranks = ranker.rank_contracts(contracts, as_of)

        # F24 and G24 should be excluded (expired)
        assert "HGF24" not in ranks
        assert "HGG24" not in ranks
        # H24 should be F1
        assert ranks.get("HGH24") == 1

    def test_rank_max_contracts(self):
        """Test max contracts limit."""
        ranker = ContractRanker(max_contracts=3)

        contracts = ["HGF24", "HGG24", "HGH24", "HGJ24", "HGK24"]
        as_of = date(2024, 1, 1)

        ranks = ranker.rank_contracts(contracts, as_of)

        # Only first 3 should be ranked
        assert len(ranks) == 3
        assert ranks.get("HGJ24") is None or ranks.get("HGJ24") > 3


class TestSpreadCalculator:
    """Tests for spread calculation."""

    def test_spread_calculation(self):
        """Test basic spread calculation."""
        calc = SpreadCalculator()

        # Create mock curve panel
        df = pd.DataFrame({
            "trade_date": [date(2024, 1, 15)],
            "bucket": [1],
            "F1_price": [3.50],
            "F2_price": [3.52],
            "F3_price": [3.55],
        })

        result = calc.calculate_spreads(df, include_zscore=False)

        # S1 = F2 - F1 = 3.52 - 3.50 = 0.02
        assert abs(result["S1"].iloc[0] - 0.02) < 1e-6

        # S2 = F3 - F2 = 3.55 - 3.52 = 0.03
        assert abs(result["S2"].iloc[0] - 0.03) < 1e-6

    def test_spread_positive_contango(self):
        """Test spread is positive in contango (upward sloping curve)."""
        calc = SpreadCalculator()

        # Contango: F2 > F1
        df = pd.DataFrame({
            "trade_date": [date(2024, 1, 15)],
            "bucket": [1],
            "F1_price": [100.0],
            "F2_price": [101.0],
        })

        result = calc.calculate_spreads(df, include_zscore=False)

        assert result["S1"].iloc[0] > 0  # Contango = positive spread

    def test_spread_negative_backwardation(self):
        """Test spread is negative in backwardation (downward sloping curve)."""
        calc = SpreadCalculator()

        # Backwardation: F2 < F1
        df = pd.DataFrame({
            "trade_date": [date(2024, 1, 15)],
            "bucket": [1],
            "F1_price": [101.0],
            "F2_price": [100.0],
        })

        result = calc.calculate_spreads(df, include_zscore=False)

        assert result["S1"].iloc[0] < 0  # Backwardation = negative spread


class TestRollDetector:
    """Tests for roll event detection."""

    def test_volume_share_calculation(self):
        """Test volume share calculation from curve-panel volumes."""
        cfg = RollShareConfig(smoothing_window=1, min_total_volume=0)
        detector = RollDetector(config=cfg)

        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 1, 15)],
                "bucket": [1],
                "ts_end_utc": [pd.Timestamp("2024-01-15 15:59:59", tz="UTC")],
                "F1_contract": ["HGF24"],
                "F2_contract": ["HGG24"],
                "F1_volume": [1000.0],
                "F2_volume": [250.0],
            }
        )

        panel = detector.build_roll_share_panel(df, frequency="bucket")

        # s(t) = V2 / (V1 + V2) = 250 / 1250 = 0.2
        assert abs(panel["volume_share"].iloc[0] - 0.2) < 1e-6
        assert abs(panel["volume_share_smooth"].iloc[0] - 0.2) < 1e-6

    def test_roll_detection_threshold_cross(self):
        """Test roll event detection with persistence (causal confirmation)."""
        cfg = RollShareConfig(
            start_threshold=0.30,
            peak_threshold=0.50,
            end_threshold=0.75,
            persistence=2,
            smoothing_window=1,
            min_total_volume=0,
            strict_gt=True,
        )
        detector = RollDetector(config=cfg)

        # Share crosses >0.30 and persists for 2 observations.
        # Confirmation should happen on the 2nd consecutive point above threshold.
        df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 13)]
                ),
                "bucket": [7, 7, 7, 7],
                "ts_end_utc": pd.to_datetime(
                    [
                        "2024-01-10 21:59:59+00:00",
                        "2024-01-11 21:59:59+00:00",
                        "2024-01-12 21:59:59+00:00",
                        "2024-01-13 21:59:59+00:00",
                    ]
                ),
                "F1_contract": ["HGF24"] * 4,
                "F2_contract": ["HGG24"] * 4,
                "F1_volume": [100, 100, 100, 100],
                "F2_volume": [10, 60, 70, 10],  # share: ~0.09, 0.375, 0.412, ~0.09
            }
        )

        panel = detector.build_roll_share_panel(df, frequency="bucket")
        events = detector.detect_roll_events(panel)

        assert len(events) == 1
        ev = events.iloc[0]
        # First crossing run is on 2024-01-11 and 2024-01-12, confirm at 2024-01-12.
        assert ev["roll_start_ts_utc"] == pd.Timestamp("2024-01-12 21:59:59+00:00")
        assert ev["roll_start_run_start_ts_utc"] == pd.Timestamp("2024-01-11 21:59:59+00:00")

    def test_no_roll_below_threshold(self):
        """Test no roll detected when below threshold."""
        cfg = RollShareConfig(
            start_threshold=0.50,
            peak_threshold=0.60,
            end_threshold=0.75,
            persistence=1,
            smoothing_window=1,
            min_total_volume=0,
        )
        detector = RollDetector(config=cfg)

        df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(pd.date_range("2024-01-10", periods=5)),
                "bucket": [7] * 5,
                "ts_end_utc": pd.to_datetime(
                    [f"2024-01-{10+i:02d} 21:59:59+00:00" for i in range(5)]
                ),
                "F1_contract": ["HGF24"] * 5,
                "F2_contract": ["HGG24"] * 5,
                "F1_volume": [100] * 5,
                "F2_volume": [10, 20, 30, 20, 10],  # share never > 0.5
            }
        )
        panel = detector.build_roll_share_panel(df, frequency="bucket")
        events = detector.detect_roll_events(panel)
        # We emit one row per F1 contract cycle; with no crossings, event timestamps are NaT.
        assert len(events) == 1
        ev = events.iloc[0]
        assert pd.isna(ev["roll_start_ts_utc"])


class TestExpiryOrdering:
    """Tests for expiry ordering validation."""

    def test_expiry_ordering_maintained(self):
        """Test that F1 <= F2 <= ... <= F12 expiries always hold."""
        ranker = ContractRanker()

        # Create contracts for full year
        contracts = [f"HG{m}{y}" for y in ["24", "25"] for m in "FGHJKMNQUVXZ"]
        as_of = date(2024, 1, 1)

        ranks = ranker.rank_contracts(contracts, as_of)

        # Get expiries for ranked contracts
        expiries = {}
        for contract, rank in ranks.items():
            expiries[rank] = ranker.get_expiry_ts_utc(contract)

        # Check ordering
        for i in range(1, len(expiries)):
            if i in expiries and (i + 1) in expiries:
                assert expiries[i] <= expiries[i + 1], \
                    f"Expiry ordering violated: F{i}={expiries[i]} > F{i+1}={expiries[i+1]}"
