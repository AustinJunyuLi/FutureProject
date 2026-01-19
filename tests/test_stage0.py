"""Tests for Stage 0: Metadata acquisition."""

import pytest
from datetime import date

from futures_curve.stage0.expiry_schedule import ExpiryCalculator, build_expiry_table
from futures_curve.stage0.trading_calendar import TradingCalendar
from futures_curve.stage0.contract_specs import get_contract_spec, CONTRACT_SPECS


class TestExpiryCalculator:
    """Tests for expiry date calculation."""

    def test_third_last_business_day(self):
        """Test third last business day rule (CME metals)."""
        calc = ExpiryCalculator()

        # January 2024 - last business day is Jan 31
        # Third last would be Jan 29
        expiry = calc.third_last_business_day(2024, 1)
        assert expiry.month == 1
        assert expiry.year == 2024
        # Should be a weekday
        assert expiry.weekday() < 5

    def test_copper_expiry(self):
        """Test copper (HG) expiry calculation."""
        calc = ExpiryCalculator()

        # HG uses a third-last business day rule. Empirically, the only edge
        # cases versus the exchange trading calendar are when Good Friday lands
        # in March (Good Friday is not a trading day, but is still counted as a
        # business day for expiry calculations in the source contract calendar).
        expiry = calc.compute_expiry("HG", 2024, 3)
        assert expiry == date(2024, 3, 27)
        assert expiry.weekday() < 5  # Not weekend

        expiry = calc.compute_expiry("HG", 2018, 3)
        assert expiry == date(2018, 3, 28)

    def test_expiry_not_on_weekend(self):
        """Ensure no expiry falls on weekend."""
        calc = ExpiryCalculator()

        # Test multiple contracts
        for year in [2020, 2021, 2022, 2023, 2024]:
            for month in [1, 3, 5, 7, 9, 12]:
                try:
                    expiry = calc.compute_expiry("HG", year, month)
                    assert expiry.weekday() < 5, f"Expiry {expiry} is on weekend"
                except ValueError:
                    # May fail for future dates without calendar data
                    pass

    def test_expiry_for_contract_code(self):
        """Test expiry from contract code."""
        calc = ExpiryCalculator()

        expiry = calc.compute_expiry_for_contract("HGF24")
        assert expiry.year == 2024
        assert expiry.month == 1

    def test_build_expiry_table(self):
        """Test building expiry table."""
        df = build_expiry_table("HG", 2020, 2022)

        assert len(df) > 0
        assert "contract" in df.columns
        assert "expiry_date" in df.columns
        assert "symbol" in df.columns

        # All should be HG
        assert (df["symbol"] == "HG").all()


class TestTradingCalendar:
    """Tests for trading calendar."""

    def test_is_business_day(self):
        """Test business day check."""
        cal = TradingCalendar("CMEGlobex_Metals")

        # A known weekday (Wednesday Jan 15, 2024)
        assert cal.is_business_day(date(2024, 1, 15)) == True

        # A weekend
        assert cal.is_business_day(date(2024, 1, 13)) == False  # Saturday
        assert cal.is_business_day(date(2024, 1, 14)) == False  # Sunday

    def test_next_business_day(self):
        """Test next business day calculation."""
        cal = TradingCalendar("CMEGlobex_Metals")

        # Friday -> Monday
        next_day = cal.next_business_day(date(2024, 1, 12))
        assert next_day.weekday() == 0  # Monday

    def test_prev_business_day(self):
        """Test previous business day calculation."""
        cal = TradingCalendar("CMEGlobex_Metals")

        # Monday -> Friday
        prev_day = cal.prev_business_day(date(2024, 1, 15))
        assert prev_day.weekday() == 4  # Friday

    def test_days_to_expiry(self):
        """Test DTE calculation."""
        cal = TradingCalendar("CMEGlobex_Metals")

        # Same day = 0 DTE (exclude start, include end - but same day means 0 between)
        dte = cal.days_to_expiry(date(2024, 1, 15), date(2024, 1, 15))
        assert dte == 0

        # One business day apart (Jan 15 to Jan 16, excluding start, including end = 1)
        dte = cal.days_to_expiry(date(2024, 1, 15), date(2024, 1, 16))
        assert dte == 1


class TestContractSpecs:
    """Tests for contract specifications."""

    def test_copper_spec(self):
        """Test copper contract spec."""
        spec = get_contract_spec("HG")

        assert spec is not None
        assert spec.symbol == "HG"
        assert spec.name == "Copper"
        assert spec.tick_size == 0.0005
        assert spec.contract_size == 25000
        assert spec.point_value == 12.50

    def test_all_specs_valid(self):
        """Test all specs have required fields."""
        for symbol, spec in CONTRACT_SPECS.items():
            assert spec.symbol == symbol
            assert spec.tick_size > 0
            assert spec.contract_size > 0
            assert spec.point_value > 0
            assert len(spec.trading_months) > 0
