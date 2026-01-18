"""Tests for utility modules."""

import pytest
from datetime import date, datetime
import pandas as pd

from futures_curve.utils.month_codes import (
    MONTH_CODE_TO_NUMBER,
    MONTH_NUMBER_TO_CODE,
    parse_contract_code,
    contract_to_expiry_month,
    format_contract,
)
from futures_curve.utils.timezone import (
    get_trade_date,
    get_bucket_number,
    localize_to_central,
    CENTRAL_TZ,
)


class TestMonthCodes:
    """Tests for month code utilities."""

    def test_month_code_mapping(self):
        """Test month code to number mapping."""
        assert MONTH_CODE_TO_NUMBER["F"] == 1
        assert MONTH_CODE_TO_NUMBER["Z"] == 12
        assert MONTH_NUMBER_TO_CODE[1] == "F"
        assert MONTH_NUMBER_TO_CODE[12] == "Z"

    def test_parse_contract_code(self):
        """Test contract code parsing."""
        # Standard format
        info = parse_contract_code("HG_F09")
        assert info is not None
        assert info.symbol == "HG"
        assert info.month_code == "F"
        assert info.year_full == 2009
        assert info.month == 1

        # Filename format
        info = parse_contract_code("HG_Z24_1min.txt")
        assert info is not None
        assert info.symbol == "HG"
        assert info.month_code == "Z"
        assert info.year_full == 2024
        assert info.month == 12

        # Invalid
        info = parse_contract_code("INVALID")
        assert info is None

    def test_contract_to_expiry_month(self):
        """Test expiry month extraction."""
        result = parse_contract_code("HGF24")
        assert result is not None
        assert result.year_full == 2024
        assert result.month == 1

    def test_format_contract(self):
        """Test contract code formatting."""
        assert format_contract("HG", 2024, 1) == "HGF24"
        assert format_contract("HG", 2024, 12) == "HGZ24"


class TestTimezone:
    """Tests for timezone utilities."""

    def test_get_trade_date_before_cutoff(self):
        """Test trade date before 5 PM cutoff."""
        # 2 PM Central = same day trade date
        dt = datetime(2024, 1, 15, 14, 0, 0)
        trade_date = get_trade_date(dt)
        assert trade_date == date(2024, 1, 15)

    def test_get_trade_date_after_cutoff(self):
        """Test trade date after 5 PM cutoff."""
        # 6 PM Central = next day trade date
        dt = datetime(2024, 1, 15, 18, 0, 0)
        trade_date = get_trade_date(dt)
        assert trade_date == date(2024, 1, 16)

    def test_get_trade_date_at_cutoff(self):
        """Test trade date at exactly 5 PM cutoff."""
        # 5 PM Central = next day trade date
        dt = datetime(2024, 1, 15, 17, 0, 0)
        trade_date = get_trade_date(dt)
        assert trade_date == date(2024, 1, 16)

    def test_bucket_us_session(self):
        """Test bucket assignment for US session."""
        # 9 AM = bucket 1
        dt = datetime(2024, 1, 15, 9, 30, 0)
        assert get_bucket_number(dt) == 1

        # 2 PM = bucket 6
        dt = datetime(2024, 1, 15, 14, 30, 0)
        assert get_bucket_number(dt) == 6

        # 3 PM = bucket 7
        dt = datetime(2024, 1, 15, 15, 30, 0)
        assert get_bucket_number(dt) == 7

    def test_bucket_post_us(self):
        """Test bucket assignment for post-reopen session."""
        # 4 PM CT is maintenance hour (bucket 0)
        dt = datetime(2024, 1, 15, 16, 0, 0)
        assert get_bucket_number(dt) == 0

        # 5 PM CT is session reopen (bucket 8)
        dt = datetime(2024, 1, 15, 17, 0, 0)
        assert get_bucket_number(dt) == 8

        # 8 PM CT = bucket 8
        dt = datetime(2024, 1, 15, 20, 0, 0)
        assert get_bucket_number(dt) == 8

    def test_bucket_overnight(self):
        """Test bucket assignment for overnight session."""
        # 9 PM = bucket 9
        dt = datetime(2024, 1, 15, 21, 0, 0)
        assert get_bucket_number(dt) == 9

        # 1 AM = bucket 9
        dt = datetime(2024, 1, 15, 1, 0, 0)
        assert get_bucket_number(dt) == 9

    def test_bucket_pre_us(self):
        """Test bucket assignment for pre-US session."""
        # 5 AM = bucket 10
        dt = datetime(2024, 1, 15, 5, 0, 0)
        assert get_bucket_number(dt) == 10

        # 8 AM = bucket 10
        dt = datetime(2024, 1, 15, 8, 30, 0)
        assert get_bucket_number(dt) == 10
