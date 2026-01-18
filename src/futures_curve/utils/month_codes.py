"""Futures month code utilities.

Standard CME month codes:
F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun
N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
"""

from dataclasses import dataclass
from typing import Optional
import re

# Month code mappings
MONTH_CODE_TO_NUMBER: dict[str, int] = {
    "F": 1,   # January
    "G": 2,   # February
    "H": 3,   # March
    "J": 4,   # April
    "K": 5,   # May
    "M": 6,   # June
    "N": 7,   # July
    "Q": 8,   # August
    "U": 9,   # September
    "V": 10,  # October
    "X": 11,  # November
    "Z": 12,  # December
}

MONTH_NUMBER_TO_CODE: dict[int, str] = {v: k for k, v in MONTH_CODE_TO_NUMBER.items()}

# Month names for display
MONTH_CODE_TO_NAME: dict[str, str] = {
    "F": "January",
    "G": "February",
    "H": "March",
    "J": "April",
    "K": "May",
    "M": "June",
    "N": "July",
    "Q": "August",
    "U": "September",
    "V": "October",
    "X": "November",
    "Z": "December",
}


@dataclass
class ContractInfo:
    """Parsed contract information."""

    symbol: str          # e.g., "HG"
    month_code: str      # e.g., "F"
    year_short: int      # e.g., 9 for 2009
    year_full: int       # e.g., 2009
    month: int           # e.g., 1 for January
    contract_code: str   # e.g., "HGF09"

    def __str__(self) -> str:
        return self.contract_code


# Pattern to match contract codes like "HG_F09" or "HGF09"
CONTRACT_PATTERN = re.compile(r"^([A-Z]{2,4})_?([FGHJKMNQUVXZ])(\d{2})$")


def parse_contract_code(code: str) -> Optional[ContractInfo]:
    """Parse a contract code into its components.

    Args:
        code: Contract code like "HG_F09", "HGF09", or filename "HG_F09_1min.txt"

    Returns:
        ContractInfo if valid, None otherwise

    Examples:
        >>> parse_contract_code("HG_F09")
        ContractInfo(symbol='HG', month_code='F', year_short=9, year_full=2009, month=1, contract_code='HGF09')
        >>> parse_contract_code("HG_Z24_1min.txt")
        ContractInfo(symbol='HG', month_code='Z', year_short=24, year_full=2024, month=12, contract_code='HGZ24')
    """
    # Handle filename format
    if code.endswith("_1min.txt"):
        code = code.replace("_1min.txt", "")

    match = CONTRACT_PATTERN.match(code.upper())
    if not match:
        return None

    symbol = match.group(1)
    month_code = match.group(2)
    year_short = int(match.group(3))

    # Convert 2-digit year to 4-digit (assume 2000s for 00-99)
    year_full = 2000 + year_short

    month = MONTH_CODE_TO_NUMBER.get(month_code)
    if month is None:
        return None

    contract_code = f"{symbol}{month_code}{year_short:02d}"

    return ContractInfo(
        symbol=symbol,
        month_code=month_code,
        year_short=year_short,
        year_full=year_full,
        month=month,
        contract_code=contract_code,
    )


def contract_to_expiry_month(contract_code: str) -> Optional[tuple[int, int]]:
    """Get the expiry year and month for a contract.

    Args:
        contract_code: Contract code like "HGF09"

    Returns:
        Tuple of (year, month) or None if invalid
    """
    info = parse_contract_code(contract_code)
    if info is None:
        return None
    return (info.year_full, info.month)


def format_contract(symbol: str, year: int, month: int) -> str:
    """Format a contract code from components.

    Args:
        symbol: Commodity symbol like "HG"
        year: Full year like 2024
        month: Month number 1-12

    Returns:
        Contract code like "HGZ24"
    """
    month_code = MONTH_NUMBER_TO_CODE.get(month)
    if month_code is None:
        raise ValueError(f"Invalid month: {month}")
    year_short = year % 100
    return f"{symbol}{month_code}{year_short:02d}"
