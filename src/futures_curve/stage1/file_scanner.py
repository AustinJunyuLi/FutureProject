"""File scanner for discovering and parsing futures data files.

Scans directories for minute-level data files and extracts contract metadata.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional
import re

from ..utils.month_codes import parse_contract_code, ContractInfo


@dataclass
class DataFile:
    """Metadata for a data file."""

    path: Path
    filename: str
    contract_info: ContractInfo
    file_size: int

    @property
    def symbol(self) -> str:
        return self.contract_info.symbol

    @property
    def contract(self) -> str:
        return self.contract_info.contract_code

    @property
    def year(self) -> int:
        return self.contract_info.year_full

    @property
    def month(self) -> int:
        return self.contract_info.month


class FileScanner:
    """Scanner for futures data files."""

    # Pattern: SYMBOL_MONTHYEAR_1min.txt (e.g., HG_F09_1min.txt)
    FILE_PATTERN = re.compile(r"^([A-Z]{2,4})_([FGHJKMNQUVXZ])(\d{2})_1min\.txt$")

    def __init__(self, root_dir: str | Path):
        """Initialize scanner.

        Args:
            root_dir: Root directory containing commodity folders
        """
        self.root_dir = Path(root_dir)

    def scan_commodity(self, commodity_dir: str | Path) -> list[DataFile]:
        """Scan a commodity directory for data files.

        Args:
            commodity_dir: Path to commodity directory (e.g., /data/copper/)

        Returns:
            List of DataFile objects sorted by contract date
        """
        commodity_dir = Path(commodity_dir)
        if not commodity_dir.exists():
            raise FileNotFoundError(f"Directory not found: {commodity_dir}")

        files = []
        for entry in commodity_dir.iterdir():
            if not entry.is_file():
                continue

            data_file = self._parse_file(entry)
            if data_file is not None:
                files.append(data_file)

        # Sort by year, then month
        files.sort(key=lambda f: (f.year, f.month))
        return files

    def scan_symbol(self, symbol: str) -> list[DataFile]:
        """Scan for all files of a specific symbol.

        Args:
            symbol: Commodity symbol (e.g., "HG")

        Returns:
            List of DataFile objects
        """
        # Map symbols to directory names
        dir_mapping = {
            "HG": "copper",
            "GC": "gold",
            "SI": "silver",
            "CL": "crude_oil",
            "NG": "natural_gas",
            "ZC": "corn",
            "ZS": "soybeans",
            "ZW": "wheat",
        }

        dir_name = dir_mapping.get(symbol.upper())
        if dir_name is None:
            raise ValueError(f"Unknown symbol: {symbol}")

        commodity_dir = self.root_dir / dir_name
        return self.scan_commodity(commodity_dir)

    def _parse_file(self, path: Path) -> Optional[DataFile]:
        """Parse a file path into DataFile.

        Args:
            path: File path

        Returns:
            DataFile if valid, None otherwise
        """
        match = self.FILE_PATTERN.match(path.name)
        if not match:
            return None

        contract_info = parse_contract_code(path.name)
        if contract_info is None:
            return None

        return DataFile(
            path=path,
            filename=path.name,
            contract_info=contract_info,
            file_size=path.stat().st_size,
        )

    def iter_files(
        self,
        symbol: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Iterator[DataFile]:
        """Iterate over files for a symbol with optional year filter.

        Args:
            symbol: Commodity symbol
            start_year: Start year (inclusive)
            end_year: End year (inclusive)

        Yields:
            DataFile objects
        """
        files = self.scan_symbol(symbol)

        for f in files:
            if start_year is not None and f.year < start_year:
                continue
            if end_year is not None and f.year > end_year:
                continue
            yield f

    def get_file_stats(self, symbol: str) -> dict:
        """Get statistics for files of a symbol.

        Args:
            symbol: Commodity symbol

        Returns:
            Dictionary with file statistics
        """
        files = self.scan_symbol(symbol)

        if not files:
            return {
                "symbol": symbol,
                "file_count": 0,
                "total_size_mb": 0,
                "year_range": None,
                "contracts": [],
            }

        total_size = sum(f.file_size for f in files)
        years = [f.year for f in files]
        contracts = [f.contract for f in files]

        return {
            "symbol": symbol,
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "year_range": (min(years), max(years)),
            "contracts": contracts,
        }


def discover_all_commodities(root_dir: str | Path) -> dict[str, list[DataFile]]:
    """Discover all commodity files in a directory.

    Args:
        root_dir: Root data directory

    Returns:
        Dictionary mapping symbol -> list of DataFile
    """
    root_dir = Path(root_dir)
    scanner = FileScanner(root_dir)

    result = {}
    symbols = ["HG", "GC", "SI", "CL", "NG", "ZC", "ZS", "ZW"]

    for symbol in symbols:
        try:
            files = scanner.scan_symbol(symbol)
            if files:
                result[symbol] = files
        except (FileNotFoundError, ValueError):
            continue

    return result
