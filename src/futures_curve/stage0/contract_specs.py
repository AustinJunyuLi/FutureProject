"""Contract specifications for futures contracts.

Stores tick size, multiplier, trading months, and other contract details.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


@dataclass
class ContractSpec:
    """Specification for a futures contract."""

    symbol: str
    name: str
    exchange: str
    tick_size: float
    contract_size: float
    point_value: float  # Dollar value per tick
    trading_months: list[int]  # 1-12
    currency: str = "USD"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Pre-defined contract specifications
CONTRACT_SPECS: dict[str, ContractSpec] = {
    "HG": ContractSpec(
        symbol="HG",
        name="Copper",
        exchange="CME",
        tick_size=0.0005,  # $0.0005 per pound
        contract_size=25000,  # 25,000 pounds
        point_value=12.50,  # $12.50 per tick
        trading_months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All months
        description="COMEX Copper futures",
    ),
    "GC": ContractSpec(
        symbol="GC",
        name="Gold",
        exchange="CME",
        tick_size=0.10,  # $0.10 per troy oz
        contract_size=100,  # 100 troy ounces
        point_value=10.00,  # $10.00 per tick
        trading_months=[2, 4, 6, 8, 10, 12],  # Even months
        description="COMEX Gold futures",
    ),
    "SI": ContractSpec(
        symbol="SI",
        name="Silver",
        exchange="CME",
        tick_size=0.005,  # $0.005 per troy oz
        contract_size=5000,  # 5,000 troy ounces
        point_value=25.00,  # $25.00 per tick
        trading_months=[3, 5, 7, 9, 12],  # H, K, N, U, Z
        description="COMEX Silver futures",
    ),
    "CL": ContractSpec(
        symbol="CL",
        name="Crude Oil",
        exchange="CME",
        tick_size=0.01,  # $0.01 per barrel
        contract_size=1000,  # 1,000 barrels
        point_value=10.00,  # $10.00 per tick
        trading_months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        description="NYMEX WTI Crude Oil futures",
    ),
    "NG": ContractSpec(
        symbol="NG",
        name="Natural Gas",
        exchange="CME",
        tick_size=0.001,  # $0.001 per MMBtu
        contract_size=10000,  # 10,000 MMBtu
        point_value=10.00,  # $10.00 per tick
        trading_months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        description="NYMEX Henry Hub Natural Gas futures",
    ),
    "ZC": ContractSpec(
        symbol="ZC",
        name="Corn",
        exchange="CME",
        tick_size=0.25,  # 1/4 cent per bushel
        contract_size=5000,  # 5,000 bushels
        point_value=12.50,  # $12.50 per tick
        trading_months=[3, 5, 7, 9, 12],  # H, K, N, U, Z
        description="CBOT Corn futures",
    ),
    "ZS": ContractSpec(
        symbol="ZS",
        name="Soybeans",
        exchange="CME",
        tick_size=0.25,  # 1/4 cent per bushel
        contract_size=5000,  # 5,000 bushels
        point_value=12.50,  # $12.50 per tick
        trading_months=[1, 3, 5, 7, 8, 9, 11],  # F, H, K, N, Q, U, X
        description="CBOT Soybeans futures",
    ),
    "ZW": ContractSpec(
        symbol="ZW",
        name="Wheat",
        exchange="CME",
        tick_size=0.25,  # 1/4 cent per bushel
        contract_size=5000,  # 5,000 bushels
        point_value=12.50,  # $12.50 per tick
        trading_months=[3, 5, 7, 9, 12],  # H, K, N, U, Z
        description="CBOT Wheat futures",
    ),
}


def get_contract_spec(symbol: str) -> Optional[ContractSpec]:
    """Get contract specification for a symbol.

    Args:
        symbol: Commodity symbol (e.g., "HG")

    Returns:
        ContractSpec if found, None otherwise
    """
    return CONTRACT_SPECS.get(symbol.upper())


def save_contract_specs(output_path: str) -> None:
    """Save all contract specs to JSON.

    Args:
        output_path: Output file path
    """
    specs_dict = {k: v.to_dict() for k, v in CONTRACT_SPECS.items()}

    with open(output_path, "w") as f:
        json.dump(specs_dict, f, indent=2)

    print(f"Saved contract specs to {output_path}: {len(specs_dict)} contracts")


def load_contract_specs(input_path: str) -> dict[str, ContractSpec]:
    """Load contract specs from JSON.

    Args:
        input_path: Input file path

    Returns:
        Dictionary of symbol -> ContractSpec
    """
    with open(input_path) as f:
        data = json.load(f)

    return {k: ContractSpec(**v) for k, v in data.items()}
