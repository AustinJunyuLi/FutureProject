"""Stage 0: Metadata acquisition - expiry schedules and trading calendars."""

from .expiry_schedule import (
    ExpiryCalculator,
    build_expiry_table,
    save_expiry_table,
)
from .trading_calendar import (
    TradingCalendar,
    build_trading_calendar,
    save_trading_calendar,
)
from .contract_specs import (
    ContractSpec,
    CONTRACT_SPECS,
    get_contract_spec,
    save_contract_specs,
    load_contract_specs,
    load_specs_from_expiry_rules,
)

__all__ = [
    "ExpiryCalculator",
    "build_expiry_table",
    "save_expiry_table",
    "TradingCalendar",
    "build_trading_calendar",
    "save_trading_calendar",
    "ContractSpec",
    "CONTRACT_SPECS",
    "get_contract_spec",
    "save_contract_specs",
    "load_contract_specs",
    "load_specs_from_expiry_rules",
]
