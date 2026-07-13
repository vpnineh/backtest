"""
Configuration
=============
Loads every parameter of the simulation from a YAML file.
Nothing in the engine is hardcoded - everything referenced here comes
from config.yaml so the same code can be re-used for optimization runs.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    symbol: str = "EURGBP"
    path: str = "data"                 # folder OR single file
    file_pattern: str = "*.csv"        # used when `path` is a folder
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    datetime_column: str = "auto"
    timezone: str = "UTC"


@dataclass
class StrategyConfig:
    initial_lot: float = 0.10
    scale_factor: float = 1.20
    grid_distance_pips: float = 15.0
    max_levels: int = 10
    grid_mode: str = "fixed"           # fixed | atr
    atr_period: int = 14
    atr_multiplier: float = 1.0
    pip_size: float = 0.0001           # 0.01 for JPY pairs
    contract_size: float = 100_000.0


@dataclass
class ExitConfig:
    mode: str = "A"                    # A | B | C | D
    target_profit: float = 50.0        # Mode B: combined floating profit (account ccy)
    convergence_pips: float = 2.0      # Mode C: weighted-avg price convergence
    equilibrium_buffer_pips: float = 1.0  # Mode D: extra safety buffer


@dataclass
class CostsConfig:
    commission_per_lot: float = 7.0    # round-turn, per 1.0 lot, account ccy
    spread_pips: float = 1.2
    slippage_pips: float = 0.2
    swap_long_per_lot: float = -0.5    # per lot per night, account ccy
    swap_short_per_lot: float = 0.2


@dataclass
class AccountConfig:
    starting_balance: float = 10_000.0
    leverage: float = 100.0
    currency: str = "USD"
    margin_call_level: float = 50.0    # % - purely informational, not enforced
    quote_to_account_rate: float = 1.0  # simplification for cross-currency P/L


@dataclass
class OutputConfig:
    results_dir: str = "results"
    generate_charts: bool = True
    save_trade_log: bool = True
    save_cycle_log: bool = True
    report_name: str = "simulation_report"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        def build(cls, key):
            return cls(**raw.get(key, {})) if raw.get(key) else cls()

        return Config(
            data=build(DataConfig, "data"),
            strategy=build(StrategyConfig, "strategy"),
            exit=build(ExitConfig, "exit"),
            costs=build(CostsConfig, "costs"),
            account=build(AccountConfig, "account"),
            output=build(OutputConfig, "output"),
        )
