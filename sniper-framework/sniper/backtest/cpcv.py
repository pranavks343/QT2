"""Combinatorial Purged Cross-Validation for backtest robustness."""

from datetime import datetime
from typing import Callable, Any

from sniper.backtest.engine import BacktestEngine
from sniper.backtest.results import BacktestResults


def run_cpcv(
    strategy_class: type,
    data_provider: Any,
    start_date: datetime,
    end_date: datetime,
    n_splits: int = 5,
    purge_gap: int = 5,
    embargo: int = 2,
) -> list[BacktestResults]:
    """
    Run Combinatorial Purged Cross-Validation.
    Stub implementation - returns single run for now.
    """
    engine = BacktestEngine(strategy=strategy_class, data_provider=data_provider)
    results = engine.run(start_date, end_date)
    return [results]
