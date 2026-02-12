"""Walk-forward optimization."""

from datetime import datetime
from typing import Any, Callable

from sniper.backtest.engine import BacktestEngine
from sniper.backtest.results import BacktestResults


def walk_forward(
    strategy_class: type,
    data_provider: Any,
    start_date: datetime,
    end_date: datetime,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
) -> list[BacktestResults]:
    """
    Walk-forward optimization: train on train_days, test on test_days, step forward.
    Stub: returns single full-period backtest.
    """
    engine = BacktestEngine(strategy=strategy_class, data_provider=data_provider)
    return [engine.run(start_date, end_date)]
