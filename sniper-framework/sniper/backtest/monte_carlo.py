"""Monte Carlo simulation for backtest robustness."""

from datetime import datetime
from typing import Any

from sniper.backtest.engine import BacktestEngine
from sniper.backtest.results import BacktestResults


def monte_carlo(
    strategy_class: type,
    data_provider: Any,
    start_date: datetime,
    end_date: datetime,
    n_simulations: int = 100,
    shuffle_trades: bool = True,
) -> list[BacktestResults]:
    """
    Monte Carlo: shuffle trade order or returns to assess robustness.
    Stub: returns single run.
    """
    engine = BacktestEngine(strategy=strategy_class, data_provider=data_provider)
    return [engine.run(start_date, end_date)]
