"""Unit tests for backtest engine."""

from datetime import datetime

from sniper.backtest.engine import BacktestEngine
from sniper.backtest.results import BacktestResults
from sniper.data.synthetic_provider import SyntheticProvider
from sniper.indicators.moving_averages import EMA
from sniper.indicators.oscillators import RSI
from sniper.core.base_strategy import BaseStrategy
from sniper.core.event_bus import EventBus


class SimpleStrategy(BaseStrategy):
    """Minimal strategy for testing."""

    def initialize(self):
        self.ema = EMA(5)
        self.rsi = RSI(5)
        self.subscribe("NIFTY", "1d")

    def on_data(self, symbol, bar):
        self.ema.update(bar)
        self.rsi.update(bar)
        if len(self.ema.values) >= 5 and self.ema[-1] > bar.get("close", 0):
            self.emit_signal(1, 0.7, metadata={"entry_price": bar.get("close", 0)}, symbol=symbol)
        elif len(self.ema.values) >= 5 and self.ema[-1] < bar.get("close", 0):
            self.emit_signal(-1, 0.7, metadata={"entry_price": bar.get("close", 0)}, symbol=symbol)


def test_backtest_engine():
    provider = SyntheticProvider(seed=123)
    engine = BacktestEngine(strategy=SimpleStrategy, data_provider=provider)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 29)
    results = engine.run(start, end)
    assert isinstance(results, BacktestResults)
    assert results.total_trades() >= 0
    assert 0 <= results.accuracy() <= 1
    assert results.verdict() in ("PASS", "INSUFFICIENT_TRADES", "LOW_ACCURACY", "LOW_SHARPE", "HIGH_DRAWDOWN")


def test_backtest_results():
    results = BacktestResults(
        trades=[{"pnl": 100}, {"pnl": -50}, {"pnl": 80}],
        equity_curve=[
            (datetime(2024, 1, 1), 100000),
            (datetime(2024, 1, 2), 100100),
            (datetime(2024, 1, 3), 100050),
            (datetime(2024, 1, 4), 100130),
        ],
        initial_capital=100000,
    )
    assert results.total_trades() == 3
    assert results.winning_trades() == 2
    assert results.accuracy() == 2 / 3
    assert results.total_pnl() == 130
    assert results.final_equity() == 100130
    assert results.max_drawdown() >= 0
