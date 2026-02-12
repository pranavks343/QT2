"""Momentum strategy using EMA, RSI, and synthetic data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from sniper.core.event_bus import EventBus, EventType
from sniper.core.base_strategy import BaseStrategy
from sniper.core.algorithm import Algorithm
from sniper.data.subscription_manager import SubscriptionManager
from sniper.data.synthetic_provider import SyntheticProvider
from sniper.risk.risk_manager import RiskManager
from sniper.execution.executor import Executor
from sniper.broker.paper_broker import PaperBroker
from sniper.indicators.moving_averages import EMA
from sniper.indicators.oscillators import RSI


class MomentumStrategy(BaseStrategy):
    """EMA crossover + RSI filter."""

    def initialize(self) -> None:
        self.ema_fast = EMA(period=9)
        self.ema_slow = EMA(period=21)
        self.rsi = RSI(period=14)
        self.add_indicator("ema_fast", self.ema_fast)
        self.add_indicator("ema_slow", self.ema_slow)
        self.add_indicator("rsi", self.rsi)
        self.subscribe("NIFTY", "1d")
        self.signal_count = 0

    def on_data(self, symbol: str, bar) -> None:
        self.ema_fast.update(bar)
        self.ema_slow.update(bar)
        self.rsi.update(bar)

        if len(self.ema_fast.values) < 21:
            return

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        rsi_val = self.rsi[-1]

        if ema_f > ema_s and rsi_val > 50 and rsi_val < 75:
            self.emit_signal(
                direction=1,
                confidence=0.7,
                symbol=symbol,
                metadata={"entry_price": float(bar.get("close", 0))},
            )
            self.signal_count += 1
        elif ema_f < ema_s and rsi_val < 50 and rsi_val > 25:
            self.emit_signal(
                direction=-1,
                confidence=0.7,
                symbol=symbol,
                metadata={"entry_price": float(bar.get("close", 0))},
            )
            self.signal_count += 1


def main():
    event_bus = EventBus()
    provider = SyntheticProvider(seed=42)
    sub_manager = SubscriptionManager(event_bus, provider)
    strategy = MomentumStrategy(event_bus)
    risk_manager = RiskManager(event_bus)
    executor = Executor()
    broker = PaperBroker(event_bus=event_bus)

    algorithm = Algorithm(
        strategy=strategy,
        subscription_manager=sub_manager,
        risk_manager=risk_manager,
        executor=executor,
        broker=broker,
        event_bus=event_bus,
    )

    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 31)
    print(f"Running backtest {start.date()} to {end.date()}...")
    algorithm.run_backtest(start, end)

    print(f"Signals generated: {strategy.signal_count}")
    print(f"Orders placed: {len(broker.orders)}")
    print("Done.")


if __name__ == "__main__":
    main()
