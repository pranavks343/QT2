"""Minimal example strategy demonstrating Phase 1 framework."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sniper.core.event_bus import EventBus, EventType
from sniper.core.base_strategy import BaseStrategy
from sniper.core.algorithm import Algorithm
from sniper.data.subscription_manager import SubscriptionManager
from sniper.risk.risk_manager import RiskManager
from sniper.execution.executor import Executor
from sniper.broker.paper_broker import PaperBroker


class MinimalStrategy(BaseStrategy):
    """Strategy that subscribes to NIFTY and logs data events."""

    def initialize(self) -> None:
        self.subscribe("NIFTY", "5m")
        self.signal_count = 0

    def on_data(self, symbol: str, bar) -> None:
        # Stub - no real data in Phase 1
        pass

    def on_signal(self, signal: dict) -> None:
        self.signal_count += 1
        self.log(f"Signal received: {signal}")


def main():
    event_bus = EventBus()
    strategy = MinimalStrategy(event_bus)
    sub_manager = SubscriptionManager(event_bus)
    risk_manager = RiskManager(event_bus)
    executor = Executor()
    broker = PaperBroker()

    algorithm = Algorithm(
        strategy=strategy,
        subscription_manager=sub_manager,
        risk_manager=risk_manager,
        executor=executor,
        broker=broker,
        event_bus=event_bus,
    )

    print("Running backtest (stub - no data feed)...")
    algorithm.run_backtest(datetime(2024, 1, 1), datetime(2024, 1, 2))
    print("Backtest complete.")
    print("Event history:", [e.type for e in event_bus.get_history()])


if __name__ == "__main__":
    main()
