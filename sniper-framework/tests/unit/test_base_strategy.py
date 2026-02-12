"""Unit tests for BaseStrategy."""

from datetime import datetime

from sniper.core.event_bus import EventBus, Event, EventType
from sniper.core.base_strategy import BaseStrategy


class ConcreteStrategy(BaseStrategy):
    def initialize(self):
        self.subscribe("NIFTY", "5m")

    def on_data(self, symbol: str, bar):
        pass


def test_subscriptions():
    bus = EventBus()
    strategy = ConcreteStrategy(bus)
    strategy.initialize()
    assert strategy.get_subscriptions() == [("NIFTY", "5m")]


def test_emit_signal():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.SIGNAL_GENERATED, lambda e: received.append(e))

    strategy = ConcreteStrategy(bus)
    strategy.current_time = datetime.now()
    strategy.emit_signal(direction=1, confidence=0.8, symbol="NIFTY")

    assert len(received) == 1
    assert received[0].data["direction"] == 1
    assert received[0].data["confidence"] == 0.8
    assert received[0].data["symbol"] == "NIFTY"


def test_data_handler():
    bus = EventBus()
    strategy = ConcreteStrategy(bus)
    strategy.initialize()

    bar = {"open": 100, "high": 105, "low": 99, "close": 103}
    bus.publish(
        Event(
            type=EventType.DATA_RECEIVED,
            timestamp=datetime.now(),
            data={"symbol": "NIFTY", "bar": bar},
        )
    )

    assert strategy.data["NIFTY"] == bar
    assert strategy.current_time is not None
