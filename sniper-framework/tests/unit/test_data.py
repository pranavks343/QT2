"""Unit tests for data layer."""

from datetime import datetime

from sniper.core.event_bus import EventBus, EventType
from sniper.data.synthetic_provider import SyntheticProvider
from sniper.data.subscription_manager import SubscriptionManager


def test_synthetic_provider():
    provider = SyntheticProvider(seed=123)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 10)
    df = provider.fetch("NIFTY", start, end, "1d")
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) >= 5


def test_subscription_manager_backtest():
    bus = EventBus()
    provider = SyntheticProvider(seed=42)
    sub = SubscriptionManager(bus, provider)
    sub.subscribe("NIFTY", "1d")

    events = []
    bus.subscribe(EventType.DATA_RECEIVED, lambda e: events.append(e))

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 5)
    sub.start_backtest(start, end)

    assert len(events) >= 1
    assert events[0].data["symbol"] == "NIFTY"
    assert "bar" in events[0].data
