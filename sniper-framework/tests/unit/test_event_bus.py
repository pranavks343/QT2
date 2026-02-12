"""Unit tests for EventBus."""

from datetime import datetime

from sniper.core.event_bus import EventBus, Event, EventType


def test_subscribe_publish():
    bus = EventBus(max_history=10)
    received = []

    def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.SIGNAL_GENERATED, handler)
    bus.publish(Event(type=EventType.SIGNAL_GENERATED, timestamp=datetime.now(), data={"x": 1}))

    assert len(received) == 1
    assert received[0].data["x"] == 1


def test_get_history():
    bus = EventBus(max_history=5)
    for i in range(3):
        bus.publish(
            Event(type=EventType.DATA_RECEIVED, timestamp=datetime.now(), data={"i": i})
        )

    history = bus.get_history()
    assert len(history) == 3

    filtered = bus.get_history(EventType.DATA_RECEIVED)
    assert len(filtered) == 3
    assert all(e.type == EventType.DATA_RECEIVED for e in filtered)


def test_history_bounded():
    bus = EventBus(max_history=3)
    for i in range(5):
        bus.publish(Event(type=EventType.ORDER_FILLED, timestamp=datetime.now(), data={}))

    assert len(bus.get_history()) == 3


def test_unsubscribe():
    bus = EventBus()
    received = []

    def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.SIGNAL_GENERATED, handler)
    bus.publish(Event(type=EventType.SIGNAL_GENERATED, timestamp=datetime.now(), data={}))
    assert len(received) == 1

    bus.unsubscribe(EventType.SIGNAL_GENERATED, handler)
    bus.publish(Event(type=EventType.SIGNAL_GENERATED, timestamp=datetime.now(), data={}))
    assert len(received) == 1
