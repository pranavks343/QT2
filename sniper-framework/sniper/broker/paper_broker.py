"""Paper trading broker - simulates orders with immediate fills."""

from datetime import datetime
from typing import Any

from sniper.core.event_bus import EventBus, Event, EventType
from sniper.broker.base_broker import BaseBroker


class PaperBroker(BaseBroker):
    """
    Paper broker: simulates fills at order price (or bar close).
    Publishes ORDER_FILLED when event_bus provided.
    """

    def __init__(self, event_bus: EventBus | None = None):
        self.orders: list[dict] = []
        self.fills: list[dict] = []
        self.event_bus = event_bus

    def place_order(self, order: dict) -> str:
        """Simulate order placement and immediate fill."""
        order_id = f"paper_{len(self.orders)}"
        order = {**order, "order_id": order_id, "status": "submitted"}
        self.orders.append(order)

        fill_price = order.get("price")
        if fill_price is None:
            fill_price = 0.0
        fill = {
            "order_id": order_id,
            "symbol": order.get("symbol", ""),
            "direction": order.get("direction", 0),
            "quantity": order.get("quantity", 0),
            "fill_price": fill_price,
            "status": "filled",
            "timestamp": order.get("timestamp", datetime.now()),
        }
        self.fills.append(fill)

        if self.event_bus:
            self.event_bus.publish(
                Event(
                    type=EventType.ORDER_FILLED,
                    timestamp=fill["timestamp"],
                    data={**order, **fill},
                )
            )
        return order_id
