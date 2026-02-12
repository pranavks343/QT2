"""Real-time portfolio tracking."""

from typing import Any

from sniper.core.event_bus import EventBus, EventType


class PortfolioMonitor:
    """
    Track portfolio state from events.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.positions: dict[str, dict] = {}
        self.equity: float = 0.0
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self.event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)

    def _on_order_filled(self, event) -> None:
        order = event.data
        symbol = order.get("symbol", "")
        qty = order.get("quantity", 0)
        direction = order.get("direction", 0)
        price = order.get("fill_price", order.get("price", 0))
        if symbol:
            key = symbol
            if key not in self.positions:
                self.positions[key] = {"quantity": 0, "avg_price": 0}
            curr = self.positions[key]
            new_qty = curr["quantity"] + direction * qty
            if new_qty == 0:
                del self.positions[key]
            else:
                curr["quantity"] = new_qty
                curr["avg_price"] = price

    def _on_position_closed(self, event) -> None:
        pos = event.data
        symbol = pos.get("symbol", "")
        if symbol in self.positions:
            del self.positions[symbol]

    def get_positions(self) -> dict[str, dict]:
        """Return current positions."""
        return dict(self.positions)
