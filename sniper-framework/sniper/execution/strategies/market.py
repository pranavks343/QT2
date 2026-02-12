"""Market order execution."""

from typing import Any


class MarketOrderStrategy:
    """Market order - fill at current price."""

    def create_order(
        self,
        symbol: str,
        direction: int,
        quantity: int,
        price: float | None = None,
        **kwargs: Any,
    ) -> dict:
        return {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "order_type": "MARKET",
            "price": price,
            **kwargs,
        }
