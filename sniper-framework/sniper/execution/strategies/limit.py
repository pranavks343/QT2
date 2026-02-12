"""Limit order execution."""

from typing import Any


class LimitOrderStrategy:
    """Limit order - fill at or better than limit price."""

    def create_order(
        self,
        symbol: str,
        direction: int,
        quantity: int,
        price: float,
        **kwargs: Any,
    ) -> dict:
        return {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "order_type": "LIMIT",
            "price": price,
            **kwargs,
        }
