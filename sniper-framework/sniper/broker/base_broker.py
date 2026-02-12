"""Abstract broker interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseBroker(ABC):
    """
    Abstract broker interface. Implement for Zerodha, paper trading, etc.
    """

    @abstractmethod
    def place_order(self, order: dict) -> Any:
        """Place an order. Returns order ID or order object."""
        pass

    def get_positions(self) -> list[dict]:
        """Get current positions. Optional override."""
        return []

    def get_orders(self) -> list[dict]:
        """Get order history. Optional override."""
        return []
