"""Slippage modeling."""

from typing import Any


class SlippageModel:
    """
    Slippage model: fixed % or ATR-based.
    """

    def __init__(self, mode: str = "fixed", value: float = 0.001):
        self.mode = mode
        self.value = value

    def apply(self, price: float, quantity: int, atr: float | None = None) -> float:
        """Return execution price after slippage."""
        if self.mode == "fixed":
            return price * (1 + self.value)
        if self.mode == "atr" and atr:
            return price + self.value * atr
        return price
