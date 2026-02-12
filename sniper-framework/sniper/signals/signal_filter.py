"""Signal filtering by confidence threshold."""

from typing import Any


class SignalFilter:
    """
    Filter signals by confidence threshold.
    """

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence

    def passes(self, signal: dict) -> bool:
        """Check if signal passes confidence filter."""
        conf = signal.get("confidence", 0.0)
        return conf >= self.min_confidence

    def filter(self, signals: list[dict]) -> list[dict]:
        """Filter list of signals."""
        return [s for s in signals if self.passes(s)]
