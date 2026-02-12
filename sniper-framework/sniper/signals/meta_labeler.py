"""Meta-labeling: filter raw signals by quality score."""

from typing import Any

import pandas as pd


class MetaLabeler:
    """
    Meta-label filter: score raw signals based on quality.
    Returns score 0-1. Signal passes if score >= threshold.
    """

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def score(self, bar: pd.Series, direction: int, regime: str = "normal") -> float:
        """
        Score a signal. Returns 0-1.
        Uses simple heuristics: volatility regime, bar range, etc.
        """
        if regime == "high_vol":
            return 0.0
        if regime == "low_vol":
            base = 0.7
        else:
            base = 0.6

        high = float(bar.get("high", 0))
        low = float(bar.get("low", 0))
        close = float(bar.get("close", 0))

        if high == low:
            return base
        range_pct = (high - low) / close if close else 0
        if range_pct > 0.05:
            base -= 0.1
        return max(0.0, min(1.0, base))

    def passes(self, bar: pd.Series, direction: int, regime: str = "normal") -> bool:
        """Check if signal passes meta-label filter."""
        return self.score(bar, direction, regime) >= self.threshold
