"""Shock detection based on volatility/return spikes."""

from typing import Any

import numpy as np
import pandas as pd


class ShockDetector:
    """
    Detect market shocks from return/volatility spikes.
    Returns True when |return| exceeds threshold.
    """

    def __init__(
        self,
        return_threshold: float = 0.03,
        lookback: int = 20,
    ):
        self.return_threshold = return_threshold
        self.lookback = lookback
        self._closes: list[float] = []
        self._in_shock = False

    def detect(self, data: pd.Series | pd.DataFrame) -> bool:
        """
        Check if current bar represents a shock.
        Returns True if shock detected.
        """
        close = self._extract_close(data)
        if close is None:
            return self._in_shock

        self._closes.append(close)
        if len(self._closes) > 500:
            self._closes = self._closes[-500:]

        if len(self._closes) < 2:
            return False

        ret = (self._closes[-1] - self._closes[-2]) / self._closes[-2] if self._closes[-2] else 0
        self._in_shock = abs(ret) >= self.return_threshold
        return self._in_shock

    def _extract_close(self, data: pd.Series | pd.DataFrame) -> float | None:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return None
            if "close" in data.columns:
                return float(data["close"].iloc[-1])
            return float(data.iloc[-1, 0])
        if isinstance(data, pd.Series):
            return float(data.get("close", 0))
        return None

    def in_shock(self) -> bool:
        """Return whether currently in shock state."""
        return self._in_shock

    def reset(self) -> None:
        """Reset detector state."""
        self._closes = []
        self._in_shock = False
