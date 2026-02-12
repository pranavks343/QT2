"""Volatility-based regime detection."""

from typing import Any

import pandas as pd
import numpy as np

from sniper.indicators.volatility import ATR


class SimpleRegimeDetector:
    """
    Volatility-based regime detection using ATR percentile.
    Regimes: low_vol, normal, high_vol.
    """

    def __init__(
        self,
        atr_period: int = 14,
        low_percentile: float = 25.0,
        high_percentile: float = 75.0,
    ):
        self.atr_period = atr_period
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.atr = ATR(period=atr_period)
        self.atr_history: list[float] = []
        self._last_regime: str | None = None

    def detect(self, data: pd.Series | pd.DataFrame) -> str:
        """
        Detect current regime from bar or series.
        Returns: 'low_vol', 'normal', or 'high_vol'.
        """
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return "normal"
            bar = data.iloc[-1]
        else:
            bar = data

        self.atr.update(bar)
        atr_val = self.atr[-1]
        self.atr_history.append(atr_val)

        if len(self.atr_history) < self.atr_period * 2:
            regime = "normal"
        else:
            p_low = np.percentile(self.atr_history, self.low_percentile)
            p_high = np.percentile(self.atr_history, self.high_percentile)
            if atr_val <= p_low:
                regime = "low_vol"
            elif atr_val >= p_high:
                regime = "high_vol"
            else:
                regime = "normal"

        self._last_regime = regime
        return regime

    def get_regime(self) -> str:
        """Return last detected regime."""
        return self._last_regime or "normal"

    def reset(self) -> None:
        """Reset detector state."""
        self.atr.reset()
        self.atr_history = []
        self._last_regime = None
