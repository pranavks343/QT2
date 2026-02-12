"""Volatility indicators."""

import pandas as pd

from sniper.indicators.base import Indicator


class ATR(Indicator):
    """Average True Range."""

    def __init__(self, period: int = 14):
        super().__init__(f"ATR_{period}")
        self.period = period
        self.tr_values: list[float] = []
        self._prev_close: float | None = None

    def compute(self, data: pd.Series) -> float:
        high = float(data["high"]) if "high" in data.index else float(data.get("high", 0))
        low = float(data["low"]) if "low" in data.index else float(data.get("low", 0))
        close = float(data["close"]) if "close" in data.index else float(data.get("close", 0))

        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close

        self.tr_values.append(tr)
        if len(self.tr_values) > self.period:
            self.tr_values.pop(0)

        return sum(self.tr_values) / len(self.tr_values)

    def reset(self) -> None:
        super().reset()
        self.tr_values = []
        self._prev_close = None


class BollingerBands(Indicator):
    """Bollinger Bands. Returns middle band (SMA). Use get_bands() for upper/lower."""

    def __init__(self, period: int = 20, std_dev: float = 2.0, source: str = "close"):
        super().__init__(f"BB_{period}")
        self.period = period
        self.std_dev = std_dev
        self.source = source
        self.window: list[float] = []

    def compute(self, data: pd.Series) -> float:
        value = float(data[self.source]) if self.source in data.index else float(data.get(self.source, 0))
        self.window.append(value)

        if len(self.window) > self.period:
            self.window.pop(0)

        mean = sum(self.window) / len(self.window)
        if len(self.window) < self.period:
            return mean

        variance = sum((x - mean) ** 2 for x in self.window) / len(self.window)
        std = variance ** 0.5
        return mean  # Middle band as primary value

    def get_bands(self) -> tuple[float, float, float]:
        """Return (upper, middle, lower)."""
        if len(self.window) < self.period:
            m = sum(self.window) / len(self.window) if self.window else 0
            return m, m, m
        middle = sum(self.window) / len(self.window)
        variance = sum((x - middle) ** 2 for x in self.window) / len(self.window)
        std = variance ** 0.5
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        return upper, middle, lower

    def reset(self) -> None:
        super().reset()
        self.window = []
