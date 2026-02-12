"""Moving average indicators."""

import pandas as pd

from sniper.indicators.base import Indicator


class EMA(Indicator):
    """Exponential Moving Average."""

    def __init__(self, period: int, source: str = "close"):
        super().__init__(f"EMA_{period}")
        self.period = period
        self.source = source
        self.alpha = 2.0 / (period + 1)
        self.ema: float | None = None

    def compute(self, data: pd.Series) -> float:
        value = float(data[self.source]) if self.source in data.index else float(data.get(self.source, 0))

        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        return self.ema

    def reset(self) -> None:
        super().reset()
        self.ema = None


class SMA(Indicator):
    """Simple Moving Average."""

    def __init__(self, period: int, source: str = "close"):
        super().__init__(f"SMA_{period}")
        self.period = period
        self.source = source
        self.window: list[float] = []

    def compute(self, data: pd.Series) -> float:
        value = float(data[self.source]) if self.source in data.index else float(data.get(self.source, 0))
        self.window.append(value)

        if len(self.window) > self.period:
            self.window.pop(0)

        return sum(self.window) / len(self.window)

    def reset(self) -> None:
        super().reset()
        self.window = []
