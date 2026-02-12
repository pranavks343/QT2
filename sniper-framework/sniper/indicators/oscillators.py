"""Oscillator indicators."""

import pandas as pd
import numpy as np

from sniper.indicators.base import Indicator


class RSI(Indicator):
    """Relative Strength Index."""

    def __init__(self, period: int = 14):
        super().__init__(f"RSI_{period}")
        self.period = period
        self.prev_close: float | None = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0

    def compute(self, data: pd.Series) -> float:
        close = float(data["close"]) if "close" in data.index else float(data.get("close", 0))

        if self.prev_close is None:
            self.prev_close = close
            return 50.0

        change = close - self.prev_close
        gain = max(change, 0)
        loss = max(-change, 0)

        if len(self.values) == 0:
            self.avg_gain = gain
            self.avg_loss = loss
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        self.prev_close = close

        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100 - (100 / (1 + rs))

    def reset(self) -> None:
        super().reset()
        self.prev_close = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0


class MACD(Indicator):
    """Moving Average Convergence Divergence. Returns (macd_line, signal_line, histogram)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD_{fast}_{slow}_{signal}")
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self.fast_alpha = 2.0 / (fast + 1)
        self.slow_alpha = 2.0 / (slow + 1)
        self.signal_alpha = 2.0 / (signal + 1)
        self.fast_ema: float | None = None
        self.slow_ema: float | None = None
        self.macd_line: float | None = None
        self.signal_ema: float | None = None

    def compute(self, data: pd.Series) -> float:
        close = float(data["close"]) if "close" in data.index else float(data.get("close", 0))

        if self.fast_ema is None:
            self.fast_ema = close
            self.slow_ema = close
            return 0.0

        self.fast_ema = self.fast_alpha * close + (1 - self.fast_alpha) * self.fast_ema
        self.slow_ema = self.slow_alpha * close + (1 - self.slow_alpha) * self.slow_ema
        macd_val = self.fast_ema - self.slow_ema

        if self.signal_ema is None:
            self.signal_ema = macd_val
        else:
            self.signal_ema = self.signal_alpha * macd_val + (1 - self.signal_alpha) * self.signal_ema

        return macd_val - self.signal_ema  # histogram as primary value

    def get_lines(self) -> tuple[float, float, float]:
        """Return (macd_line, signal_line, histogram)."""
        if self.fast_ema is None or self.slow_ema is None:
            return 0.0, 0.0, 0.0
        macd = self.fast_ema - self.slow_ema
        sig = self.signal_ema or macd
        hist = macd - sig
        return macd, sig, hist

    def reset(self) -> None:
        super().reset()
        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None


class Stochastic(Indicator):
    """Stochastic Oscillator. Returns %K (or average of %K and %D)."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__(f"Stoch_{k_period}_{d_period}")
        self.k_period = k_period
        self.d_period = d_period
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.closes: list[float] = []
        self.k_values: list[float] = []

    def compute(self, data: pd.Series) -> float:
        high = float(data["high"]) if "high" in data.index else float(data.get("high", 0))
        low = float(data["low"]) if "low" in data.index else float(data.get("low", 0))
        close = float(data["close"]) if "close" in data.index else float(data.get("close", 0))

        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        if len(self.highs) > self.k_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)

        if len(self.highs) < self.k_period:
            return 50.0

        hh = max(self.highs)
        ll = min(self.lows)
        if hh == ll:
            k = 50.0
        else:
            k = 100 * (close - ll) / (hh - ll)

        self.k_values.append(k)
        if len(self.k_values) > self.d_period:
            self.k_values.pop(0)

        d = sum(self.k_values) / len(self.k_values) if self.k_values else 50.0
        return (k + d) / 2  # Return smoothed value

    def reset(self) -> None:
        super().reset()
        self.highs = []
        self.lows = []
        self.closes = []
        self.k_values = []
