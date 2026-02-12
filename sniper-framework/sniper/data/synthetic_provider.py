"""Synthetic data provider for testing."""

from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

from sniper.data.base_provider import BaseDataProvider


class SyntheticProvider(BaseDataProvider):
    """
    Generate random OHLCV data for testing.
    Uses geometric Brownian motion for price trajectory.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        freq = self._freq_from_timeframe(timeframe)
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        if len(dates) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        base_price = 100.0
        returns = np.random.normal(0.0002, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        high = prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01))
        low = prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01))
        open_ = np.roll(prices, 1)
        open_[0] = base_price

        volume = np.random.randint(1000, 100000, len(dates)).astype(float)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(high, np.maximum(open_, prices)),
                "low": np.minimum(low, np.minimum(open_, prices)),
                "close": prices,
                "volume": volume,
            },
            index=dates,
        )
        return df

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1d",
        bars: int = 1,
    ) -> pd.DataFrame | None:
        end = datetime.now()
        start = end - timedelta(days=30)
        df = self.fetch(symbol, start, end, timeframe)
        return df.tail(bars) if not df.empty else None

    def _freq_from_timeframe(self, timeframe: str) -> str:
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "1d": "1D",
            "1wk": "1W",
        }
        return mapping.get(timeframe.lower(), "1D")
