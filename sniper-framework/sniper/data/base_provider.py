"""Abstract data provider interface."""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class BaseDataProvider(ABC):
    """
    Abstract data provider. Implement for yfinance, CSV, synthetic, etc.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        Returns DataFrame with columns: open, high, low, close, volume.
        Index should be DatetimeIndex.
        """
        pass

    @abstractmethod
    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1d",
        bars: int = 1,
    ) -> pd.DataFrame | None:
        """
        Fetch most recent bar(s). For live/streaming.
        Returns DataFrame or None if unavailable.
        """
        pass
