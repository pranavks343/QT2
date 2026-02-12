"""YFinance data provider for NSE symbols."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from sniper.data.base_provider import BaseDataProvider


def _nse_symbol(symbol: str) -> str:
    """Append .NS for NSE if not already present."""
    if "." not in symbol:
        return f"{symbol}.NS"
    return symbol


class YFinanceProvider(BaseDataProvider):
    """
    Data provider using yfinance.
    Uses .NS suffix for NSE stocks (e.g. NIFTY.NS, RELIANCE.NS).
    For indices like NIFTY 50, use ^NSEI or NSEI.NS.
    """

    def __init__(self):
        if yf is None:
            raise ImportError("yfinance required: pip install yfinance")

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        interval = self._interval_from_timeframe(timeframe)
        ticker = _nse_symbol(symbol)
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = self._normalize(df)
        return df

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1d",
        bars: int = 1,
    ) -> pd.DataFrame | None:
        end = datetime.now()
        start = end - timedelta(days=7)  # Fetch a bit extra
        df = self.fetch(symbol, start, end, timeframe)
        if df.empty:
            return None
        return df.tail(bars)

    def _interval_from_timeframe(self, timeframe: str) -> str:
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "1d": "1d",
            "1wk": "1wk",
        }
        return mapping.get(timeframe.lower(), "1d")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns: open, high, low, close, volume."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0
        return df[required].dropna(how="all")
