"""
SNIPER FRAMEWORK - DATA LAYER
Abstract base class + implementations for each data source.
Swap providers by changing config.DATA["provider"] — nothing else changes.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DATA, STRATEGY


# ─── ABSTRACT BASE ──────────────────────────────────────────────────────────
class BaseDataProvider(ABC):
    """
    All data providers must return a DataFrame with these columns:
    datetime | open | high | low | close | volume
    Index: DatetimeIndex
    """

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV data for symbol between start and end dates."""
        pass

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Data missing column: {col}")
        df = df.sort_index()
        df = df.dropna(subset=required)
        return df


# ─── SYNTHETIC DATA PROVIDER ────────────────────────────────────────────────
class SyntheticProvider(BaseDataProvider):
    """
    Generates realistic NSE-like OHLCV data with:
    - Regime shifts (trending / ranging / high-vol phases)
    - Realistic intraday volatility
    - Volume correlation with price moves
    Use for development and testing without any real data.
    """

    def fetch(self, symbol: str = "NIFTY", start: str = None, end: str = None) -> pd.DataFrame:
        np.random.seed(DATA["synthetic_seed"])
        n = DATA["synthetic_bars"]

        # Generate time index (5-min bars, NSE session: 9:15 to 15:30)
        dates = self._generate_timestamps(n)

        # Simulate price with regime changes
        prices = self._simulate_price(n, base=22000.0 if symbol == "NIFTY" else 48000.0)

        # Build OHLCV
        df = self._build_ohlcv(dates, prices, n)
        return self.validate(df)

    def _generate_timestamps(self, n: int) -> list:
        timestamps = []
        dt = datetime(2021, 1, 4, 9, 15)
        bars_per_day = 75  # ~75 bars per NSE session at 5-min
        count = 0
        while count < n:
            if dt.weekday() < 5:  # Mon-Fri only
                for m in range(0, 375, 5):  # 9:15 to 15:30 = 375 min
                    if count >= n:
                        break
                    timestamps.append(dt + timedelta(minutes=m))
                    count += 1
            dt += timedelta(days=1)
        return timestamps

    def _simulate_price(self, n: int, base: float) -> np.ndarray:
        """
        Simulate price with 4 regime phases cycling through the series:
        bull-trend, ranging, bear-trend, high-volatility
        """
        prices = np.zeros(n)
        prices[0] = base

        # Define regime segments
        segment_size = n // 4
        regimes = [
            {"drift": 0.0003, "vol": 0.002},   # bull trend
            {"drift": 0.0000, "vol": 0.001},   # ranging
            {"drift": -0.0003, "vol": 0.002},  # bear trend
            {"drift": 0.0001, "vol": 0.005},   # high vol / choppy
        ]

        for i in range(1, n):
            regime_idx = min(i // segment_size, 3)
            r = regimes[regime_idx]
            shock = np.random.normal(r["drift"], r["vol"])
            prices[i] = prices[i - 1] * (1 + shock)

        return prices

    def _build_ohlcv(self, dates: list, closes: np.ndarray, n: int) -> pd.DataFrame:
        noise = np.abs(np.random.normal(0.001, 0.0005, n))
        highs = closes * (1 + noise)
        lows = closes * (1 - noise)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        # Volume: higher on big moves
        price_change = np.abs(np.diff(closes, prepend=closes[0])) / closes
        volume_base = 100_000
        volumes = (volume_base * (1 + 5 * price_change) * np.random.lognormal(0, 0.3, n)).astype(int)

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }, index=pd.DatetimeIndex(dates))


# ─── CSV PROVIDER ────────────────────────────────────────────────────────────
class CSVProvider(BaseDataProvider):
    """
    Load data from a CSV file.
    Expected columns: datetime, open, high, low, close, volume
    """

    def fetch(self, symbol: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        df = pd.read_csv(DATA["csv_path"], parse_dates=["datetime"], index_col="datetime")
        df.columns = [c.lower().strip() for c in df.columns]
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return self.validate(df)


# ─── KITE (ZERODHA) PROVIDER ─────────────────────────────────────────────────
class KiteProvider(BaseDataProvider):
    """
    Live/historical data from Zerodha Kite API.
    Requires: pip install kiteconnect
    Set your api_key and access_token below or via environment variables.
    """

    def __init__(self, api_key: str = None, access_token: str = None):
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
        except ImportError:
            raise ImportError("Install kiteconnect: pip install kiteconnect")

    def fetch(self, symbol: str = "NSE:NIFTY 50", start: str = None, end: str = None) -> pd.DataFrame:
        interval_map = {"1min": "minute", "5min": "5minute", "15min": "15minute", "day": "day"}
        interval = interval_map.get(STRATEGY["timeframe"], "5minute")
        data = self.kite.historical_data(
            instrument_token=self._get_token(symbol),
            from_date=start,
            to_date=end,
            interval=interval,
        )
        df = pd.DataFrame(data)
        df = df.rename(columns={"date": "datetime"})
        df = df.set_index("datetime")
        return self.validate(df)

    def _get_token(self, symbol: str) -> int:
        # Map symbol to Kite instrument token — extend as needed
        tokens = {
            "NIFTY": 256265,
            "BANKNIFTY": 260105,
        }
        return tokens.get(symbol, 256265)


# ─── FACTORY ─────────────────────────────────────────────────────────────────
def get_provider() -> BaseDataProvider:
    """Returns the configured data provider."""
    provider = DATA["provider"]
    if provider == "synthetic":
        return SyntheticProvider()
    elif provider == "csv":
        return CSVProvider()
    elif provider == "kite":
        return KiteProvider()
    elif provider == "yfinance":
        from .yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    else:
        raise ValueError(f"Unknown provider: {provider}. Options: synthetic, csv, kite, yfinance")
