"""
SNIPER FRAMEWORK - YFINANCE DATA PROVIDER
Real historical + recent OHLCV data from Yahoo Finance.
Falls back to synthetic data if fetch fails.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .base_provider import BaseDataProvider, SyntheticProvider


class YFinanceProvider(BaseDataProvider):
    """
    Fetch real market data using yfinance.
    Maps Indian symbols:
      NIFTY → ^NSEI
      BANKNIFTY → ^NSEBANK
      RELIANCE → RELIANCE.NS
      TCS → TCS.NS
    etc.
    
    Supports multiple timeframes: 1m, 5m, 15m, 1h, 1d
    Falls back to synthetic data if yfinance fetch fails.
    """
    
    SYMBOL_MAP = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "WIPRO": "WIPRO.NS",
        "ITC": "ITC.NS",
    }
    
    INTERVAL_MAP = {
        "1m": "1m",
        "1min": "1m",
        "5m": "5m",
        "5min": "5m",
        "15m": "15m",
        "15min": "15m",
        "1h": "1h",
        "1hour": "1h",
        "1d": "1d",
        "1day": "1d",
        "day": "1d",
    }
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            print("⚠️  yfinance not installed. Install with: pip install yfinance")
            print("    Falling back to synthetic data provider.")
            self.yf = None
    
    def fetch(self, symbol: str = "NIFTY", start: str = None, end: str = None, 
              timeframe: str = "5m", bars: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Symbol name (NIFTY, BANKNIFTY, or stock name)
            start: Start date (YYYY-MM-DD) - optional
            end: End date (YYYY-MM-DD) - optional
            timeframe: Interval (1m, 5m, 15m, 1h, 1d)
            bars: Number of bars to fetch if start/end not specified
        
        Returns:
            DataFrame with OHLCV data
        """
        # Fallback to synthetic if yfinance not available
        if self.yf is None:
            print(f"  → Using synthetic data for {symbol}")
            return SyntheticProvider().fetch(symbol, start, end)
        
        # Map symbol
        yf_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())
        
        # Map interval
        interval = self.INTERVAL_MAP.get(timeframe.lower(), "5m")
        
        # Determine date range
        if end is None:
            end = datetime.now()
        else:
            end = pd.to_datetime(end)
        
        if start is None:
            # Calculate start based on bars count and interval
            start = self._calculate_start_date(end, interval, bars)
        else:
            start = pd.to_datetime(start)
        
        try:
            # Fetch data from yfinance
            ticker = self.yf.Ticker(yf_symbol)
            
            # For intraday data (1m, 5m, etc), yfinance limits to last 7 days
            if interval in ["1m", "5m", "15m"]:
                # Limit to last 7 days for intraday
                max_start = datetime.now() - timedelta(days=7)
                if start < max_start:
                    start = max_start
                    print(f"  ⚠️  Intraday data limited to 7 days. Adjusted start to {start.date()}")
            
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                actions=False
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {yf_symbol}")
            
            # Rename columns to match our schema
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure we have required columns
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")
            
            # Keep only required columns
            df = df[required]
            
            # Reset index to make datetime a column, then set it back
            df = df.reset_index()
            df = df.rename(columns={"index": "datetime", "date": "datetime"})
            
            # Handle timezone-aware datetime
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                if df["datetime"].dt.tz is not None:
                    df["datetime"] = df["datetime"].dt.tz_localize(None)
            
            df = df.set_index("datetime")
            
            # Sort and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            # If we don't have enough bars, pad with earlier data
            if len(df) < bars * 0.5:  # Less than 50% of requested bars
                print(f"  ⚠️  Only {len(df)} bars fetched (requested {bars}). Using available data.")
            
            print(f"  ✓ Fetched {len(df)} bars for {symbol} ({yf_symbol}) @ {interval}")
            
            return self.validate(df)
            
        except Exception as e:
            print(f"  ⚠️  yfinance fetch failed for {symbol}: {e}")
            print(f"  → Falling back to synthetic data")
            return SyntheticProvider().fetch(symbol, start, end)
    
    def fetch_latest(self, symbol: str, timeframe: str = "5m", bars: int = 200) -> pd.DataFrame:
        """
        Fetch most recent N bars for a symbol.
        Optimized for WebSocket streaming.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1d)
            bars: Number of bars to fetch

        Returns:
            DataFrame with latest bars
        """
        end = datetime.now()
        return self.fetch(symbol=symbol, start=None, end=None, timeframe=timeframe, bars=bars)

    def _calculate_start_date(self, end: datetime, interval: str, bars: int) -> datetime:
        """Calculate start date based on number of bars and interval."""
        # Estimate time delta based on interval
        if interval == "1m":
            # Account for market hours: ~375 trading minutes per day
            days = max(7, (bars * 1) / 375)  # Max 7 days for yfinance limit
            return end - timedelta(days=min(days, 7))
        elif interval == "5m":
            days = max(7, (bars * 5) / 375)
            return end - timedelta(days=min(days, 7))
        elif interval == "15m":
            days = max(7, (bars * 15) / 375)
            return end - timedelta(days=min(days, 7))
        elif interval == "1h":
            days = (bars * 60) / 375
            return end - timedelta(days=days)
        elif interval == "1d":
            # Account for weekends: ~252 trading days per year
            days = bars * 1.4  # Add 40% for weekends
            return end - timedelta(days=days)
        else:
            return end - timedelta(days=30)  # Default 30 days


# For easy import
__all__ = ["YFinanceProvider"]
