"""Yahoo Finance data provider."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from .base_provider import BaseDataProvider, SyntheticProvider


class YFinanceProvider(BaseDataProvider):
    """Market data provider backed by yfinance."""

    SYMBOL_MAP: Dict[str, str] = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
    }

    TIMEFRAME_MAP: Dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1d": "1d",
    }

    def fetch(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: str = "5m",
        bars: int = 200,
    ) -> pd.DataFrame:
        mapped_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol)
        interval = self.TIMEFRAME_MAP.get(timeframe, "5m")

        try:
            end_dt = datetime.fromisoformat(end) if end else datetime.now()
            if start:
                start_dt = datetime.fromisoformat(start)
            else:
                lookback_days = 10 if interval in {"1m", "5m", "15m"} else max(30, int(bars * 1.5))
                start_dt = end_dt - timedelta(days=lookback_days)

            df = yf.download(
                mapped_symbol,
                start=start_dt,
                end=end_dt,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                raise ValueError(f"No yfinance data for {symbol}")

            df.columns = [str(col).lower() for col in df.columns]
            required_cols = ["open", "high", "low", "close", "volume"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            df = df[required_cols].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            return self.validate(df)
        except Exception as exc:
            print(f"[yfinance] warning: {exc}. Falling back to synthetic provider.")
            return SyntheticProvider().fetch(symbol=symbol, start=start, end=end)

    def fetch_latest(self, symbol: str, timeframe: str = "5m", bars: int = 200) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=14 if timeframe in {"1m", "5m", "15m"} else bars * 2)
        df = self.fetch(
            symbol=symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            timeframe=timeframe,
            bars=bars,
        )
        return df.tail(bars)


__all__ = ["YFinanceProvider"]
