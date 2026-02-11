"""Yahoo Finance data provider."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from config import STRATEGY

from .base_provider import BaseDataProvider, SyntheticProvider

LOGGER = logging.getLogger(__name__)


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
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "day": "1d",
    }

    def fetch(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Base provider-compatible fetch signature."""
        timeframe = self.TIMEFRAME_MAP.get(STRATEGY.get("timeframe", "5min"), "5m")
        return self._download(symbol=symbol, start=start, end=end, timeframe=timeframe, bars=200)

    def fetch_latest(self, symbol: str, timeframe: str = "5m", bars: int = 200) -> pd.DataFrame:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=14 if timeframe in {"1m", "5m", "15m"} else max(30, bars * 2))
        df = self._download(
            symbol=symbol,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            timeframe=timeframe,
            bars=bars,
        )
        return df.tail(bars)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            flattened = []
            for col in df.columns:
                if isinstance(col, tuple):
                    flattened.append(str(col[0]))
                else:
                    flattened.append(str(col))
            df.columns = flattened
        df.columns = [str(col).strip().lower() for col in df.columns]
        return df

    def _download(self, symbol: str, start: Optional[str], end: Optional[str], timeframe: str, bars: int) -> pd.DataFrame:
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

            df = self._normalize_columns(df)
            required_cols = ["open", "high", "low", "close", "volume"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            df = df[required_cols].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df["_data_source"] = "real"
            return self.validate(df)
        except (ValueError, RuntimeError) as exc:
            LOGGER.warning("yfinance fetch failed for %s (%s): %s. Falling back to synthetic provider.", symbol, timeframe, exc)
            fallback_df = SyntheticProvider().fetch(symbol=symbol, start=start, end=end)
            fallback_df["_data_source"] = "synthetic"
            return fallback_df


__all__ = ["YFinanceProvider"]
