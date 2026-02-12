"""CSV data provider for backtests."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from sniper.data.base_provider import BaseDataProvider


class CsvProvider(BaseDataProvider):
    """
    Read OHLCV data from CSV files.
    Expected columns: date/datetime, open, high, low, close, volume.
    Date column is used as index.
    """

    def __init__(self, base_path: str | Path = "."):
        self.base_path = Path(base_path)

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        path = self._path_for_symbol(symbol)
        if not path.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.read_csv(path)
        df = self._parse_and_index(df)

        if df.empty:
            return df

        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask]

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1d",
        bars: int = 1,
    ) -> pd.DataFrame | None:
        path = self._path_for_symbol(symbol)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df = self._parse_and_index(df)
        return df.tail(bars) if not df.empty else None

    def _path_for_symbol(self, symbol: str) -> Path:
        """Resolve CSV path. Looks for {symbol}.csv or {symbol}_{timeframe}.csv."""
        candidates = [
            self.base_path / f"{symbol}.csv",
            self.base_path / f"{symbol.upper()}.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    def _parse_and_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column and set as index. Normalize column names."""
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        date_cols = ["date", "datetime", "time"]
        date_col = None
        for dc in date_cols:
            if dc in df.columns:
                date_col = dc
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0
        return df[required]
