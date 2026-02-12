"""Manages data subscriptions for backtest and live modes."""

from typing import Any
import threading
import time

from sniper.core.event_bus import EventBus, Event, EventType
from sniper.data.base_provider import BaseDataProvider
from datetime import datetime

import pandas as pd


class SubscriptionManager:
    """
    Manages data subscriptions for both backtest and live.
    In backtest: fetches historical data, feeds bar by bar in timestamp order.
    In live: polls data provider every interval (default 3s).
    """

    def __init__(self, event_bus: EventBus, data_provider: BaseDataProvider | None = None):
        self.event_bus = event_bus
        self.data_provider = data_provider
        self.subscriptions: dict[str, dict] = {}  # symbol -> config
        self.is_running = False
        self.mode: str | None = None
        self._live_thread: threading.Thread | None = None
        self._live_interval = 3.0

    def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe to a symbol."""
        self.subscriptions[symbol] = {"timeframe": timeframe}

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from a symbol."""
        if symbol in self.subscriptions:
            del self.subscriptions[symbol]

    def start_backtest(self, start_date: datetime, end_date: datetime) -> None:
        """Feed historical data bar by bar in timestamp order."""
        self.mode = "backtest"
        self.is_running = True

        if not self.data_provider or not self.subscriptions:
            return

        all_bars: list[tuple[datetime, str, pd.Series]] = []

        for symbol, config in self.subscriptions.items():
            timeframe = config.get("timeframe", "1d")
            df = self.data_provider.fetch(symbol, start_date, end_date, timeframe)

            if df.empty:
                continue

            for idx, row in df.iterrows():
                ts = idx if isinstance(idx, datetime) else pd.Timestamp(idx)
                all_bars.append((ts, symbol, row))

        all_bars.sort(key=lambda x: x[0])

        for ts, symbol, bar in all_bars:
            if not self.is_running:
                break

            self.event_bus.publish(
                Event(
                    type=EventType.DATA_RECEIVED,
                    timestamp=ts,
                    data={"symbol": symbol, "bar": bar},
                )
            )

    def start_live(self, interval_seconds: float = 3.0) -> None:
        """Connect to live data stream via polling."""
        self.mode = "live"
        self.is_running = True
        self._live_interval = interval_seconds
        self._live_thread = threading.Thread(target=self._live_stream_loop, daemon=True)
        self._live_thread.start()

    def stop_live(self) -> None:
        """Stop live streaming."""
        self.is_running = False
        if self._live_thread:
            self._live_thread.join(timeout=5.0)
            self._live_thread = None

    def _live_stream_loop(self) -> None:
        """Background loop for live data polling."""
        while self.is_running:
            if not self.data_provider:
                time.sleep(self._live_interval)
                continue

            for symbol, config in self.subscriptions.items():
                if not self.is_running:
                    break

                timeframe = config.get("timeframe", "1d")
                latest = self.data_provider.fetch_latest(symbol, timeframe, bars=1)

                if latest is not None and not latest.empty:
                    bar = latest.iloc[-1]
                    self.event_bus.publish(
                        Event(
                            type=EventType.DATA_RECEIVED,
                            timestamp=datetime.now(),
                            data={"symbol": symbol, "bar": bar},
                        )
                    )

            time.sleep(self._live_interval)
