"""Base strategy class - subclass to implement custom strategies."""

from abc import ABC, abstractmethod
from typing import Any

from sniper.core.event_bus import EventBus, Event, EventType
from datetime import datetime

# Avoid circular import - use TYPE_CHECKING for pd.Series
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseStrategy(ABC):
    """
    Base class for all strategies. Inspired by QuantConnect's QCAlgorithm.

    Lifecycle:
    1. __init__() — called once on creation
    2. initialize() — setup indicators, subscriptions (user override)
    3. on_data(bar) — called for each new bar (user override)
    4. on_signal(signal) — called when signal generated (user override)
    5. on_order_filled(order) — called when order fills (user override)
    6. on_position_closed(position) — called when position closes (user override)
    7. shutdown() — cleanup (user override)
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.data: dict[str, Any] = {}  # Store data by symbol
        self.indicators: dict[str, Any] = {}  # Store indicators by name
        self.positions: dict[str, Any] = {}  # Active positions
        self.portfolio_value = 0.0
        self.is_backtesting = True
        self.current_time: datetime | None = None
        self._subscriptions: list[tuple[str, str]] = []  # (symbol, timeframe)

        # Subscribe to core events
        self.event_bus.subscribe(EventType.DATA_RECEIVED, self._handle_data)
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._handle_signal)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self.event_bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)

    @abstractmethod
    def initialize(self) -> None:
        """Setup indicators, subscribe to symbols. User must implement."""
        pass

    @abstractmethod
    def on_data(self, symbol: str, bar: "pd.Series") -> None:
        """Called on each new bar. User must implement."""
        pass

    def on_signal(self, signal: dict) -> None:
        """Called when signal generated. User can override."""
        pass

    def on_order_filled(self, order: dict) -> None:
        """Called when order fills. User can override."""
        pass

    def on_position_closed(self, position: dict) -> None:
        """Called when position closes. User can override."""
        pass

    def shutdown(self) -> None:
        """Cleanup. User can override."""
        pass

    def subscribe(self, symbol: str, timeframe: str) -> None:
        """Request subscription to a symbol. Algorithm registers with SubscriptionManager."""
        self._subscriptions.append((symbol, timeframe))

    def get_subscriptions(self) -> list[tuple[str, str]]:
        """Return requested (symbol, timeframe) subscriptions."""
        return list(self._subscriptions)

    # Internal event handlers
    def _handle_data(self, event: Event) -> None:
        symbol = event.data["symbol"]
        bar = event.data["bar"]
        self.current_time = event.timestamp
        self.data[symbol] = bar
        self.on_data(symbol, bar)

    def _handle_signal(self, event: Event) -> None:
        self.on_signal(event.data)

    def _handle_order_filled(self, event: Event) -> None:
        self.on_order_filled(event.data)

    def _handle_position_closed(self, event: Event) -> None:
        self.on_position_closed(event.data)

    # Helper methods for strategy authors
    def add_indicator(self, name: str, indicator: Any) -> None:
        """Register an indicator."""
        self.indicators[name] = indicator

    def get_indicator(self, name: str) -> Any:
        """Retrieve indicator value."""
        return self.indicators.get(name)

    def emit_signal(
        self,
        direction: int,
        confidence: float,
        metadata: dict | None = None,
        symbol: str | None = None,
    ) -> None:
        """Emit a trading signal. direction: 1=long, -1=short, 0=close."""
        meta = metadata or {}
        if symbol and "symbol" not in meta:
            meta["symbol"] = symbol
        signal = {
            "direction": direction,
            "confidence": confidence,
            "timestamp": self.current_time,
            "metadata": meta,
            "symbol": symbol,
        }
        self.event_bus.publish(
            Event(
                type=EventType.SIGNAL_GENERATED,
                timestamp=self.current_time or datetime.now(),
                data=signal,
            )
        )

    def get_position(self, symbol: str) -> Any:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log message (will be picked up by logging system)."""
        print(f"[{level}] {self.current_time} | {message}")
