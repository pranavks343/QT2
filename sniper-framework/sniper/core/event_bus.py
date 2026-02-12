"""Event-driven architecture for the trading framework."""

from typing import Callable, Dict, List
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """All event types in the trading system."""

    # Data events
    DATA_RECEIVED = "data_received"
    DATA_VALIDATED = "data_validated"

    # Strategy events
    INDICATORS_COMPUTED = "indicators_computed"
    REGIME_DETECTED = "regime_detected"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_FILTERED = "signal_filtered"
    SHOCK_DETECTED = "shock_detected"

    # RL events
    RL_OBSERVATION_READY = "rl_observation_ready"
    RL_DECISION_MADE = "rl_decision_made"

    # Risk events
    POSITION_SIZED = "position_sized"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_BREACH = "drawdown_breach"

    # Execution events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"

    # Performance events
    PNL_REALIZED = "pnl_realized"
    ALPHA_DECAY = "alpha_decay"

    # System events
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_ENDED = "backtest_ended"
    LIVE_STARTED = "live_started"
    LIVE_STOPPED = "live_stopped"


class Event(BaseModel):
    """Immutable event with type, timestamp, and payload."""

    type: EventType
    timestamp: datetime
    data: dict
    trace_id: str | None = None  # For tracing signal → trade lifecycle


class EventBus:
    """
    Central event bus for publish/subscribe communication.
    All framework components communicate via events.
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = max_history

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register a handler for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Remove a handler for an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        if event.type in self._subscribers:
            for handler in self._subscribers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but don't crash
                    print(f"Error in event handler for {event.type}: {e}")

    def get_history(self, event_type: EventType | None = None) -> List[Event]:
        """Get event history, optionally filtered by type."""
        if event_type:
            return [e for e in self._event_history if e.type == event_type]
        return list(self._event_history)
