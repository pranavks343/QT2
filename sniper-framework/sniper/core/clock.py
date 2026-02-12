"""Time management - optional stub for Phase 1."""

from datetime import datetime


class Clock:
    """
    Stub for time management. Used for simulation time in backtest.
    Full implementation when needed.
    """

    def __init__(self):
        self.current_time: datetime | None = None

    def set_time(self, dt: datetime) -> None:
        """Set current simulation time."""
        self.current_time = dt

    def now(self) -> datetime:
        """Get current time (simulation or real)."""
        return self.current_time or datetime.now()
