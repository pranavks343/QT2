"""Risk management - position sizing, drawdown limits, VaR."""

from datetime import datetime
from typing import Any

from sniper.core.event_bus import EventBus, Event, EventType


class RiskManager:
    """
    Risk manager: Kelly sizing, max drawdown halt, VaR/CVaR, stress tests.
    """

    def __init__(
        self,
        event_bus: EventBus,
        initial_capital: float = 500000.0,
        max_drawdown_pct: float = 0.10,
        kelly_fraction: float = 0.25,
    ):
        self.event_bus = event_bus
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.positions: dict[str, Any] = {}
        self.trading_halted = False
        self.halt_reason: str | None = None
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction
        self.position_limits: dict[str, int] = {}
        self.sector_limits: dict[str, float] = {}
        self.var_confidence = 0.95

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self.trading_halted:
            return False
        drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital else 0
        if drawdown >= self.max_drawdown_pct:
            self._halt("drawdown_breach", drawdown)
            return False
        if drawdown >= self.max_drawdown_pct * 0.8:
            self.event_bus.publish(
                Event(
                    type=EventType.DRAWDOWN_WARNING,
                    timestamp=datetime.now(),
                    data={"drawdown": drawdown, "peak": self.peak_capital},
                )
            )
        return True

    def _halt(self, reason: str, drawdown: float) -> None:
        self.trading_halted = True
        self.halt_reason = f"{reason}: drawdown {drawdown:.1%}"
        self.event_bus.publish(
            Event(
                type=EventType.DRAWDOWN_BREACH,
                timestamp=datetime.now(),
                data={"drawdown": drawdown, "reason": reason},
            )
        )

    def update_capital(self, value: float) -> None:
        """Update portfolio value."""
        self.capital = value
        if value > self.peak_capital:
            self.peak_capital = value

    def compute_position_size(self, signal: dict) -> dict:
        """Kelly-based position sizing."""
        confidence = signal.get("confidence", 0.5)
        p = max(0.01, min(0.99, confidence))
        q = 1 - p
        kelly = p - q if p > q else 0
        kelly = max(0, kelly * self.kelly_fraction)
        lot_size = 25
        max_lots = int((self.capital * 0.02 * kelly) / (lot_size * 100)) if kelly > 0 else 0
        lots = max(1, min(max_lots or 1, 10))
        return {
            "lots": lots,
            "quantity": lots * lot_size,
            "confidence": confidence,
            "kelly_fraction": kelly,
        }

    def check_concentration_risk(self) -> dict:
        """Warn if too concentrated in one symbol/sector."""
        total = sum(
            p.get("value", 0) for p in self.positions.values()
        )
        if total == 0:
            return {"warn": False}
        max_single = max(
            (p.get("value", 0) / total for p in self.positions.values()),
            default=0
        )
        return {"warn": max_single > 0.5, "max_concentration": max_single}

    def compute_var(self, confidence: float = 0.95) -> float:
        """Value at Risk (stub)."""
        return self.capital * 0.02

    def compute_cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (stub)."""
        return self.capital * 0.03

    def stress_test(self, scenarios: list[dict]) -> dict:
        """Run portfolio through stress scenarios (stub)."""
        return {"passed": True, "min_value": self.capital * 0.9}
