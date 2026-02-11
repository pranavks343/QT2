"""
SNIPER FRAMEWORK - RISK MANAGER
Handles:
  1. Position sizing (Fractional Kelly)
  2. Per-trade risk limits
  3. Drawdown Guardian (hard kill switch at 10% DD)
  4. Daily loss limit
  5. Concurrent position tracking
"""

import numpy as np
from config import RISK


class RiskManager:
    """
    Central risk controller. Must be called before every trade.
    Maintains running state: capital, peak capital, daily P&L, open positions.
    """

    def __init__(self, capital: float = None):
        self.initial_capital = capital or RISK["capital"]
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.daily_start_capital = self.initial_capital
        self.open_positions = 0
        self.trading_halted = False
        self.halt_reason = None

        # Tracking
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.win_streak = 0
        self.loss_streak = 0
        self._win_prob_estimate = 0.5   # Updated dynamically

    # ─── POSITION SIZING ─────────────────────────────────────────────────────
    def kelly_position_size(self, win_prob: float, avg_win: float, avg_loss: float,
                             entry_price: float, stop_loss: float) -> dict:
        """
        Fractional Kelly position sizing.

        Kelly formula: f = (bp - q) / b
          b = avg_win / avg_loss (reward-to-risk ratio)
          p = probability of winning
          q = 1 - p

        Returns: {lots, capital_at_risk, position_value, valid}
        """
        if self.trading_halted:
            return self._zero_size("Trading halted")

        if self.open_positions >= RISK["max_concurrent_positions"]:
            return self._zero_size("Max concurrent positions reached")

        # Guard against bad inputs
        if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
            return self._zero_size("Invalid Kelly inputs")

        b = avg_win / avg_loss  # reward-to-risk
        p = win_prob
        q = 1 - p

        kelly_f = (b * p - q) / b
        kelly_f = max(0.0, kelly_f)  # Kelly can't be negative — means no trade

        # Apply fractional Kelly (quarter-Kelly is conservative)
        fractional_kelly = kelly_f * RISK["kelly_fraction"]

        # Cap at maximum position size
        fractional_kelly = min(fractional_kelly, RISK["max_position_pct"])

        capital_at_risk = self.current_capital * fractional_kelly

        # Convert capital at risk to lots
        # Risk per lot = price distance to stop × lot size (use NIFTY 75 as default)
        from config import COSTS
        lot_size = COSTS["lot_size_nifty"]
        risk_per_lot = abs(entry_price - stop_loss) * lot_size

        if risk_per_lot <= 0:
            return self._zero_size("Zero risk per lot")

        lots = int(capital_at_risk / risk_per_lot)
        lots = max(0, lots)

        if lots == 0:
            return self._zero_size("Kelly suggests 0 lots")

        return {
            "valid": True,
            "lots": lots,
            "capital_at_risk": risk_per_lot * lots,
            "position_value": entry_price * lot_size * lots,
            "kelly_fraction_used": fractional_kelly,
            "reward_to_risk": b,
            "reason": "OK",
        }

    # ─── DRAWDOWN GUARDIAN ───────────────────────────────────────────────────
    def update_capital(self, pnl: float, is_new_day: bool = False) -> dict:
        """
        Call after every trade closes.
        Updates capital, checks drawdown limits, fires kill switch if needed.
        Returns: {status, current_capital, drawdown_pct, daily_pnl_pct, halted}
        """
        if is_new_day:
            self.daily_start_capital = self.current_capital

        self.current_capital += pnl

        # Update peak (only on profit)
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Track win/loss
        if pnl > 0:
            self.wins += 1
            self.win_streak += 1
            self.loss_streak = 0
        elif pnl < 0:
            self.losses += 1
            self.loss_streak += 1
            self.win_streak = 0

        self.total_trades += 1

        # Calculate drawdowns
        drawdown_pct = (self.peak_capital - self.current_capital) / self.peak_capital
        daily_pnl_pct = (self.current_capital - self.daily_start_capital) / self.daily_start_capital

        status = "OK"

        # Hard kill switch: max drawdown breach
        if drawdown_pct >= RISK["max_drawdown_pct"] and not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown breached: {drawdown_pct:.1%}"
            status = "HALTED_DD"

        # Daily loss limit
        elif daily_pnl_pct <= -RISK["daily_loss_limit_pct"] and not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit breached: {daily_pnl_pct:.1%}"
            status = "HALTED_DAILY"

        return {
            "status": status,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "drawdown_pct": drawdown_pct,
            "daily_pnl_pct": daily_pnl_pct,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
        }

    # ─── POSITION TRACKING ───────────────────────────────────────────────────
    def open_trade(self):
        self.open_positions += 1

    def close_trade(self):
        self.open_positions = max(0, self.open_positions - 1)

    def reset_daily(self):
        """Call at start of each trading day."""
        self.daily_start_capital = self.current_capital
        # Note: We do NOT reset trading_halted here.
        # Max drawdown halt is permanent until manual review.
        # Daily halt resets at day start.
        if self.halt_reason and "Daily" in (self.halt_reason or ""):
            self.trading_halted = False
            self.halt_reason = None

    def can_trade(self) -> bool:
        return not self.trading_halted

    # ─── STATS ───────────────────────────────────────────────────────────────
    def accuracy(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    def summary(self) -> dict:
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        max_dd = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        return {
            "initial_capital": self.initial_capital,
            "current_capital": round(self.current_capital, 2),
            "total_return_pct": round(total_return * 100, 2),
            "peak_capital": round(self.peak_capital, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "accuracy": round(self.accuracy() * 100, 2),
            "open_positions": self.open_positions,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
        }

    # ─── HELPERS ─────────────────────────────────────────────────────────────
    @staticmethod
    def _zero_size(reason: str) -> dict:
        return {
            "valid": False,
            "lots": 0,
            "capital_at_risk": 0,
            "position_value": 0,
            "kelly_fraction_used": 0,
            "reward_to_risk": 0,
            "reason": reason,
        }
