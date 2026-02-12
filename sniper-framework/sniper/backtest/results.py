"""Backtest results and metrics."""

from datetime import datetime
from typing import Any

import numpy as np


class BacktestResults:
    """
    Backtest results: trades, equity curve, key metrics.
    """

    def __init__(
        self,
        trades: list[dict],
        equity_curve: list[tuple[datetime, float]],
        initial_capital: float,
        fills: list[dict] | None = None,
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.fills = fills or []

    def total_trades(self) -> int:
        """Number of completed trades."""
        return len(self.trades)

    def winning_trades(self) -> int:
        """Number of profitable trades."""
        return sum(1 for t in self.trades if t.get("pnl", 0) > 0)

    def losing_trades(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self.trades if t.get("pnl", 0) < 0)

    def accuracy(self) -> float:
        """Win rate (0-1)."""
        n = self.total_trades()
        return self.winning_trades() / n if n else 0.0

    def total_pnl(self) -> float:
        """Total realized P&L."""
        return sum(t.get("pnl", 0) for t in self.trades)

    def final_equity(self) -> float:
        """Final portfolio value."""
        if not self.equity_curve:
            return self.initial_capital
        return self.equity_curve[-1][1]

    def returns(self) -> list[float]:
        """Period returns from equity curve."""
        if len(self.equity_curve) < 2:
            return []
        vals = [e[1] for e in self.equity_curve]
        return [
            (vals[i] - vals[i - 1]) / vals[i - 1]
            for i in range(1, len(vals))
            if vals[i - 1] != 0
        ]

    def sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: float = 252.0) -> float:
        """Annualized Sharpe ratio."""
        rets = self.returns()
        if not rets:
            return 0.0
        arr = np.array(rets)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        excess = mean - risk_free_rate / periods_per_year
        return excess / std * np.sqrt(periods_per_year)

    def max_drawdown(self) -> float:
        """Maximum drawdown (0-1)."""
        if not self.equity_curve:
            return 0.0
        vals = [e[1] for e in self.equity_curve]
        peak = vals[0]
        max_dd = 0.0
        for v in vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    def verdict(self) -> str:
        """Qualification verdict."""
        if self.total_trades() < 10:
            return "INSUFFICIENT_TRADES"
        if self.accuracy() < 0.5:
            return "LOW_ACCURACY"
        if self.sharpe_ratio() < 1.0:
            return "LOW_SHARPE"
        if self.max_drawdown() > 0.15:
            return "HIGH_DRAWDOWN"
        return "PASS"

    def to_dict(self) -> dict:
        """Export metrics as dict."""
        return {
            "total_trades": self.total_trades(),
            "winning_trades": self.winning_trades(),
            "losing_trades": self.losing_trades(),
            "accuracy": round(self.accuracy(), 4),
            "total_pnl": round(self.total_pnl(), 2),
            "final_equity": round(self.final_equity(), 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 4),
            "max_drawdown": round(self.max_drawdown(), 4),
            "profit_factor": round(self.profit_factor(), 4) if self.profit_factor() != float("inf") else "inf",
            "verdict": self.verdict(),
        }
