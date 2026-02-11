"""
SNIPER FRAMEWORK - ALPHA DECAY MONITOR
Tracks rolling performance metrics to detect strategy degradation in live trading.

Monitors:
  - Rolling 50-trade Sharpe ratio
  - Rolling 50-trade accuracy
  - Compares against baseline (first 200 trades)

Raises warning if:
  - Rolling Sharpe drops below 50% of baseline, OR
  - Rolling accuracy drops below 65%

Use in live trading to know when to:
  - Retrain models
  - Adjust parameters
  - Pause trading

Circular buffer design for memory efficiency.
"""

import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Optional, List


class AlphaDecayMonitor:
    """
    Monitors rolling strategy performance to detect alpha decay.
    
    Usage:
        monitor = AlphaDecayMonitor(window=50, baseline_trades=200)
        
        # After each trade closes:
        monitor.update(trade_outcome="win", trade_pnl=1500.0)
        
        # Check status:
        status = monitor.status()
        if status["decay_flag"]:
            print("⚠️ ALPHA DECAY DETECTED")
    """
    
    def __init__(self, window: int = 50, baseline_trades: int = 200,
                 sharpe_decay_threshold: float = 0.5, accuracy_threshold: float = 0.65):
        """
        Initialize alpha decay monitor.
        
        Args:
            window: Rolling window size for metrics (default: 50 trades)
            baseline_trades: Number of trades to establish baseline (default: 200)
            sharpe_decay_threshold: Alert if rolling Sharpe < threshold × baseline (default: 0.5)
            accuracy_threshold: Alert if rolling accuracy < threshold (default: 0.65)
        """
        self.window = window
        self.baseline_trades = baseline_trades
        self.sharpe_decay_threshold = sharpe_decay_threshold
        self.accuracy_threshold = accuracy_threshold
        
        # Circular buffers for rolling window
        self.outcomes: deque = deque(maxlen=window)
        self.pnls: deque = deque(maxlen=window)
        
        # Baseline metrics (computed after baseline_trades)
        self.baseline_sharpe: Optional[float] = None
        self.baseline_accuracy: Optional[float] = None
        self.baseline_computed = False
        
        # Full trade history (for baseline computation)
        self.all_outcomes: List[str] = []
        self.all_pnls: List[float] = []
        
        # Status
        self.decay_flag = False
        self.total_trades = 0
    
    def __repr__(self) -> str:
        return (f"AlphaDecayMonitor(window={self.window}, "
                f"trades={self.total_trades}, decay={self.decay_flag})")
    
    def update(self, trade_outcome: str, trade_pnl: float) -> None:
        """
        Update monitor with a closed trade.
        
        Args:
            trade_outcome: "win" or "loss"
            trade_pnl: Net P&L of the trade
        """
        # Add to circular buffer
        self.outcomes.append(trade_outcome)
        self.pnls.append(trade_pnl)
        
        # Add to full history
        self.all_outcomes.append(trade_outcome)
        self.all_pnls.append(trade_pnl)
        
        self.total_trades += 1
        
        # Compute baseline after baseline_trades
        if not self.baseline_computed and len(self.all_pnls) >= self.baseline_trades:
            self._compute_baseline()
        
        # Check for decay (only after baseline established)
        if self.baseline_computed:
            self._check_decay()
    
    def _compute_baseline(self) -> None:
        """Compute baseline metrics from first N trades."""
        baseline_pnls = self.all_pnls[:self.baseline_trades]
        baseline_outcomes = self.all_outcomes[:self.baseline_trades]
        
        # Baseline Sharpe
        pnl_series = pd.Series(baseline_pnls)
        if pnl_series.std() > 0:
            self.baseline_sharpe = pnl_series.mean() / pnl_series.std()
        else:
            self.baseline_sharpe = 0.0
        
        # Baseline accuracy
        wins = sum(1 for o in baseline_outcomes if o == "win")
        self.baseline_accuracy = wins / len(baseline_outcomes)
        
        self.baseline_computed = True
        
        print(f"\n  📊 Baseline established ({self.baseline_trades} trades):")
        print(f"     Sharpe: {self.baseline_sharpe:.2f}")
        print(f"     Accuracy: {self.baseline_accuracy:.1%}\n")
    
    def _check_decay(self) -> None:
        """Check if current rolling metrics indicate alpha decay."""
        if len(self.pnls) < self.window:
            # Not enough data yet
            self.decay_flag = False
            return
        
        # Compute rolling metrics
        rolling_sharpe = self._compute_rolling_sharpe()
        rolling_accuracy = self._compute_rolling_accuracy()
        
        # Check decay conditions
        sharpe_decayed = (rolling_sharpe < self.baseline_sharpe * self.sharpe_decay_threshold)
        accuracy_decayed = (rolling_accuracy < self.accuracy_threshold)
        
        # Update flag
        prev_flag = self.decay_flag
        self.decay_flag = sharpe_decayed or accuracy_decayed
        
        # Log warning if decay just detected
        if self.decay_flag and not prev_flag:
            print("\n" + "=" * 70)
            print("  ⚠️  ALPHA DECAY WARNING")
            print("=" * 70)
            print(f"  Rolling Sharpe:   {rolling_sharpe:.2f}  "
                  f"(baseline: {self.baseline_sharpe:.2f}, "
                  f"threshold: {self.baseline_sharpe * self.sharpe_decay_threshold:.2f})")
            print(f"  Rolling Accuracy: {rolling_accuracy:.1%}  "
                  f"(threshold: {self.accuracy_threshold:.1%})")
            print("\n  Recommendations:")
            print("    • Review recent trades for anomalies")
            print("    • Consider retraining models")
            print("    • Check if market regime has shifted")
            print("    • May need to pause trading and re-optimize")
            print("=" * 70 + "\n")
    
    def _compute_rolling_sharpe(self) -> float:
        """Compute Sharpe ratio over rolling window."""
        if len(self.pnls) < 2:
            return 0.0
        
        pnl_series = pd.Series(list(self.pnls))
        
        if pnl_series.std() == 0:
            return 0.0
        
        return pnl_series.mean() / pnl_series.std()
    
    def _compute_rolling_accuracy(self) -> float:
        """Compute win rate over rolling window."""
        if len(self.outcomes) == 0:
            return 0.0
        
        wins = sum(1 for o in self.outcomes if o == "win")
        return wins / len(self.outcomes)
    
    def status(self) -> Dict:
        """
        Get current monitoring status.
        
        Returns:
            Dictionary with:
              - total_trades: Total trades processed
              - rolling_sharpe: Current rolling Sharpe (or None)
              - baseline_sharpe: Baseline Sharpe (or None)
              - rolling_accuracy: Current rolling accuracy (or None)
              - baseline_accuracy: Baseline accuracy (or None)
              - decay_flag: Boolean indicating decay detected
              - baseline_computed: Boolean indicating baseline ready
        """
        rolling_sharpe = self._compute_rolling_sharpe() if len(self.pnls) >= self.window else None
        rolling_accuracy = self._compute_rolling_accuracy() if len(self.outcomes) >= self.window else None
        
        return {
            "total_trades": self.total_trades,
            "rolling_sharpe": round(rolling_sharpe, 4) if rolling_sharpe is not None else None,
            "baseline_sharpe": round(self.baseline_sharpe, 4) if self.baseline_sharpe is not None else None,
            "rolling_accuracy": round(rolling_accuracy, 4) if rolling_accuracy is not None else None,
            "baseline_accuracy": round(self.baseline_accuracy, 4) if self.baseline_accuracy is not None else None,
            "decay_flag": self.decay_flag,
            "baseline_computed": self.baseline_computed,
        }
    
    def reset(self) -> None:
        """Reset monitor (useful for starting new trading session)."""
        self.outcomes.clear()
        self.pnls.clear()
        self.all_outcomes.clear()
        self.all_pnls.clear()
        self.baseline_sharpe = None
        self.baseline_accuracy = None
        self.baseline_computed = False
        self.decay_flag = False
        self.total_trades = 0


# Global instance
_monitor = None

def get_alpha_monitor(reset: bool = False) -> AlphaDecayMonitor:
    """
    Get or create the global alpha decay monitor instance.
    
    Args:
        reset: Reset the monitor instead of using cached state
    
    Returns:
        AlphaDecayMonitor instance
    """
    global _monitor
    
    if _monitor is None or reset:
        _monitor = AlphaDecayMonitor()
    
    return _monitor
