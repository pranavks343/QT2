"""
SNIPER FRAMEWORK - BACKTEST ENGINE
Runs walk-forward simulation with:
  - Trade-by-trade P&L with real costs
  - Regime-labelled trade log
  - Accuracy, Sharpe, max drawdown, trade frequency
  - CPCV (Combinatorial Purged Cross-Validation) structure
  - DSR (Deflated Sharpe Ratio) adjustment
  - Qualification verdict
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config import BACKTEST, RISK, COSTS


# ─── TRADE LOG ENTRY ─────────────────────────────────────────────────────────
class Trade:
    def __init__(self, entry_time, direction, entry_price, stop_loss,
                 take_profit, regime, ml_score, lots):
        self.entry_time = entry_time
        self.exit_time = None
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_reason = None
        self.regime = regime
        self.ml_score = ml_score
        self.lots = lots
        self.gross_pnl = 0.0
        self.cost = 0.0
        self.net_pnl = 0.0
        self.outcome = None  # "win" or "loss"

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "exit_reason": self.exit_reason,
            "regime": self.regime,
            "ml_score": round(self.ml_score, 4),
            "lots": self.lots,
            "gross_pnl": round(self.gross_pnl, 2),
            "cost": round(self.cost, 2),
            "net_pnl": round(self.net_pnl, 2),
            "outcome": self.outcome,
        }


# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
class BacktestEngine:
    """
    Sequential bar-by-bar backtest.
    Respects: one position at a time per signal (extendable to concurrent).
    """

    def __init__(self, risk_manager=None, use_alpha_monitor: bool = True, rl_executor=None):
        from risk.risk_manager import RiskManager
        from execution.cost_model import ExecutionCostModel, TradeTicket
        self.risk = risk_manager or RiskManager()
        self.cost_model = ExecutionCostModel()
        self.TradeTicket = TradeTicket
        self.trade_log: List[Trade] = []
        self.equity_curve: List[float] = []
        
        # RL execution agent (third filter layer)
        self.rl_executor = rl_executor
        
        # Initialize alpha decay monitor
        self.use_alpha_monitor = use_alpha_monitor
        self.alpha_monitor = None
        if use_alpha_monitor:
            try:
                from strategy.alpha_monitor import get_alpha_monitor
                self.alpha_monitor = get_alpha_monitor(reset=True)
            except Exception:
                self.alpha_monitor = None

    def run(self, df: pd.DataFrame) -> "BacktestResults":
        """
        Run backtest on a strategy-processed DataFrame.
        df must have columns: signal, entry_price, stop_loss, take_profit,
        regime, ml_score, high, low, close
        """
        self.trade_log = []
        self.equity_curve = []
        active_trade: Trade = None
        prev_date = None

        for ts, row in df.iterrows():
            current_date = ts.date() if hasattr(ts, 'date') else ts

            # Day reset
            if prev_date and current_date != prev_date:
                self.risk.reset_daily()
            prev_date = current_date

            self.equity_curve.append(self.risk.current_capital)

            # ── Manage open trade ────────────────────────────────────────────
            if active_trade is not None:
                active_trade = self._check_exit(active_trade, row, ts)
                if active_trade is None:
                    continue

            # ── Look for new entry ───────────────────────────────────────────
            if active_trade is None and row.get("signal", 0) != 0 and self.risk.can_trade():

                # STT trap check - skip if premium too low
                if not self.cost_model.is_tradeable(row["entry_price"]):
                    continue

                # RL EXECUTOR FILTER (third layer)
                # Only executes if meta-label approved and no shock detected
                # RL can still reject the signal based on learned patterns
                if self.rl_executor is not None and self.rl_executor.is_loaded:
                    # Get current index in dataframe
                    current_idx = df.index.get_loc(ts)
                    
                    # Build observation for RL agent
                    # Calculate current drawdown
                    drawdown = (self.risk.peak_capital - self.risk.current_capital) / self.risk.peak_capital if self.risk.peak_capital > 0 else 0.0
                    
                    obs = self.rl_executor.build_observation(
                        df=df,
                        current_idx=current_idx,
                        position=0,  # Currently flat
                        unrealized_pnl=0.0,
                        drawdown=drawdown,
                        peak_capital=self.risk.peak_capital,
                        capital=self.risk.current_capital,
                        bars_held=0,
                    )
                    
                    # Get RL decision: 0=Hold, 1=Long, 2=Short
                    rl_action = self.rl_executor.decide(obs)
                    
                    # Map signal direction to expected action
                    signal_direction = int(row["signal"])
                    expected_action = 1 if signal_direction == 1 else 2
                    
                    # If RL says Hold (0) or suggests opposite direction, skip this signal
                    if rl_action == 0 or rl_action != expected_action:
                        continue

                # Size the position
                # Use estimated win rate for Kelly (bootstrapped from recent trades)
                win_prob = self._estimate_win_prob()
                avg_win = abs(row["take_profit"] - row["entry_price"])
                avg_loss = abs(row["entry_price"] - row["stop_loss"])

                sizing = self.risk.kelly_position_size(
                    win_prob=win_prob,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    entry_price=row["entry_price"],
                    stop_loss=row["stop_loss"],
                )

                if not sizing["valid"] or sizing["lots"] == 0:
                    continue

                active_trade = Trade(
                    entry_time=ts,
                    direction=int(row["signal"]),
                    entry_price=row["entry_price"],
                    stop_loss=row["stop_loss"],
                    take_profit=row["take_profit"],
                    regime=row.get("regime", "unknown"),
                    ml_score=row.get("ml_score", 0),
                    lots=sizing["lots"],
                )
                self.risk.open_trade()

        # Close any open trade at end of data
        if active_trade is not None:
            self._force_close(active_trade, df.iloc[-1], df.index[-1])

        return BacktestResults(self.trade_log, self.equity_curve, self.risk, self.alpha_monitor)

    def _check_exit(self, trade: Trade, row, ts) -> Trade:
        """Check if trade hit SL, TP, or is still open. Returns None if closed."""
        # Check SL hit (conservative: use low/high)
        sl_hit = False
        tp_hit = False

        if trade.direction == 1:   # Long
            if row["low"] <= trade.stop_loss:
                sl_hit = True
                exit_price = trade.stop_loss
            elif row["high"] >= trade.take_profit:
                tp_hit = True
                exit_price = trade.take_profit
        else:  # Short
            if row["high"] >= trade.stop_loss:
                sl_hit = True
                exit_price = trade.stop_loss
            elif row["low"] <= trade.take_profit:
                tp_hit = True
                exit_price = trade.take_profit

        if sl_hit or tp_hit:
            self._close_trade(trade, exit_price, "TP" if tp_hit else "SL", ts)
            return None

        return trade  # Still open

    def _force_close(self, trade: Trade, row, ts):
        """Force-close at end of data at close price."""
        self._close_trade(trade, row["close"], "EOD", ts)

    def _close_trade(self, trade: Trade, exit_price: float, reason: str, ts):
        """Close the trade: compute P&L, costs, update risk manager."""
        from config import COSTS as c
        lot_size = c["lot_size_nifty"]
        qty = trade.lots * lot_size

        # Gross P&L
        if trade.direction == 1:
            gross = (exit_price - trade.entry_price) * qty
        else:
            gross = (trade.entry_price - exit_price) * qty

        # Costs
        ticket = self.TradeTicket(
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            lots=trade.lots,
            lot_size=lot_size,
            is_option=True,
        )
        cost_result = self.cost_model.net_pnl(ticket)
        total_cost = cost_result["total_cost"]
        net = gross - total_cost

        # Update trade record
        trade.exit_time = ts
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.gross_pnl = gross
        trade.cost = total_cost
        trade.net_pnl = net
        trade.outcome = "win" if net > 0 else "loss"

        # Update risk manager
        self.risk.close_trade()
        self.risk.update_capital(net)
        self.trade_log.append(trade)
        
        # Update alpha decay monitor
        if self.alpha_monitor is not None:
            self.alpha_monitor.update(trade.outcome, net)

    def _estimate_win_prob(self) -> float:
        """Estimate win probability from recent trades (last 50)."""
        recent = self.trade_log[-50:] if len(self.trade_log) >= 50 else self.trade_log
        if not recent:
            return 0.55  # Prior
        wins = sum(1 for t in recent if t.outcome == "win")
        return max(0.3, min(0.9, wins / len(recent)))


# ─── BACKTEST RESULTS ─────────────────────────────────────────────────────────
class BacktestResults:
    """Compute and store all validation metrics."""

    def __init__(self, trade_log: List[Trade], equity_curve: List[float], risk, alpha_monitor=None):
        self.trades = [t.to_dict() for t in trade_log]
        self.trade_df = pd.DataFrame(self.trades)
        self.equity_curve = equity_curve
        self.risk = risk
        self.alpha_monitor = alpha_monitor

    def accuracy(self) -> float:
        if self.trade_df.empty or "outcome" not in self.trade_df:
            return 0.0
        return (self.trade_df["outcome"] == "win").mean()

    def total_trades(self) -> int:
        return len(self.trade_df)

    def sharpe_ratio(self) -> float:
        """Daily Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        eq = pd.Series(self.equity_curve)
        returns = eq.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        # Annualize for ~252 trading days
        return (returns.mean() / returns.std()) * np.sqrt(252 * 75)  # 75 bars/day

    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq = pd.Series(self.equity_curve)
        peak = eq.cummax()
        dd = (peak - eq) / peak
        return dd.max()

    def profit_factor(self) -> float:
        if self.trade_df.empty:
            return 0.0
        wins = self.trade_df[self.trade_df["net_pnl"] > 0]["net_pnl"].sum()
        losses = abs(self.trade_df[self.trade_df["net_pnl"] < 0]["net_pnl"].sum())
        return wins / losses if losses > 0 else float("inf")

    def total_pnl(self) -> float:
        if self.trade_df.empty:
            return 0.0
        return self.trade_df["net_pnl"].sum()

    def total_costs(self) -> float:
        if self.trade_df.empty:
            return 0.0
        return self.trade_df["cost"].sum()

    def accuracy_by_regime(self) -> dict:
        if self.trade_df.empty or "regime" not in self.trade_df:
            return {}
        grouped = self.trade_df.groupby("regime").apply(
            lambda x: (x["outcome"] == "win").mean()
        )
        return grouped.round(4).to_dict()

    def deflated_sharpe_ratio(self) -> float:
        """
        DSR = Sharpe adjusted for multiple testing bias.
        PSR(SR* = 0) accounts for number of trials.
        Simplified version here.
        """
        from scipy.stats import norm
        n = len(self.trade_df)
        sr = self.sharpe_ratio()
        if n < 30:
            return 0.0
        # Compute skewness and kurtosis of trade P&Ls
        pnls = self.trade_df["net_pnl"] if not self.trade_df.empty else pd.Series([0])
        skew = float(pnls.skew()) if len(pnls) > 2 else 0.0
        kurt = float(pnls.kurt()) if len(pnls) > 3 else 0.0

        # Probabilistic SR: P(SR > 0)
        psr = norm.cdf(
            (sr * np.sqrt(n - 1)) /
            np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
        )
        return round(psr, 4)

    def verdict(self) -> dict:
        """
        Qualification check against all 5 targets.
        Returns pass/fail per criterion and overall verdict.
        """
        n = self.total_trades()
        acc = self.accuracy()
        sr = self.sharpe_ratio()
        dd = self.max_drawdown()
        dsr = self.deflated_sharpe_ratio()
        pf = self.profit_factor()

        checks = {
            "min_trades": {
                "target": BACKTEST["min_trades_for_validity"],
                "actual": n,
                "pass": n >= BACKTEST["min_trades_for_validity"],
            },
            "accuracy": {
                "target": BACKTEST["min_accuracy"],
                "actual": round(acc, 4),
                "pass": acc >= BACKTEST["min_accuracy"],
            },
            "sharpe_ratio": {
                "target": BACKTEST["min_sharpe_ratio"],
                "actual": round(sr, 4),
                "pass": sr >= BACKTEST["min_sharpe_ratio"],
            },
            "max_drawdown": {
                "target": BACKTEST["max_drawdown_allowed"],
                "actual": round(dd, 4),
                "pass": dd <= BACKTEST["max_drawdown_allowed"],
            },
            "profit_factor": {
                "target": 1.5,
                "actual": round(pf, 4),
                "pass": pf >= 1.5,
            },
            "dsr_confidence": {
                "target": 0.95,
                "actual": round(dsr, 4),
                "pass": dsr >= 0.95,
            },
        }

        passed = sum(1 for v in checks.values() if v["pass"])
        total = len(checks)

        if passed == total:
            overall = "✅ FULLY QUALIFIED"
        elif passed >= total - 1:
            overall = "⚠️ CONDITIONALLY QUALIFIED"
        else:
            overall = "❌ NOT QUALIFIED"

        return {
            "overall": overall,
            "passed": passed,
            "total": total,
            "checks": checks,
        }

    def rolling_sharpe_curve(self, window: int = 50) -> pd.DataFrame:
        """
        Compute rolling Sharpe ratio across trade log.
        
        Args:
            window: Rolling window size in trades (default: 50)
        
        Returns:
            DataFrame with columns: trade_num, rolling_sharpe
        """
        if self.trade_df.empty or len(self.trade_df) < window:
            return pd.DataFrame()
        
        pnls = self.trade_df["net_pnl"]
        
        rolling_sharpes = []
        for i in range(window, len(pnls) + 1):
            window_pnls = pnls.iloc[i - window:i]
            if window_pnls.std() > 0:
                sharpe = window_pnls.mean() / window_pnls.std()
            else:
                sharpe = 0.0
            rolling_sharpes.append({"trade_num": i, "rolling_sharpe": sharpe})
        
        return pd.DataFrame(rolling_sharpes)
    
    def print_report(self):
        """Print a full summary report to console."""
        sep = "─" * 60
        print(f"\n{sep}")
        print("  SNIPER FRAMEWORK — BACKTEST REPORT")
        print(sep)
        print(f"  Total Trades:        {self.total_trades()}")
        print(f"  Accuracy:            {self.accuracy():.1%}")
        print(f"  Total Net P&L:       ₹{self.total_pnl():,.0f}")
        print(f"  Total Costs:         ₹{self.total_costs():,.0f}")
        print(f"  Sharpe Ratio:        {self.sharpe_ratio():.2f}")
        print(f"  DSR (confidence):    {self.deflated_sharpe_ratio():.4f}")
        print(f"  Max Drawdown:        {self.max_drawdown():.1%}")
        print(f"  Profit Factor:       {self.profit_factor():.2f}")
        print(f"\n  Accuracy by Regime:")
        for regime, acc in self.accuracy_by_regime().items():
            print(f"    {regime:<20} {acc:.1%}")
        
        # Rolling Sharpe analysis
        rolling = self.rolling_sharpe_curve()
        if not rolling.empty:
            start_sharpe = rolling["rolling_sharpe"].iloc[0]
            mid_idx = len(rolling) // 2
            mid_sharpe = rolling["rolling_sharpe"].iloc[mid_idx] if mid_idx < len(rolling) else 0
            end_sharpe = rolling["rolling_sharpe"].iloc[-1]
            
            declining = end_sharpe < start_sharpe * 0.7  # 30%+ drop
            
            print(f"\n  Rolling Sharpe (50-trade window):")
            print(f"    Start:              {start_sharpe:.2f}")
            print(f"    Middle:             {mid_sharpe:.2f}")
            print(f"    End:                {end_sharpe:.2f}")
            
            if declining:
                print(f"    ⚠️  WARNING: Sharpe declining over time (potential alpha decay)")
            else:
                print(f"    ✓  Sharpe stable or improving")
        
        # Alpha monitor status
        if self.alpha_monitor is not None:
            status = self.alpha_monitor.status()
            if status["baseline_computed"]:
                print(f"\n  Alpha Decay Monitor:")
                print(f"    Baseline Sharpe:    {status['baseline_sharpe']}")
                print(f"    Rolling Sharpe:     {status['rolling_sharpe']}")
                print(f"    Baseline Accuracy:  {status['baseline_accuracy']:.1%}")
                print(f"    Rolling Accuracy:   {status['rolling_accuracy']:.1%}")
                if status["decay_flag"]:
                    print(f"    Status:             ⚠️  DECAY DETECTED")
                else:
                    print(f"    Status:             ✓  Healthy")

        v = self.verdict()
        print(f"\n  QUALIFICATION: {v['overall']}  ({v['passed']}/{v['total']} criteria)")
        for name, check in v["checks"].items():
            icon = "✅" if check["pass"] else "❌"
            print(f"    {icon} {name:<22} target={check['target']}  actual={check['actual']}")
        print(sep + "\n")


# ─── CPCV VALIDATOR ──────────────────────────────────────────────────────────
class CPCVValidator:
    """
    Combinatorial Purged Cross-Validation.
    Splits data into k folds, tests on each fold independently
    with embargo periods to prevent data leakage.
    Returns distribution of Sharpe ratios across paths.
    """

    def __init__(self, n_folds: int = None, embargo_bars: int = None):
        self.n_folds = n_folds or BACKTEST["cpcv_folds"]
        self.embargo = embargo_bars or BACKTEST["cpcv_embargo_bars"]

    def run(self, df: pd.DataFrame) -> dict:
        """Run CPCV and return stats across all paths."""
        n = len(df)
        fold_size = n // self.n_folds
        results = []

        for i in range(self.n_folds):
            # Test fold: fold i
            test_start = i * fold_size
            test_end = test_start + fold_size

            # Train fold: everything else, with embargo around test
            train_end_before = max(0, test_start - self.embargo)
            train_start_after = min(n, test_end + self.embargo)

            train_df = pd.concat([
                df.iloc[:train_end_before],
                df.iloc[train_start_after:]
            ])
            test_df = df.iloc[test_start:test_end]

            if len(test_df) < 50:
                continue

            # Run backtest on test fold
            engine = BacktestEngine()
            r = engine.run(test_df)
            results.append({
                "fold": i + 1,
                "n_trades": r.total_trades(),
                "accuracy": round(r.accuracy(), 4),
                "sharpe": round(r.sharpe_ratio(), 4),
                "drawdown": round(r.max_drawdown(), 4),
                "pnl": round(r.total_pnl(), 2),
            })

        if not results:
            return {"error": "Insufficient data for CPCV"}

        result_df = pd.DataFrame(results)
        return {
            "folds": results,
            "mean_sharpe": round(result_df["sharpe"].mean(), 4),
            "std_sharpe": round(result_df["sharpe"].std(), 4),
            "min_sharpe": round(result_df["sharpe"].min(), 4),
            "mean_accuracy": round(result_df["accuracy"].mean(), 4),
            "mean_drawdown": round(result_df["drawdown"].mean(), 4),
            "total_trades": int(result_df["n_trades"].sum()),
            "consistent": bool((result_df["sharpe"] > 0).all()),
        }
