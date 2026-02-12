"""
SNIPER FRAMEWORK - NIFTY TRADING ENVIRONMENT (Gymnasium)
Custom RL environment for training PPO execution agent.

The agent learns WHEN and HOW to enter trades given confirmed meta-label signals.
This is the third filter layer:
  Raw Signal → Meta-Label → Shock Filter → RL Agent → Execute

Observation Space (normalized to [0,1]):
  - Last 20 bars: close, volume, ema_fast, ema_slow, ema_trend, rsi, atr, vol_ratio (160 features)
  - Current position: 0=flat, 1=long, -1=short (scaled to [0,1])
  - Unrealized P&L (scaled)
  - Bars held in current position (scaled)
  - Current drawdown % (scaled)
  - Regime one-hot encoded (4 values)
  - ML score of last signal
  - Shock detected flag

Action Space:
  0 = Hold/Stay flat
  1 = Enter/Stay Long
  2 = Enter/Stay Short

Reward Function:
  - Base: Net P&L after transaction costs
  - Penalty: -0.5 per trade opened (discourage overtrading)
  - Penalty: -2.0 if drawdown > 5%
  - Penalty: -10.0 if drawdown > 10% (hard stop)
  - Bonus: +1.0 for holding a winning trade
  - All scaled by 1/1000 for numerical stability
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict
import warnings


class NiftyTradingEnv(gym.Env):
    """
    Gymnasium environment for training RL trading agents.
    
    Usage:
        env = NiftyTradingEnv(df, initial_capital=100000)
        obs, info = env.reset()
        action = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000,
        lookback: int = 20,
        max_steps: int = None,
        cost_model=None,
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV + indicators from run_strategy()
            initial_capital: Starting capital in INR
            lookback: Number of historical bars in observation (default: 20)
            max_steps: Maximum steps per episode (default: len(df) - lookback)
            cost_model: ExecutionCostModel instance (default: create new)
        """
        super().__init__()
        
        # Data
        self.df = df.reset_index(drop=True)  # Reset to integer index
        self.lookback = lookback
        self.max_steps = max_steps or (len(df) - lookback - 1)
        
        # Capital tracking
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        
        # Position tracking
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.entry_idx = 0
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        
        # Cost model
        if cost_model is None:
            try:
                from execution.cost_model import ExecutionCostModel, TradeTicket
                self.cost_model = ExecutionCostModel()
                self.TradeTicket = TradeTicket
            except ImportError:
                warnings.warn("ExecutionCostModel not available, using zero costs")
                self.cost_model = None
                self.TradeTicket = None
        else:
            self.cost_model = cost_model
        
        # Episode state
        self.current_step = 0
        self.done = False
        self.total_trades = 0
        self.winning_trades = 0
        
        # Define observation space
        # 20 bars × 8 features + 8 scalars + 4 regime one-hot = 168 features
        obs_dim = (
            lookback * 8 +  # close, volume, ema_fast, ema_slow, ema_trend, rsi, atr, vol_ratio
            8 +             # position, unrealized_pnl, bars_held, drawdown, ml_score, shock, regime(2 remaining)
            4               # regime one-hot
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Normalization stats (computed during reset)
        self._norm_stats = {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to start of a new episode.
        Starts at a random bar in the data (after lookback period).
        
        Returns:
            observation: Initial observation vector
            info: Dictionary with episode metadata
        """
        super().reset(seed=seed)
        
        # Random starting point (after lookback, before end)
        if seed is not None:
            np.random.seed(seed)
        
        max_start = len(self.df) - self.max_steps - self.lookback
        self.current_step = np.random.randint(self.lookback, max(self.lookback + 1, max_start))
        
        # Reset state
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        self.done = False
        self.total_trades = 0
        self.winning_trades = 0
        
        # Compute normalization stats from visible data
        self._compute_normalization_stats()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Hold, 1=Long, 2=Short
        
        Returns:
            observation: Next observation vector
            reward: Reward for this step
            terminated: Episode ended (max steps or hard stop)
            truncated: Episode truncated (not used here)
            info: Additional info dict
        """
        if self.done:
            raise RuntimeError("Episode is done, call reset()")
        
        # Get current bar
        current_bar = self.df.iloc[self.current_step]
        current_price = current_bar["close"]
        
        reward = 0.0
        trade_opened = False
        
        # ── Execute action ─────────────────────────────────────────────
        # Action: 0=Hold, 1=Long, 2=Short
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # If changing position, close existing first
        if self.position != 0 and target_position != self.position:
            reward += self._close_position(current_price)
        
        # If entering new position
        if self.position == 0 and target_position != 0:
            self._open_position(target_position, current_price)
            trade_opened = True
        
        # ── Compute reward ─────────────────────────────────────────────
        # Base reward: unrealized P&L if holding
        if self.position != 0:
            self.unrealized_pnl = (
                (current_price - self.entry_price) * self.position * 75  # lot_size
            )
            reward += self.unrealized_pnl / 1000.0  # Scale for stability
            
            # Bonus for holding winning trade
            if self.unrealized_pnl > 0:
                reward += 1.0
        
        # Penalty for opening trade (discourage overtrading)
        if trade_opened:
            reward -= 0.5
        
        # Drawdown penalties
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if drawdown > 0.05:
            reward -= 2.0
        if drawdown > 0.10:
            reward -= 10.0
            self.done = True  # Hard stop
        
        # Scale all rewards
        reward = reward / 1000.0
        
        # ── Update state ───────────────────────────────────────────────
        self.current_step += 1
        if self.position != 0:
            self.bars_held += 1
        
        # Check if episode ends
        if self.current_step >= len(self.df) - 1:
            # Force close any open position at end
            if self.position != 0:
                reward += self._close_position(self.df.iloc[-1]["close"])
            self.done = True
        
        if self.current_step - self.lookback >= self.max_steps:
            self.done = True
        
        # Update peak capital
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.done, False, info
    
    def _open_position(self, direction: int, price: float):
        """Open a new position."""
        self.position = direction
        self.entry_price = price
        self.entry_idx = self.current_step
        self.bars_held = 0
        self.total_trades += 1
    
    def _close_position(self, exit_price: float) -> float:
        """
        Close current position and compute P&L with costs.
        
        Returns:
            Net P&L (scaled for reward)
        """
        if self.position == 0:
            return 0.0
        
        # Gross P&L
        lot_size = 75  # NIFTY lot size
        qty = lot_size
        gross_pnl = (exit_price - self.entry_price) * self.position * qty
        
        # Compute costs using ExecutionCostModel
        if self.cost_model and self.TradeTicket:
            try:
                ticket = self.TradeTicket(
                    direction=self.position,
                    entry_price=self.entry_price,
                    exit_price=exit_price,
                    lots=1,
                    lot_size=lot_size,
                    is_option=True,
                )
                cost_result = self.cost_model.net_pnl(ticket)
                total_cost = cost_result["total_cost"]
            except Exception:
                total_cost = gross_pnl * 0.01  # 1% fallback
        else:
            total_cost = gross_pnl * 0.01  # 1% fallback
        
        net_pnl = gross_pnl - total_cost
        
        # Update capital
        self.capital += net_pnl
        
        # Track wins
        if net_pnl > 0:
            self.winning_trades += 1
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        
        return net_pnl / 1000.0  # Scaled for reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector from current state.
        
        Features (all normalized to [0,1]):
          - Last 20 bars: close, volume, ema_fast, ema_slow, ema_trend, rsi, atr, vol_ratio
          - Position (0/1/-1 → 0/0.5/1)
          - Unrealized P&L (scaled)
          - Bars held (scaled)
          - Drawdown % (scaled)
          - Regime one-hot (4 values)
          - ML score (already 0-1)
          - Shock detected (0 or 1)
        """
        obs = []
        
        # ── Historical bars (last 20) ──────────────────────────────────
        start_idx = max(0, self.current_step - self.lookback)
        history = self.df.iloc[start_idx:self.current_step]
        
        # Pad if insufficient history
        if len(history) < self.lookback:
            padding = self.lookback - len(history)
            history = pd.concat([
                pd.DataFrame([history.iloc[0]] * padding),
                history
            ]).reset_index(drop=True)
        
        # Extract and normalize features
        for col in ["close", "volume", "ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "vol_ratio"]:
            values = history[col].fillna(0).values
            normalized = self._normalize(values, col)
            obs.extend(normalized)
        
        # ── Current state ──────────────────────────────────────────────
        # Position (map -1/0/1 to 0/0.5/1)
        obs.append((self.position + 1) / 2.0)
        
        # Unrealized P&L (scale to roughly [-1, 1] then clip to [0,1])
        pnl_scaled = np.clip((self.unrealized_pnl / 10000.0 + 1) / 2.0, 0, 1)
        obs.append(pnl_scaled)
        
        # Bars held (scale to [0,1], assume max ~100 bars)
        obs.append(min(self.bars_held / 100.0, 1.0))
        
        # Drawdown (0-1, where 1 = 100% drawdown)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        obs.append(min(drawdown, 1.0))
        
        # ── Current bar features ───────────────────────────────────────
        current = self.df.iloc[self.current_step]
        
        # Regime one-hot (4 states)
        regime = str(current.get("regime", "ranging"))
        regime_map = {"trending_bull": 0, "trending_bear": 1, "ranging": 2, "high_vol": 3}
        regime_idx = regime_map.get(regime, 2)
        regime_one_hot = [0.0] * 4
        regime_one_hot[regime_idx] = 1.0
        obs.extend(regime_one_hot)
        
        # ML score (already 0-1)
        ml_score = current.get("ml_score", 0.0)
        obs.append(min(max(ml_score, 0.0), 1.0))
        
        # Shock detected (0 or 1)
        shock = 1.0 if current.get("shock_detected", False) else 0.0
        obs.append(shock)
        
        return np.array(obs, dtype=np.float32)
    
    def _normalize(self, values: np.ndarray, col: str) -> np.ndarray:
        """
        Normalize array to [0,1] using min-max scaling.
        Uses cached stats computed in reset().
        """
        if col not in self._norm_stats:
            return np.zeros_like(values, dtype=np.float32)
        
        vmin, vmax = self._norm_stats[col]
        if vmax == vmin:
            return np.full_like(values, 0.5, dtype=np.float32)
        
        normalized = (values - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    def _compute_normalization_stats(self):
        """Compute min/max for each feature column for normalization."""
        cols = ["close", "volume", "ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "vol_ratio"]
        
        for col in cols:
            if col in self.df.columns:
                series = self.df[col].dropna()
                if len(series) > 0:
                    self._norm_stats[col] = (float(series.min()), float(series.max()))
                else:
                    self._norm_stats[col] = (0.0, 1.0)
            else:
                self._norm_stats[col] = (0.0, 1.0)
    
    def _get_info(self) -> dict:
        """Return episode info dict."""
        return {
            "step": self.current_step,
            "capital": self.capital,
            "position": self.position,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "drawdown": (self.peak_capital - self.capital) / self.peak_capital,
        }
    
    def render(self, mode="human"):
        """Print current state (for debugging)."""
        info = self._get_info()
        current_bar = self.df.iloc[self.current_step]
        
        print(f"\nStep {self.current_step}/{len(self.df)}")
        print(f"  Price: {current_bar['close']:.2f}")
        print(f"  Position: {['FLAT', 'LONG', 'SHORT'][self.position + 1]}")
        print(f"  Capital: ₹{self.capital:,.0f} (P&L: ₹{self.capital - self.initial_capital:,.0f})")
        print(f"  Trades: {info['total_trades']} (Win rate: {info['win_rate']:.1%})")
        print(f"  Drawdown: {info['drawdown']:.1%}")
        print(f"  Regime: {current_bar.get('regime', 'N/A')}")


# Example usage / testing
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing NiftyTradingEnv...")
    
    # Create dummy data
    from data.base_provider import SyntheticProvider
    from strategy.core_engine import run_strategy
    
    provider = SyntheticProvider()
    df = provider.fetch(symbol="NIFTY")
    df = run_strategy(df)
    
    # Create environment
    env = NiftyTradingEnv(df, initial_capital=100_000)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\n✓ Observation shape: {obs.shape}")
    print(f"✓ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"✓ Info: {info}")
    
    # Test a few random steps
    print("\n✓ Testing 10 random steps:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.4f}, capital={info['capital']:.0f}")
        if done:
            print("  Episode done!")
            break
    
    print("\n✓ Environment test passed!")
