"""
SNIPER FRAMEWORK - RL EXECUTOR
Inference interface for trained PPO agent.

Integrates with backtest engine to add RL-based entry confirmation:
  Meta-Label Signal → Shock Filter → RL Executor → Execute

The RL agent acts as a third filter layer. Even if meta-label approves a signal
and no shock is detected, the RL agent can still reject it based on learned patterns.

Usage:
    from rl.rl_executor import RLExecutor
    
    executor = RLExecutor()
    executor.load("models/rl_agent.zip")
    
    # During backtest, on each signal bar:
    obs = executor.build_observation(df, idx, position, unrealized_pnl, drawdown)
    action = executor.decide(obs)
    
    if action == 0:  # Hold
        skip_this_signal()
    elif action == 1:  # Long
        enter_long()
    elif action == 2:  # Short
        enter_short()
"""

import os
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple


# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("stable-baselines3 not available. RL executor will not work.")


class RLExecutor:
    """
    Inference wrapper for trained PPO agent.
    
    Acts as third-layer filter on meta-label signals:
      Raw Signal → Meta-Label → Shock Filter → RL Executor → Execute
    
    Usage:
        executor = RLExecutor()
        executor.load("models/rl_agent.zip")
        
        if executor.is_loaded:
            action = executor.decide(observation)
    """
    
    def __init__(self, lookback: int = 20):
        """
        Initialize RL executor.
        
        Args:
            lookback: Number of historical bars in observation (must match training)
        """
        self.model: Optional[PPO] = None
        self.is_loaded = False
        self.lookback = lookback
        self._norm_stats = {}
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        available = "available" if SB3_AVAILABLE else "not available"
        return f"RLExecutor(status={status}, sb3={available})"
    
    def load(self, path: str) -> bool:
        """
        Load trained PPO model from disk.
        
        Args:
            path: Path to model file (with or without .zip extension)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not SB3_AVAILABLE:
            print("  ⚠️  Cannot load RL model: stable-baselines3 not available")
            return False
        
        # Add .zip if not present
        if not path.endswith(".zip"):
            path = path + ".zip"
        
        if not os.path.exists(path):
            print(f"  ⚠️  RL model not found: {path}")
            return False
        
        try:
            self.model = PPO.load(path)
            self.is_loaded = True
            print(f"  ✓ RL model loaded from {path}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load RL model: {e}")
            self.is_loaded = False
            return False
    
    def decide(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Get action from trained PPO model.
        
        Args:
            observation: Normalized observation vector
            deterministic: Use deterministic policy (default: True for inference)
        
        Returns:
            Action (0=Hold, 1=Long, 2=Short)
        """
        if not self.is_loaded or self.model is None:
            # Fallback: return Hold
            return 0
        
        try:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return int(action)
        except Exception as e:
            warnings.warn(f"RL prediction failed: {e}")
            return 0  # Fallback to Hold
    
    def build_observation(
        self,
        df: pd.DataFrame,
        current_idx: int,
        position: int = 0,
        unrealized_pnl: float = 0.0,
        drawdown: float = 0.0,
        peak_capital: float = 100_000,
        capital: float = 100_000,
        bars_held: int = 0,
    ) -> np.ndarray:
        """
        Build observation vector from current market state.
        
        This must match the observation space in NiftyTradingEnv exactly.
        
        Args:
            df: Full DataFrame with OHLCV + indicators
            current_idx: Current bar index
            position: Current position (0/1/-1)
            unrealized_pnl: Current unrealized P&L
            drawdown: Current drawdown (0-1)
            peak_capital: Peak capital reached
            capital: Current capital
            bars_held: Bars held in current position
        
        Returns:
            Normalized observation vector
        """
        # Compute normalization stats if not cached
        if not self._norm_stats:
            self._compute_normalization_stats(df)
        
        obs = []
        
        # ── Historical bars (last 20) ──────────────────────────────────
        start_idx = max(0, current_idx - self.lookback)
        history = df.iloc[start_idx:current_idx]
        
        # Pad if insufficient
        if len(history) < self.lookback:
            padding = self.lookback - len(history)
            first_row = df.iloc[0]
            pad_df = pd.DataFrame([first_row] * padding)
            history = pd.concat([pad_df, history]).reset_index(drop=True)
        
        # Extract and normalize features
        for col in ["close", "volume", "ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "vol_ratio"]:
            values = history[col].fillna(0).values
            normalized = self._normalize(values, col)
            obs.extend(normalized)
        
        # ── Current state ──────────────────────────────────────────────
        # Position (map -1/0/1 to 0/0.5/1)
        obs.append((position + 1) / 2.0)
        
        # Unrealized P&L (scale to [-1,1] then shift to [0,1])
        pnl_scaled = np.clip((unrealized_pnl / 10000.0 + 1) / 2.0, 0, 1)
        obs.append(pnl_scaled)
        
        # Bars held (scale to [0,1], max ~100 bars)
        obs.append(min(bars_held / 100.0, 1.0))
        
        # Drawdown (0-1)
        obs.append(min(drawdown, 1.0))
        
        # ── Current bar features ───────────────────────────────────────
        current = df.iloc[current_idx]
        
        # Regime one-hot
        regime = str(current.get("regime", "ranging"))
        regime_map = {"trending_bull": 0, "trending_bear": 1, "ranging": 2, "high_vol": 3}
        regime_idx = regime_map.get(regime, 2)
        regime_one_hot = [0.0] * 4
        regime_one_hot[regime_idx] = 1.0
        obs.extend(regime_one_hot)
        
        # ML score
        ml_score = current.get("ml_score", 0.0)
        obs.append(min(max(ml_score, 0.0), 1.0))
        
        # Shock detected
        shock = 1.0 if current.get("shock_detected", False) else 0.0
        obs.append(shock)
        
        return np.array(obs, dtype=np.float32)
    
    def _normalize(self, values: np.ndarray, col: str) -> np.ndarray:
        """Normalize array to [0,1] using cached min-max stats."""
        if col not in self._norm_stats:
            return np.zeros_like(values, dtype=np.float32)
        
        vmin, vmax = self._norm_stats[col]
        if vmax == vmin:
            return np.full_like(values, 0.5, dtype=np.float32)
        
        normalized = (values - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    def _compute_normalization_stats(self, df: pd.DataFrame):
        """Compute min/max for each feature column."""
        cols = ["close", "volume", "ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "vol_ratio"]
        
        for col in cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    self._norm_stats[col] = (float(series.min()), float(series.max()))
                else:
                    self._norm_stats[col] = (0.0, 1.0)
            else:
                self._norm_stats[col] = (0.0, 1.0)


# Global singleton instance
_executor = None


def get_rl_executor(force_new: bool = False) -> RLExecutor:
    """
    Get or create the global RL executor instance.
    
    Args:
        force_new: Create new executor instead of using cached one
    
    Returns:
        RLExecutor instance
    """
    global _executor
    
    if _executor is None or force_new:
        _executor = RLExecutor()
        
        # Try to auto-load from default path
        model_path = "models/rl_agent.zip"
        if os.path.exists(model_path):
            _executor.load(model_path)
    
    return _executor


# For testing
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing RLExecutor...")
    
    from data.base_provider import SyntheticProvider
    from strategy.core_engine import run_strategy
    
    # Create test data
    provider = SyntheticProvider()
    df = provider.fetch(symbol="NIFTY")
    df = run_strategy(df)
    
    # Create executor
    executor = RLExecutor()
    print(f"\n✓ Created: {executor}")
    
    # Test observation building
    obs = executor.build_observation(
        df, current_idx=100, position=0,
        unrealized_pnl=0, drawdown=0,
    )
    
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test decision (will return 0 if model not loaded)
    action = executor.decide(obs)
    print(f"✓ Decision: {['Hold', 'Long', 'Short'][action]}")
    
    print("\n✓ RLExecutor test passed!")
