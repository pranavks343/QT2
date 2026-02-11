"""
SNIPER FRAMEWORK - HMM REGIME DETECTION
Hidden Markov Model for market regime classification.

Replaces simple threshold-based regime detection with a probabilistic model
that learns regime transitions from historical data patterns.

Regimes (4 hidden states):
  - trending_bull: Uptrend with consistent directional movement
  - trending_bear: Downtrend with consistent directional movement  
  - ranging: Sideways/choppy with low directional bias
  - high_vol: Elevated volatility regime (any direction)

Features used for training:
  - Returns (price momentum)
  - Realized volatility (rolling std)
  - Volume ratio (vs moving average)

Falls back to simple rule-based detection if hmmlearn not installed.
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Optional

# Check for hmmlearn availability
try:
    from hmmlearn import hmm
    import joblib
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn(
        "hmmlearn not installed. Install with: pip install hmmlearn\n"
        "Falling back to simple regime detection."
    )


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection with 4 states.
    
    Usage:
        detector = HMMRegimeDetector(n_states=4)
        detector.fit(df)  # Train on historical data
        df = detector.predict(df)  # Add regime labels
        detector.save("models/hmm_regime.pkl")
    """
    
    def __init__(self, n_states: int = 4, random_state: int = 42):
        """
        Initialize HMM regime detector.
        
        Args:
            n_states: Number of hidden states (default: 4)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.is_fitted = False
        self.regime_map = {
            0: "ranging",
            1: "trending_bull",
            2: "trending_bear",
            3: "high_vol",
        }
        
        if not HMM_AVAILABLE:
            self.model = None
            return
        
        # Initialize Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_state,
            verbose=False,
        )
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        available = "available" if HMM_AVAILABLE else "not available"
        return f"HMMRegimeDetector(states={self.n_states}, status={status}, hmmlearn={available})"
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for HMM training/prediction.
        
        Features:
          1. Returns (close.pct_change)
          2. Realized volatility (rolling std of returns)
          3. Volume ratio (volume / volume_ma)
        
        Returns:
            Feature matrix (n_samples, 3)
        """
        # Ensure required columns exist
        required = ["close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Feature 1: Returns
        returns = df["close"].pct_change()
        
        # Feature 2: Realized volatility (20-bar rolling std)
        vol = returns.rolling(20).std()
        
        # Feature 3: Volume ratio
        vol_ma = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"] / vol_ma
        
        # Combine features
        features = pd.DataFrame({
            "returns": returns,
            "volatility": vol,
            "vol_ratio": vol_ratio,
        })
        
        # Fill NaNs with forward fill then backward fill
        features = features.fillna(method="ffill").fillna(method="bfill")
        
        # Standardize features (zero mean, unit variance)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features.values
    
    def fit(self, df: pd.DataFrame, window_size: int = 1000) -> "HMMRegimeDetector":
        """
        Train the HMM on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Use last N bars for training (default: 1000)
        
        Returns:
            Self (for chaining)
        """
        if not HMM_AVAILABLE or self.model is None:
            print("  ⚠️  HMM training skipped (hmmlearn not available)")
            return self
        
        # Use most recent window for training
        train_df = df.tail(min(window_size, len(df)))
        
        # Prepare features
        X = self._prepare_features(train_df)
        
        # Train HMM
        try:
            self.model.fit(X)
            self.is_fitted = True
            print(f"  ✓ HMM trained on {len(X)} bars with {self.n_states} states")
        except Exception as e:
            print(f"  ⚠️  HMM training failed: {e}")
            print("  → Will use simple regime detection as fallback")
            self.is_fitted = False
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime labels for each bar in the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data (must have indicators computed)
        
        Returns:
            DataFrame with added 'regime' column
        """
        if not HMM_AVAILABLE or self.model is None or not self.is_fitted:
            # Fallback to simple regime detection
            return self._simple_regime_detection(df)
        
        # Prepare features
        X = self._prepare_features(df)
        
        try:
            # Predict hidden states
            hidden_states = self.model.predict(X)
            
            # Map states to regime names using heuristics
            regime_labels = self._map_states_to_regimes(df, hidden_states)
            df["regime"] = regime_labels
            
        except Exception as e:
            print(f"  ⚠️  HMM prediction failed: {e}")
            print("  → Using simple regime detection as fallback")
            df = self._simple_regime_detection(df)
        
        return df
    
    def _map_states_to_regimes(self, df: pd.DataFrame, states: np.ndarray) -> list:
        """
        Map HMM hidden states to regime labels using feature statistics.
        
        Strategy:
          - Compute mean returns and volatility for each state
          - Assign labels based on these characteristics:
            * High vol + any returns → high_vol
            * Positive returns + low vol → trending_bull
            * Negative returns + low vol → trending_bear
            * Near-zero returns + low vol → ranging
        """
        # Create state-to-regime mapping based on observed characteristics
        state_characteristics = {}
        
        for state_id in range(self.n_states):
            mask = states == state_id
            if mask.sum() == 0:
                continue
            
            # Compute characteristics for this state
            state_returns = df.loc[mask, "close"].pct_change().mean()
            state_vol = df.loc[mask, "close"].pct_change().std()
            
            state_characteristics[state_id] = {
                "mean_return": state_returns,
                "volatility": state_vol,
            }
        
        # Sort states by volatility
        sorted_states = sorted(
            state_characteristics.items(),
            key=lambda x: x[1]["volatility"],
            reverse=True
        )
        
        # Assign regime labels
        state_to_regime = {}
        
        if len(sorted_states) >= 4:
            # Highest volatility → high_vol
            state_to_regime[sorted_states[0][0]] = "high_vol"
            
            # Sort remaining by returns
            remaining = sorted_states[1:]
            remaining_by_returns = sorted(remaining, key=lambda x: x[1]["mean_return"])
            
            # Most negative → trending_bear
            state_to_regime[remaining_by_returns[0][0]] = "trending_bear"
            # Most positive → trending_bull
            state_to_regime[remaining_by_returns[-1][0]] = "trending_bull"
            # Middle → ranging
            state_to_regime[remaining_by_returns[1][0]] = "ranging"
        else:
            # Fallback if insufficient states
            for state_id in range(self.n_states):
                state_to_regime[state_id] = self.regime_map.get(state_id, "ranging")
        
        # Map states to regime labels
        regime_labels = [state_to_regime.get(s, "ranging") for s in states]
        
        return regime_labels
    
    def _simple_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback to simple rule-based regime detection.
        This is the original implementation from core_engine.py.
        """
        from config import REGIME
        
        # Ensure required columns exist
        if "rolling_vol" not in df.columns:
            df["rolling_vol"] = df["close"].pct_change().rolling(20).std()
        if "mean_vol" not in df.columns:
            df["mean_vol"] = df["rolling_vol"].rolling(100).mean()
        if "momentum" not in df.columns:
            df["momentum"] = df["close"].pct_change(5)
        if "ema_spread" not in df.columns and "ema_fast" in df.columns and "ema_slow" in df.columns:
            df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
        
        vol = df["rolling_vol"]
        mean_vol = df["mean_vol"]
        ema_spread = df.get("ema_spread", pd.Series(0, index=df.index))
        momentum = df["momentum"]
        
        conditions = [
            (vol > mean_vol * REGIME["trending_threshold"]),
            (ema_spread > 0.002) & (momentum > 0) & (df["close"] > df.get("ema_trend", df["close"])),
            (ema_spread < -0.002) & (momentum < 0) & (df["close"] < df.get("ema_trend", df["close"])),
        ]
        choices = ["high_vol", "trending_bull", "trending_bear"]
        df["regime"] = np.select(conditions, choices, default="ranging")
        
        return df
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if not HMM_AVAILABLE or self.model is None or not self.is_fitted:
            print("  ⚠️  Cannot save: model not fitted or hmmlearn not available")
            return
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump({
                "model": self.model,
                "n_states": self.n_states,
                "regime_map": self.regime_map,
                "is_fitted": self.is_fitted,
            }, path)
            print(f"  ✓ HMM model saved to {path}")
        except Exception as e:
            print(f"  ✗ Failed to save model: {e}")
    
    @classmethod
    def load(cls, path: str) -> "HMMRegimeDetector":
        """Load trained model from disk."""
        if not HMM_AVAILABLE:
            print("  ⚠️  Cannot load: hmmlearn not available")
            return cls()
        
        try:
            data = joblib.load(path)
            detector = cls(n_states=data["n_states"])
            detector.model = data["model"]
            detector.regime_map = data["regime_map"]
            detector.is_fitted = data["is_fitted"]
            print(f"  ✓ HMM model loaded from {path}")
            return detector
        except Exception as e:
            print(f"  ⚠️  Failed to load model: {e}")
            print("  → Creating new detector")
            return cls()


# Singleton instance for easy import
_detector = None

def get_hmm_detector(force_new: bool = False) -> HMMRegimeDetector:
    """
    Get or create the global HMM detector instance.
    
    Args:
        force_new: Create a new detector instead of using cached one
    
    Returns:
        HMMRegimeDetector instance
    """
    global _detector
    
    if _detector is None or force_new:
        # Try to load from disk first
        model_path = "models/hmm_regime.pkl"
        if os.path.exists(model_path) and not force_new:
            _detector = HMMRegimeDetector.load(model_path)
        else:
            _detector = HMMRegimeDetector()
    
    return _detector
