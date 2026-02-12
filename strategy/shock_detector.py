"""
SNIPER FRAMEWORK - FAST-SHOCK DETECTION
Detects sudden volatility shocks and suppresses trading signals during unstable conditions.

Detects two types of shocks:
  1. Volatility regime shock: Realized vol >> Implied vol (vol_ratio > 1.5)
  2. Price shock: Single-bar move > 2× ATR

When shock detected:
  - Set signal = 0 for current bar
  - Apply 3-bar cooldown (suppress signals for next 3 bars)
  - Add shock_detected boolean flag to DataFrame

Implied volatility sources (in order of preference):
  1. India VIX (^INDIAVIX) via yfinance
  2. Estimated from ATR/price ratio (fallback)

Gracefully handles missing yfinance data.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional

# Try to import yfinance for India VIX
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Using ATR-based implied vol estimate.")


class ShockDetector:
    """
    Detects volatility and price shocks to suppress signals during unstable conditions.
    
    Usage:
        detector = ShockDetector()
        df = detector.detect(df)  # Adds 'shock_detected' column
    """
    
    def __init__(self, vol_threshold: float = 1.5, price_shock_multiplier: float = 2.0,
                 cooldown_bars: int = 3):
        """
        Initialize shock detector.
        
        Args:
            vol_threshold: Realized/Implied vol ratio threshold (default: 1.5)
            price_shock_multiplier: Price move / ATR threshold (default: 2.0)
            cooldown_bars: Number of bars to suppress signals after shock (default: 3)
        """
        self.vol_threshold = vol_threshold
        self.price_shock_multiplier = price_shock_multiplier
        self.cooldown_bars = cooldown_bars
        self._vix_cache: Optional[pd.Series] = None
    
    def __repr__(self) -> str:
        return (f"ShockDetector(vol_threshold={self.vol_threshold}, "
                f"price_shock={self.price_shock_multiplier}x, "
                f"cooldown={self.cooldown_bars}bars)")
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect shocks and add shock_detected column.
        
        Args:
            df: DataFrame with OHLCV + indicators (must have 'close', 'atr')
        
        Returns:
            DataFrame with added columns:
              - realized_vol: Rolling realized volatility
              - implied_vol: Estimated or fetched implied volatility
              - vol_ratio: Realized / Implied
              - price_shock: Boolean for single-bar price shocks
              - shock_detected: Boolean for any shock (with cooldown)
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Compute realized volatility (annualized)
        df["realized_vol"] = self._compute_realized_vol(df)
        
        # Get or estimate implied volatility
        df["implied_vol"] = self._get_implied_vol(df)
        
        # Compute volatility ratio
        df["vol_ratio"] = df["realized_vol"] / df["implied_vol"].replace(0, np.nan)
        
        # Detect volatility regime shock
        vol_shock = df["vol_ratio"] > self.vol_threshold
        
        # Detect price shock (single-bar move > 2× ATR)
        df["price_shock"] = self._detect_price_shock(df)
        
        # Combine shock types
        shock_flags = vol_shock | df["price_shock"]
        
        # Apply cooldown: if shock detected, suppress next N bars
        df["shock_detected"] = self._apply_cooldown(shock_flags)
        
        return df
    
    def _compute_realized_vol(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Compute realized volatility (annualized).
        
        Uses 5-bar rolling std of returns, annualized for Indian markets:
          - Assume 252 trading days/year
          - Assume 75 bars/day (5-min bars in NSE session)
          - Annualization factor = sqrt(252 * 75)
        """
        returns = df["close"].pct_change()
        realized_vol = returns.rolling(window).std()
        
        # Annualize (assuming 5-min bars)
        annualization_factor = np.sqrt(252 * 75)
        realized_vol = realized_vol * annualization_factor
        
        return realized_vol.fillna(0)
    
    def _get_implied_vol(self, df: pd.DataFrame) -> pd.Series:
        """
        Get implied volatility from India VIX or estimate from ATR.
        
        Strategy:
          1. Try to fetch India VIX (^INDIAVIX) from yfinance
          2. If unavailable, estimate from ATR/price ratio
        """
        # Try to fetch India VIX
        if YFINANCE_AVAILABLE and self._vix_cache is None:
            try:
                vix_data = self._fetch_india_vix(df)
                if vix_data is not None and not vix_data.empty:
                    self._vix_cache = vix_data
            except Exception as e:
                warnings.warn(f"Failed to fetch India VIX: {e}")
        
        # Use VIX if available
        if self._vix_cache is not None and not self._vix_cache.empty:
            # Align VIX data with df index
            implied_vol = df.index.map(
                lambda ts: self._vix_cache.asof(ts) if ts <= self._vix_cache.index[-1] else np.nan
            )
            implied_vol = pd.Series(implied_vol, index=df.index)
            
            # Fill NaNs with forward fill
            implied_vol = implied_vol.fillna(method="ffill")
            
            # Convert VIX (percentage) to decimal
            implied_vol = implied_vol / 100.0
        else:
            # Fallback: estimate from ATR/price ratio
            implied_vol = self._estimate_implied_vol_from_atr(df)
        
        return implied_vol.fillna(0.20)  # Default ~20% vol if all else fails
    
    def _fetch_india_vix(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Fetch India VIX (^INDIAVIX) from Yahoo Finance.
        
        Returns:
            Series with VIX values indexed by datetime, or None if fetch fails
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # Get date range from df
            start = df.index[0] - pd.Timedelta(days=30)  # Extra buffer
            end = df.index[-1]
            
            # Fetch VIX data
            vix = yf.download("^INDIAVIX", start=start, end=end, progress=False)
            
            if vix.empty:
                return None
            
            # Use closing VIX values
            vix_close = vix["Close"] if "Close" in vix.columns else vix["close"]
            
            # Handle timezone
            if vix_close.index.tz is not None:
                vix_close.index = vix_close.index.tz_localize(None)
            
            return vix_close
            
        except Exception:
            return None
    
    def _estimate_implied_vol_from_atr(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Estimate implied volatility from ATR/price ratio.
        
        Formula:
          IV ≈ (ATR / Price) × sqrt(252 * 75)
        
        This is a rough approximation but works as fallback.
        """
        if "atr" not in df.columns:
            # Compute ATR if not present
            df["atr"] = self._compute_atr(df)
        
        atr_ratio = df["atr"] / df["close"]
        
        # Smooth the ratio
        atr_ratio_ma = atr_ratio.rolling(window).mean()
        
        # Annualize
        annualization_factor = np.sqrt(252 * 75)
        implied_vol = atr_ratio_ma * annualization_factor
        
        return implied_vol.fillna(0.20)
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR if not already in DataFrame."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean()
    
    def _detect_price_shock(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect single-bar price shocks.
        
        Shock = abs(bar_move) > price_shock_multiplier × ATR
        """
        if "atr" not in df.columns:
            df["atr"] = self._compute_atr(df)
        
        # Bar-to-bar price move
        price_move = df["close"].diff().abs()
        
        # Shock threshold
        shock_threshold = self.price_shock_multiplier * df["atr"]
        
        # Detect shocks
        price_shock = price_move > shock_threshold
        
        return price_shock.fillna(False)
    
    def _apply_cooldown(self, shock_flags: pd.Series) -> pd.Series:
        """
        Apply cooldown period after shock detection.
        
        If shock detected at bar i, set shock_detected = True for bars i, i+1, i+2, i+3
        """
        result = shock_flags.copy().astype(bool)
        
        # Forward-fill shock flags for cooldown period
        shock_indices = np.where(shock_flags)[0]
        
        for pos in shock_indices:
            # Set cooldown bars
            for offset in range(1, self.cooldown_bars + 1):
                if pos + offset < len(result):
                    result.iloc[pos + offset] = True
        
        return result


# Global instance
_detector = None

def get_shock_detector() -> ShockDetector:
    """Get or create the global shock detector instance."""
    global _detector
    if _detector is None:
        _detector = ShockDetector()
    return _detector
