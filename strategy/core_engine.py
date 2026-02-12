"""
SNIPER FRAMEWORK - CORE ENGINE (Strategy Layer)
Pipeline:
  1. Compute Indicators (EMA, RSI, ATR, Volume)
  2. Detect Market Regime (trending_bull / trending_bear / ranging / high_vol)
  3. Generate Raw Signals (momentum + confirmation)
  4. Meta-Label Filter (confidence scoring — only high-quality signals pass)
"""

import pandas as pd
import numpy as np
from config import STRATEGY, REGIME


# ─── INDICATORS ──────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the OHLCV dataframe."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # EMAs
    df["ema_fast"] = c.ewm(span=STRATEGY["ema_fast"], adjust=False).mean()
    df["ema_slow"] = c.ewm(span=STRATEGY["ema_slow"], adjust=False).mean()
    df["ema_trend"] = c.ewm(span=STRATEGY["ema_trend"], adjust=False).mean()

    # RSI
    df["rsi"] = _rsi(c, STRATEGY["rsi_period"])

    # ATR (for stop sizing)
    df["atr"] = _atr(h, l, c, STRATEGY["atr_period"])

    # Volume MA & ratio
    df["vol_ma"] = v.rolling(STRATEGY["volume_ma_period"]).mean()
    df["vol_ratio"] = v / df["vol_ma"]

    # Rolling volatility (for regime)
    df["rolling_vol"] = c.pct_change().rolling(REGIME["volatility_lookback"]).std()
    df["mean_vol"] = df["rolling_vol"].rolling(100).mean()

    # Price momentum (rate of change)
    df["momentum"] = c.pct_change(5)

    # EMA spread (how far fast is from slow, normalized)
    df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]

    return df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─── REGIME DETECTION ────────────────────────────────────────────────────────
def detect_regime(df: pd.DataFrame, use_hmm: bool = True) -> pd.DataFrame:
    """
    Label each bar with a market regime.
    
    Uses HMM-based detection if available and trained, otherwise falls back
    to simple rule-based detection.
    
    Regimes:
      trending_bull  - uptrend with normal/low vol
      trending_bear  - downtrend with normal/low vol
      ranging        - sideways, low vol
      high_vol       - elevated volatility, any direction
    """
    if use_hmm:
        try:
            from .regime_hmm import get_hmm_detector
            detector = get_hmm_detector()
            if detector.is_fitted:
                df = detector.predict(df)
                return df
        except Exception as e:
            # Fall through to simple detection
            pass
    
    # Simple rule-based detection (original implementation)
    vol = df["rolling_vol"]
    mean_vol = df["mean_vol"]
    ema_spread = df["ema_spread"]
    momentum = df["momentum"]

    conditions = [
        (vol > mean_vol * REGIME["trending_threshold"]),                          # high vol
        (ema_spread > 0.002) & (momentum > 0) & (df["close"] > df["ema_trend"]), # bull trend
        (ema_spread < -0.002) & (momentum < 0) & (df["close"] < df["ema_trend"]),# bear trend
    ]
    choices = ["high_vol", "trending_bull", "trending_bear"]
    df["regime"] = np.select(conditions, choices, default="ranging")

    return df


# ─── RAW SIGNAL GENERATION ───────────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate raw directional signals: +1 (long), -1 (short), 0 (no trade).
    Rules:
      LONG:  fast EMA > slow EMA, close > trend EMA, RSI > oversold, volume confirms
      SHORT: fast EMA < slow EMA, close < trend EMA, RSI < overbought, volume confirms
    Regime filter: skip signals during high_vol regime.
    """
    long_cond = (
        (df["ema_fast"] > df["ema_slow"]) &
        (df["close"] > df["ema_trend"]) &
        (df["rsi"] > STRATEGY["rsi_oversold"]) &
        (df["rsi"] < 75) &
        (df["vol_ratio"] > 1.0) &
        (df["regime"] != "high_vol")
    )

    short_cond = (
        (df["ema_fast"] < df["ema_slow"]) &
        (df["close"] < df["ema_trend"]) &
        (df["rsi"] < STRATEGY["rsi_overbought"]) &
        (df["rsi"] > 25) &
        (df["vol_ratio"] > 1.0) &
        (df["regime"] != "high_vol")
    )

    # Avoid signal on same bar as previous signal (no repeated entries)
    raw_signal = np.where(long_cond, 1, np.where(short_cond, -1, 0))

    # Deduplicate: only fire on regime transitions or crossover changes
    df["raw_signal"] = raw_signal
    df["raw_signal"] = _deduplicate_signals(df["raw_signal"])

    return df


def _deduplicate_signals(signals: pd.Series) -> pd.Series:
    """Only take a signal if it differs from the previous signal."""
    deduped = signals.copy()
    prev = signals.shift(1).fillna(0)
    deduped[signals == prev] = 0
    return deduped


# ─── META-LABELING FILTER ────────────────────────────────────────────────────
def meta_label_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each raw signal on 4 factors (0-1 each). Only pass signals
    with composite score >= meta_label_threshold.

    Factors:
      1. Trend alignment  - how far price is from trend EMA
      2. Momentum strength - abs value of momentum
      3. Volume confirmation - vol ratio strength
      4. RSI positioning - distance from midpoint (50) towards signal direction

    This is the key filter that lifts accuracy above base rate.
    """
    df["ml_score"] = 0.0

    mask = df["raw_signal"] != 0

    if mask.sum() == 0:
        df["signal"] = 0
        return df

    # Factor 1: Trend alignment (normalize EMA spread, cap at 1)
    trend_strength = (df["ema_spread"].abs() / 0.01).clip(0, 1)

    # Factor 2: Momentum strength
    mom_strength = (df["momentum"].abs() / 0.02).clip(0, 1)

    # Factor 3: Volume confirmation
    vol_strength = ((df["vol_ratio"] - 1.0) / 2.0).clip(0, 1)

    # Factor 4: RSI alignment
    rsi_score = np.where(
        df["raw_signal"] == 1,
        ((df["rsi"] - 50) / 30).clip(0, 1),            # Long: rsi should be above 50
        ((50 - df["rsi"]) / 30).clip(0, 1),            # Short: rsi should be below 50
    )

    composite = (trend_strength * 0.35 + mom_strength * 0.25 +
                 vol_strength * 0.20 + pd.Series(rsi_score, index=df.index) * 0.20)

    df["ml_score"] = composite
    
    # Calibrate scores using Platt scaling if calibrator is fitted
    try:
        from .calibration import get_calibrator
        calibrator = get_calibrator()
        if calibrator.is_fitted:
            df["ml_score_calibrated"] = calibrator.calibrate(df["ml_score"])
            # Use calibrated score for threshold
            composite = df["ml_score_calibrated"]
        else:
            df["ml_score_calibrated"] = df["ml_score"]
    except Exception:
        df["ml_score_calibrated"] = df["ml_score"]

    # Apply threshold filter
    threshold = STRATEGY["meta_label_threshold"]
    df["signal"] = np.where(
        (df["raw_signal"] != 0) & (composite >= threshold),
        df["raw_signal"],
        0
    )

    return df


# ─── ENTRY / EXIT LEVELS ─────────────────────────────────────────────────────
def compute_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal bar, compute entry, stop-loss, and take-profit levels.
    Uses ATR-based dynamic stops.
    """
    from config import RISK
    atr = df["atr"]
    sl_mult = RISK["stop_loss_atr_multiplier"]
    tp_mult = RISK["target_atr_multiplier"]

    df["entry_price"] = df["close"]  # Enter at close of signal bar (simplified)
    df["stop_loss"] = np.where(
        df["signal"] == 1,
        df["close"] - sl_mult * atr,
        np.where(df["signal"] == -1, df["close"] + sl_mult * atr, np.nan)
    )
    df["take_profit"] = np.where(
        df["signal"] == 1,
        df["close"] + tp_mult * atr,
        np.where(df["signal"] == -1, df["close"] - tp_mult * atr, np.nan)
    )

    return df


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────
def run_strategy(df: pd.DataFrame, use_hmm: bool = True, use_shock_detection: bool = True) -> pd.DataFrame:
    """
    Run full strategy pipeline with all upgrades.
    
    Args:
        df: OHLCV DataFrame
        use_hmm: Use HMM regime detection if available (default: True)
        use_shock_detection: Apply shock detection filter (default: True)
    
    Returns:
        DataFrame with indicators, signals, and all computed columns
    """
    df = compute_indicators(df)
    df = detect_regime(df, use_hmm=use_hmm)
    df = generate_signals(df)
    df = meta_label_filter(df)
    df = compute_levels(df)
    
    # Apply shock detection filter
    if use_shock_detection:
        try:
            from .shock_detector import get_shock_detector
            detector = get_shock_detector()
            df = detector.detect(df)
            
            # Suppress signals during shocks
            df.loc[df["shock_detected"], "signal"] = 0
        except Exception as e:
            # Graceful fallback if shock detection fails
            df["shock_detected"] = False
    else:
        df["shock_detected"] = False
    
    return df.dropna(subset=["ema_fast", "rsi", "atr"])
