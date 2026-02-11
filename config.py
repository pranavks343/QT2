"""
SNIPER TRADING FRAMEWORK - CONFIG
All parameters in one place. Change here, affects the entire system.
"""

# ─── STRATEGY PARAMETERS ───────────────────────────────────────────────────
STRATEGY = {
    "instrument": "NIFTY",              # NIFTY or BANKNIFTY
    "timeframe": "5min",                # 1min / 5min / 15min
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,                    # Trend filter
    "rsi_period": 14,
    "rsi_oversold": 40,
    "rsi_overbought": 60,
    "atr_period": 14,
    "volume_ma_period": 20,
    "meta_label_threshold": 0.60,       # Min confidence to take trade (0-1)
}

# ─── REGIME DETECTION ──────────────────────────────────────────────────────
REGIME = {
    "volatility_lookback": 20,          # Bars to measure rolling volatility
    "trending_threshold": 1.2,          # Vol multiplier above mean = trending
    "ranging_threshold": 0.8,           # Vol multiplier below mean = ranging
    # Regimes: "trending_bull", "trending_bear", "ranging", "high_vol"
}

# ─── RISK PARAMETERS ───────────────────────────────────────────────────────
RISK = {
    "capital": 500_000,                 # Starting capital in INR
    "max_drawdown_pct": 0.10,           # 10% hard stop - kills all trading
    "daily_loss_limit_pct": 0.03,       # 3% daily loss limit
    "kelly_fraction": 0.25,             # Quarter-Kelly (conservative)
    "max_position_pct": 0.05,           # Max 5% capital per trade
    "max_concurrent_positions": 3,
    "stop_loss_atr_multiplier": 1.5,    # SL = entry ± 1.5 × ATR
    "target_atr_multiplier": 3.0,       # TP = entry ± 3.0 × ATR (R:R = 2:1)
}

# ─── EXECUTION COST MODEL (NSE Derivatives) ────────────────────────────────
COSTS = {
    # Options costs (on premium value)
    "stt_sell_pct": 0.0005,             # 0.05% STT on sell of options
    "brokerage_per_order": 20,          # Flat ₹20 per order (Zerodha model)
    "stamp_duty_buy_pct": 0.00003,      # 0.003% on buy side
    "exchange_txn_pct": 0.00053,        # NSE transaction charges
    "sebi_charges_pct": 0.000001,       # SEBI charges (negligible)
    "gst_on_brokerage_pct": 0.18,       # 18% GST on brokerage
    "slippage_bps": 5,                  # 5 bps slippage per side
    "lot_size_nifty": 75,               # Post-SEBI 2024 lot size
    "lot_size_banknifty": 30,
    "min_trade_premium": 10,            # Skip trades below ₹10 premium (STT trap)
}

# ─── BACKTEST / VALIDATION ─────────────────────────────────────────────────
BACKTEST = {
    "min_trades_for_validity": 1000,    # Statistical credibility threshold
    "cpcv_folds": 10,                   # Combinatorial Purged Cross-Val folds
    "cpcv_embargo_bars": 10,            # Embargo period between folds
    "min_sharpe_ratio": 1.5,            # Minimum acceptable Sharpe
    "min_accuracy": 0.80,               # 80% accuracy target
    "max_drawdown_allowed": 0.10,       # 10% max drawdown
    "start_date": "2021-01-01",
    "end_date": "2024-12-31",
    "commission_included": True,        # Always test with real costs
}

# ─── DATA LAYER ────────────────────────────────────────────────────────────
DATA = {
    "provider": "synthetic",            # "synthetic" | "csv" | "kite" | "yfinance" | "custom"
    "csv_path": "./data/nifty_data.csv",
    "synthetic_bars": 5000,
    "synthetic_seed": 42,
}
