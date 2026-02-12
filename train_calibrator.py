#!/usr/bin/env python3
"""
SNIPER FRAMEWORK - CALIBRATOR TRAINING SCRIPT
Trains the Platt scaling calibrator for meta-label probability calibration.

Process:
  1. Loads historical data (synthetic or real)
  2. Runs full strategy pipeline to generate signals + ml_scores
  3. Simulates forward outcomes (TP or SL hit first)
  4. Trains MetaLabelCalibrator on scores vs outcomes
  5. Saves trained model to models/calibrator.pkl
  6. Prints reliability diagram (predicted prob vs actual win rate)

Usage:
  python train_calibrator.py --bars 5000
  python train_calibrator.py --provider yfinance --symbol NIFTY --bars 3000
  python train_calibrator.py --provider csv
"""

import sys
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import DATA, BACKTEST
from data.base_provider import get_provider
from strategy.core_engine import run_strategy
from strategy.calibration import MetaLabelCalibrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train meta-label calibrator for Sniper Framework"
    )
    parser.add_argument(
        "--bars", type=int, default=5000,
        help="Number of bars to use for training (default: 5000)"
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["synthetic", "csv", "kite", "yfinance"],
        help="Data provider (default: use config.DATA['provider'])"
    )
    parser.add_argument(
        "--symbol", type=str, default="NIFTY",
        help="Symbol to fetch (default: NIFTY)"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD), overrides --bars"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD)"
    )
    
    return parser.parse_args()


def simulate_trade_outcomes(df: pd.DataFrame) -> pd.Series:
    """
    Simulate forward outcomes for each signal: did price hit TP or SL first?
    
    Args:
        df: DataFrame with signals, entry_price, stop_loss, take_profit, high, low
    
    Returns:
        Series with outcomes ("win"/"loss") for each signal bar
    """
    outcomes = pd.Series(index=df.index, dtype=object)
    
    # Get signal bars
    signal_bars = df[df["signal"] != 0].copy()
    
    if signal_bars.empty:
        return outcomes
    
    for idx, signal_row in signal_bars.iterrows():
        direction = int(signal_row["signal"])
        entry = signal_row["entry_price"]
        sl = signal_row["stop_loss"]
        tp = signal_row["take_profit"]
        
        # Get position in dataframe
        pos = df.index.get_loc(idx)
        
        # Look forward to find TP or SL hit
        outcome = None
        
        for i in range(pos + 1, min(pos + 100, len(df))):  # Look ahead max 100 bars
            future_bar = df.iloc[i]
            
            if direction == 1:  # Long
                # Check SL hit (conservative: use low)
                if future_bar["low"] <= sl:
                    outcome = "loss"
                    break
                # Check TP hit (conservative: use high)
                if future_bar["high"] >= tp:
                    outcome = "win"
                    break
            else:  # Short
                # Check SL hit (conservative: use high)
                if future_bar["high"] >= sl:
                    outcome = "loss"
                    break
                # Check TP hit (conservative: use low)
                if future_bar["low"] <= tp:
                    outcome = "win"
                    break
        
        # If neither hit within 100 bars, consider it neutral (exclude from training)
        if outcome is not None:
            outcomes.loc[idx] = outcome
    
    return outcomes


def print_reliability_diagram(calibrator: MetaLabelCalibrator, signals_df: pd.DataFrame, 
                              outcomes: pd.Series) -> None:
    """Print reliability diagram data (predicted prob vs actual win rate)."""
    reliability = calibrator.reliability_diagram(signals_df, outcomes, n_bins=10)
    
    if reliability.empty:
        print("  ⚠️  Insufficient data for reliability diagram")
        return
    
    print("\n" + "─" * 70)
    print("  RELIABILITY DIAGRAM")
    print("─" * 70)
    print(f"  {'Predicted':>12}  {'Actual':>12}  {'Count':>8}  {'Calibration':>12}")
    print(f"  {'Probability':>12}  {'Win Rate':>12}  {'':>8}  {'Error':>12}")
    print("─" * 70)
    
    for _, row in reliability.iterrows():
        pred = row["predicted_prob"]
        actual = row["actual_win_rate"]
        count = int(row["count"])
        error = abs(pred - actual)
        
        print(f"  {pred:>12.1%}  {actual:>12.1%}  {count:>8}  {error:>12.1%}")
    
    # Overall calibration error
    overall_error = (reliability["predicted_prob"] - reliability["actual_win_rate"]).abs().mean()
    print("─" * 70)
    print(f"  Mean Absolute Calibration Error: {overall_error:.1%}")
    print("─" * 70 + "\n")


def main():
    """Main calibrator training pipeline."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("  SNIPER FRAMEWORK - CALIBRATOR TRAINING")
    print("=" * 70 + "\n")
    
    # ── STEP 1: LOAD DATA ────────────────────────────────────────────────────
    print(f"[1/5] Loading data...")
    
    if args.provider:
        DATA["provider"] = args.provider
    
    provider = get_provider()
    
    # Load data
    if args.start:
        df = provider.fetch(symbol=args.symbol, start=args.start, end=args.end)
    else:
        # Use bars count
        if DATA["provider"] == "synthetic":
            DATA["synthetic_bars"] = args.bars
        df = provider.fetch(symbol=args.symbol)
        df = df.tail(args.bars)
    
    print(f"      Loaded {len(df):,} bars")
    
    # ── STEP 2: RUN STRATEGY ─────────────────────────────────────────────────
    print("[2/5] Running strategy pipeline...")
    df = run_strategy(df)
    
    n_signals = (df["signal"] != 0).sum()
    print(f"      Generated {n_signals} signals")
    
    if n_signals < 50:
        print("\n  ✗ Insufficient signals for calibration (need 50+)")
        print("    Try increasing --bars or adjusting strategy parameters")
        sys.exit(1)
    
    # ── STEP 3: SIMULATE OUTCOMES ────────────────────────────────────────────
    print("[3/5] Simulating trade outcomes...")
    outcomes = simulate_trade_outcomes(df)
    
    # Filter to signal bars with known outcomes
    signal_mask = (df["signal"] != 0) & outcomes.notna()
    signals_with_outcomes = df[signal_mask].copy()
    outcomes_filtered = outcomes[signal_mask]
    
    n_outcomes = len(outcomes_filtered)
    win_rate = (outcomes_filtered == "win").mean()
    
    print(f"      Simulated {n_outcomes} trade outcomes")
    print(f"      Win rate: {win_rate:.1%}")
    
    if n_outcomes < 50:
        print("\n  ✗ Insufficient resolved trades for calibration (need 50+)")
        print("    Try increasing --bars to get more complete trades")
        sys.exit(1)
    
    # ── STEP 4: TRAIN CALIBRATOR ─────────────────────────────────────────────
    print("[4/5] Training Platt scaling calibrator...")
    
    calibrator = MetaLabelCalibrator()
    calibrator.fit(signals_with_outcomes, outcomes_filtered)
    
    if not calibrator.is_fitted:
        print("\n  ✗ Calibrator training failed")
        sys.exit(1)
    
    # ── STEP 5: SAVE MODEL ───────────────────────────────────────────────────
    print("[5/5] Saving trained model...")
    
    model_path = "models/calibrator.pkl"
    calibrator.save(model_path)
    
    # ── PRINT RELIABILITY DIAGRAM ────────────────────────────────────────────
    print_reliability_diagram(calibrator, signals_with_outcomes, outcomes_filtered)
    
    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved to:     {model_path}")
    print(f"  Training samples:   {n_outcomes}")
    print(f"  Baseline win rate:  {win_rate:.1%}")
    print("\n  Next steps:")
    print("    • The calibrator will be automatically used in strategy pipeline")
    print("    • Run backtest to see calibrated probabilities in action")
    print("    • Retrain periodically as market conditions change")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
