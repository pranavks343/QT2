"""
SNIPER TRADING FRAMEWORK
Entry point - runs the full pipeline:
  Data → Strategy → Backtest → Risk → Cost → Report

Usage:
  python main.py                      # Run with defaults (synthetic data)
  python main.py --provider csv       # Use your CSV file
  python main.py --cpcv               # Also run CPCV validation
  python main.py --report             # Save trade log to CSV
"""

import sys
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import DATA, BACKTEST


def parse_args():
    p = argparse.ArgumentParser(description="Sniper Trading Framework")
    p.add_argument("--provider", default=DATA["provider"],
                   choices=["synthetic", "csv", "kite"],
                   help="Data provider (default: synthetic)")
    p.add_argument("--symbol", default="NIFTY", help="Instrument symbol")
    p.add_argument("--start", default=BACKTEST["start_date"], help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=BACKTEST["end_date"], help="End date YYYY-MM-DD")
    p.add_argument("--cpcv", action="store_true", help="Run CPCV cross-validation")
    p.add_argument("--report", action="store_true", help="Save trade log to trade_log.csv")
    p.add_argument("--rl", action="store_true", help="Use RL executor if model exists")
    p.add_argument("--train-rl", action="store_true", help="Train RL agent first, then backtest")
    return p.parse_args()


def run(args=None):
    if args is None:
        args = parse_args()

    print("\n╔══════════════════════════════════════════╗")
    print("║     SNIPER TRADING FRAMEWORK v1.0        ║")
    print("╚══════════════════════════════════════════╝\n")

    # ── RL TRAINING (OPTIONAL) ───────────────────────────────────────────────
    if args.train_rl:
        print("[0/5] Training RL agent first...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "rl/train_rl.py",
                 "--provider", args.provider,
                 "--symbol", args.symbol],
                cwd=".",
            )
            if result.returncode != 0:
                print("\n  ⚠️  RL training failed or interrupted.")
                print("     Continuing with backtest...\n")
        except Exception as e:
            print(f"\n  ⚠️  RL training error: {e}")
            print("     Continuing with backtest...\n")

    # ── STEP 1: LOAD DATA ────────────────────────────────────────────────────
    print(f"[1/5] Loading data ({args.provider})...")
    DATA["provider"] = args.provider
    from data.base_provider import get_provider
    provider = get_provider()
    df = provider.fetch(symbol=args.symbol, start=args.start, end=args.end)
    print(f"      Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")

    # ── STEP 2: RUN STRATEGY ─────────────────────────────────────────────────
    print("[2/5] Running strategy pipeline...")
    from strategy.core_engine import run_strategy
    df = run_strategy(df)
    n_signals = (df["signal"] != 0).sum()
    n_raw = (df["raw_signal"] != 0).sum()
    print(f"      Raw signals: {n_raw}  →  After meta-label filter: {n_signals}")

    regime_dist = df["regime"].value_counts(normalize=True).round(3).to_dict()
    print(f"      Regime distribution: {regime_dist}")

    # ── STEP 3: BACKTEST ─────────────────────────────────────────────────────
    print("[3/5] Running backtest...")
    from backtest.engine import BacktestEngine
    from risk.risk_manager import RiskManager

    risk = RiskManager()
    
    # Load RL executor if requested
    rl_executor = None
    if args.rl or args.train_rl:
        try:
            from rl.rl_executor import RLExecutor
            rl_executor = RLExecutor()
            if rl_executor.load("models/rl_agent.zip"):
                print("      ✓ RL executor loaded (third filter layer active)")
            else:
                print("      ⚠️  RL model not found, running without RL filter")
                rl_executor = None
        except ImportError:
            print("      ⚠️  RL module not available (install stable-baselines3)")
            rl_executor = None
    
    engine = BacktestEngine(risk_manager=risk, rl_executor=rl_executor)
    results = engine.run(df)

    # ── STEP 4: PRINT REPORT ─────────────────────────────────────────────────
    print("[4/5] Computing results...")
    results.print_report()

    # ── STEP 5: CPCV (OPTIONAL) ──────────────────────────────────────────────
    if args.cpcv:
        print("[5/5] Running CPCV validation...")
        from backtest.engine import CPCVValidator
        cpcv = CPCVValidator()
        cpcv_results = cpcv.run(df)
        print("\n  CPCV RESULTS:")
        print(f"    Folds run:         {len(cpcv_results.get('folds', []))}")
        print(f"    Mean Sharpe:       {cpcv_results.get('mean_sharpe', 'N/A')}")
        print(f"    Sharpe Std Dev:    {cpcv_results.get('std_sharpe', 'N/A')}")
        print(f"    Min Sharpe:        {cpcv_results.get('min_sharpe', 'N/A')}")
        print(f"    Mean Accuracy:     {cpcv_results.get('mean_accuracy', 'N/A')}")
        print(f"    Consistent +ve:    {cpcv_results.get('consistent', 'N/A')}")
        print(f"    Total Trades:      {cpcv_results.get('total_trades', 'N/A')}\n")
    else:
        print("[5/5] Skipping CPCV (pass --cpcv to enable)\n")

    # ── SAVE TRADE LOG ───────────────────────────────────────────────────────
    if args.report and results.trade_df is not None and not results.trade_df.empty:
        path = "trade_log.csv"
        results.trade_df.to_csv(path, index=False)
        print(f"  Trade log saved to: {path}\n")

    return results


if __name__ == "__main__":
    run()
