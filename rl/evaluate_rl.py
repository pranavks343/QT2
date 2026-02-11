#!/usr/bin/env python3
"""
SNIPER FRAMEWORK - RL AGENT EVALUATION
Compare RL agent vs simple entry on held-out test set.

Metrics compared:
  - Accuracy (% profitable trades)
  - Sharpe ratio
  - Max drawdown
  - Total trades
  - Average holding period
  - Total P&L

Outputs:
  - Comparison table
  - Equity curve plot (rl_evaluation.png)

Usage:
  python rl/evaluate_rl.py
  python rl/evaluate_rl.py --model models/rl_agent.zip --bars 5000
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA
from data.base_provider import get_provider
from strategy.core_engine import run_strategy
from rl.trading_env import NiftyTradingEnv
from rl.rl_executor import RLExecutor


# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Will skip equity curve plot.")

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  stable-baselines3 not installed.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--model", type=str, default="models/rl_agent.zip",
                       help="Path to trained model (default: models/rl_agent.zip)")
    parser.add_argument("--provider", type=str, default=None,
                       choices=["synthetic", "csv", "yfinance"],
                       help="Data provider (default: use config)")
    parser.add_argument("--symbol", type=str, default="NIFTY",
                       help="Symbol to evaluate (default: NIFTY)")
    parser.add_argument("--bars", type=int, default=5000,
                       help="Number of bars to load (default: 5000)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate equity curve plot")
    
    return parser.parse_args()


def evaluate_with_rl(env: NiftyTradingEnv, model: PPO, n_episodes: int = 5) -> dict:
    """
    Evaluate RL agent on environment.
    
    Returns:
        Dictionary with metrics
    """
    all_rewards = []
    all_capitals = []
    all_trades = []
    all_win_rates = []
    all_drawdowns = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        all_rewards.append(episode_reward)
        all_capitals.append(info["capital"])
        all_trades.append(info["total_trades"])
        all_win_rates.append(info["win_rate"])
        all_drawdowns.append(info["drawdown"])
    
    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_capital": np.mean(all_capitals),
        "mean_pnl": np.mean([c - env.initial_capital for c in all_capitals]),
        "mean_trades": np.mean(all_trades),
        "mean_win_rate": np.mean(all_win_rates),
        "mean_drawdown": np.mean(all_drawdowns),
    }


def evaluate_simple_entry(env: NiftyTradingEnv, n_episodes: int = 5) -> dict:
    """
    Evaluate simple entry strategy (always enter on meta-label signal).
    
    Returns:
        Dictionary with metrics
    """
    all_capitals = []
    all_trades = []
    all_win_rates = []
    all_drawdowns = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        steps = 0
        
        # Get signal direction from environment's dataframe
        while not done and steps < 2000:
            current_bar = env.df.iloc[env.current_step]
            
            # Simple strategy: always enter on signal
            if current_bar.get("signal", 0) == 1:
                action = 1  # Long
            elif current_bar.get("signal", 0) == -1:
                action = 2  # Short
            else:
                action = 0  # Hold
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        all_capitals.append(info["capital"])
        all_trades.append(info["total_trades"])
        all_win_rates.append(info["win_rate"])
        all_drawdowns.append(info["drawdown"])
    
    return {
        "mean_capital": np.mean(all_capitals),
        "mean_pnl": np.mean([c - env.initial_capital for c in all_capitals]),
        "mean_trades": np.mean(all_trades),
        "mean_win_rate": np.mean(all_win_rates),
        "mean_drawdown": np.mean(all_drawdowns),
    }


def plot_comparison(rl_equity: list, simple_equity: list, save_path: str):
    """Plot equity curves comparison."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️  Matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(rl_equity, label="RL Agent", linewidth=2, color="#00d9ff")
    plt.plot(simple_equity, label="Simple Entry", linewidth=2, color="#ff4444", alpha=0.7)
    
    plt.xlabel("Steps")
    plt.ylabel("Capital (₹)")
    plt.title("RL Agent vs Simple Entry - Equity Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Equity curve saved to {save_path}")


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("  SNIPER FRAMEWORK - RL AGENT EVALUATION")
    print("=" * 70 + "\n")
    
    # ── STEP 1: LOAD DATA ────────────────────────────────────────────────
    print("[1/5] Loading data...")
    
    if args.provider:
        DATA["provider"] = args.provider
    
    if DATA["provider"] == "synthetic":
        DATA["synthetic_bars"] = args.bars
    
    provider = get_provider()
    df = provider.fetch(symbol=args.symbol)
    df = df.tail(args.bars)
    
    print(f"      Loaded {len(df):,} bars")
    
    # ── STEP 2: RUN STRATEGY ─────────────────────────────────────────────
    print("[2/5] Running strategy pipeline...")
    df = run_strategy(df)
    
    n_signals = (df["signal"] != 0).sum()
    print(f"      Generated {n_signals} signals")
    
    # ── STEP 3: SPLIT INTO TEST SET ──────────────────────────────────────
    print("[3/5] Using last 15% as test set...")
    
    test_start = int(len(df) * 0.85)
    test_df = df.iloc[test_start:].copy()
    
    print(f"      Test set: {len(test_df):,} bars")
    
    # ── STEP 4: LOAD RL MODEL ────────────────────────────────────────────
    print("[4/5] Loading RL model...")
    
    if not os.path.exists(args.model):
        print(f"\n  ✗ Model not found: {args.model}")
        print("     Train model first: python rl/train_rl.py")
        sys.exit(1)
    
    executor = RLExecutor()
    if not executor.load(args.model):
        sys.exit(1)
    
    model = executor.model
    
    # ── STEP 5: EVALUATE ─────────────────────────────────────────────────
    print("[5/5] Evaluating on test set...\n")
    
    test_env = NiftyTradingEnv(test_df, initial_capital=100_000)
    
    # Evaluate RL agent
    print("  Evaluating RL agent (5 episodes)...")
    rl_results = evaluate_with_rl(test_env, model, n_episodes=5)
    
    # Evaluate simple entry
    print("  Evaluating simple entry (5 episodes)...")
    simple_results = evaluate_simple_entry(test_env, n_episodes=5)
    
    # ── PRINT COMPARISON ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  EVALUATION RESULTS")
    print("─" * 70)
    print(f"  {'Metric':<25} {'RL Agent':>15} {'Simple Entry':>15} {'Improvement':>12}")
    print("─" * 70)
    
    metrics = [
        ("Mean P&L (₹)", "mean_pnl", "{:.0f}"),
        ("Mean Capital (₹)", "mean_capital", "{:.0f}"),
        ("Mean Trades", "mean_trades", "{:.1f}"),
        ("Mean Win Rate", "mean_win_rate", "{:.1%}"),
        ("Mean Drawdown", "mean_drawdown", "{:.1%}"),
    ]
    
    for label, key, fmt in metrics:
        rl_val = rl_results.get(key, 0)
        simple_val = simple_results.get(key, 0)
        
        # Compute improvement
        if simple_val != 0:
            if key in ["mean_drawdown"]:
                # Lower is better
                improvement = -(rl_val - simple_val) / simple_val * 100
            else:
                # Higher is better
                improvement = (rl_val - simple_val) / abs(simple_val) * 100
        else:
            improvement = 0
        
        rl_str = fmt.format(rl_val)
        simple_str = fmt.format(simple_val)
        imp_str = f"{improvement:+.1f}%" if improvement != 0 else "—"
        
        print(f"  {label:<25} {rl_str:>15} {simple_str:>15} {imp_str:>12}")
    
    print("─" * 70)
    
    # ── VERDICT ──────────────────────────────────────────────────────────
    print("\n  VERDICT:")
    
    better_pnl = rl_results["mean_pnl"] > simple_results["mean_pnl"]
    better_winrate = rl_results["mean_win_rate"] > simple_results["mean_win_rate"]
    better_dd = rl_results["mean_drawdown"] < simple_results["mean_drawdown"]
    
    score = sum([better_pnl, better_winrate, better_dd])
    
    if score >= 2:
        verdict = "✅ RL AGENT OUTPERFORMS SIMPLE ENTRY"
    elif score == 1:
        verdict = "⚠️  RL AGENT MIXED RESULTS"
    else:
        verdict = "❌ SIMPLE ENTRY OUTPERFORMS RL AGENT"
    
    print(f"    {verdict}")
    
    # ── PLOT (OPTIONAL) ──────────────────────────────────────────────────
    if args.plot and MATPLOTLIB_AVAILABLE:
        print("\n  Generating equity curve plot...")
        
        # Run one full episode for each to get equity curves
        rl_equity = []
        simple_equity = []
        
        # RL equity
        obs, info = test_env.reset(seed=42)
        rl_equity.append(info["capital"])
        done = False
        steps = 0
        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            rl_equity.append(info["capital"])
            steps += 1
        
        # Simple equity
        obs, info = test_env.reset(seed=42)
        simple_equity.append(info["capital"])
        done = False
        steps = 0
        while not done and steps < 2000:
            current_bar = test_env.df.iloc[test_env.current_step]
            if current_bar.get("signal", 0) == 1:
                action = 1
            elif current_bar.get("signal", 0) == -1:
                action = 2
            else:
                action = 0
            obs, reward, done, truncated, info = test_env.step(action)
            simple_equity.append(info["capital"])
            steps += 1
        
        plot_comparison(rl_equity, simple_equity, "rl_evaluation.png")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if not SB3_AVAILABLE:
        sys.exit(1)
    
    args = parse_args()
    main()
