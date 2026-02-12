#!/usr/bin/env python3
"""
SNIPER FRAMEWORK - RL AGENT TRAINING
Trains a PPO (Proximal Policy Optimization) agent for execution timing.

Why PPO:
  - Stable training with clipped objective
  - Handles discrete actions well
  - Industry standard for trading RL
  - Works well with financial time series
  - Better than DQN/A3C for this use case

Process:
  1. Load data via configured provider
  2. Run strategy pipeline to generate signals + indicators
  3. Split into train (70%), validation (15%), test (15%) - time-based
  4. Train PPO agent on train set
  5. Evaluate on validation every 50k steps
  6. Save best model to models/rl_agent.zip
  7. Print training metrics: reward curve, win rate, avg trade duration

Usage:
  python rl/train_rl.py
  python rl/train_rl.py --timesteps 1000000 --provider yfinance --symbol NIFTY
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

# Import RL libraries
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  stable-baselines3 not installed.")
    print("   Install with: pip install stable-baselines3 gymnasium torch")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL execution agent")
    parser.add_argument("--timesteps", type=int, default=500_000,
                       help="Total training timesteps (default: 500k)")
    parser.add_argument("--provider", type=str, default=None,
                       choices=["synthetic", "csv", "yfinance"],
                       help="Data provider (default: use config)")
    parser.add_argument("--symbol", type=str, default="NIFTY",
                       help="Symbol to train on (default: NIFTY)")
    parser.add_argument("--bars", type=int, default=10000,
                       help="Number of bars to load (default: 10000)")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                       help="Evaluate every N steps (default: 50k)")
    parser.add_argument("--save-path", type=str, default="models/rl_agent",
                       help="Path to save model (default: models/rl_agent)")
    
    return parser.parse_args()


class ProgressCallback(BaseCallback):
    """Custom callback to print training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_episodes = 0
    
    def _on_step(self) -> bool:
        # Collect episode info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.n_episodes += 1
                
                # Print every 10 episodes
                if self.n_episodes % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_lengths = self.episode_lengths[-10:]
                    print(f"  Episode {self.n_episodes:>4} | "
                          f"Reward: {np.mean(recent_rewards):>8.2f} | "
                          f"Length: {np.mean(recent_lengths):>6.1f}")
        
        return True


def train_rl_agent(args):
    """Main training pipeline."""
    
    print("\n" + "=" * 70)
    print("  SNIPER FRAMEWORK - RL AGENT TRAINING")
    print("=" * 70 + "\n")
    
    # ── STEP 1: LOAD DATA ────────────────────────────────────────────────
    print("[1/7] Loading data...")
    
    if args.provider:
        DATA["provider"] = args.provider
    
    if DATA["provider"] == "synthetic":
        DATA["synthetic_bars"] = args.bars
    
    provider = get_provider()
    df = provider.fetch(symbol=args.symbol)
    
    # Limit to requested bars
    df = df.tail(args.bars)
    
    print(f"      Loaded {len(df):,} bars ({df.index[0]} → {df.index[-1]})")
    
    # ── STEP 2: RUN STRATEGY ─────────────────────────────────────────────
    print("[2/7] Running strategy pipeline...")
    df = run_strategy(df)
    
    n_signals = (df["signal"] != 0).sum()
    print(f"      Generated {n_signals} signals")
    
    if n_signals < 50:
        print("\n  ⚠️  Warning: Few signals for RL training (need 50+)")
        print("     Consider lowering meta_label_threshold or using more data")
    
    # ── STEP 3: SPLIT DATA ───────────────────────────────────────────────
    print("[3/7] Splitting data (70/15/15 train/val/test)...")
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"      Train: {len(train_df):,} bars")
    print(f"      Val:   {len(val_df):,} bars")
    print(f"      Test:  {len(test_df):,} bars")
    
    # ── STEP 4: CREATE ENVIRONMENTS ──────────────────────────────────────
    print("[4/7] Creating training and validation environments...")
    
    train_env = NiftyTradingEnv(train_df, initial_capital=100_000)
    val_env = NiftyTradingEnv(val_df, initial_capital=100_000)
    
    # Wrap with Monitor for logging
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    print(f"      Obs space: {train_env.observation_space.shape}")
    print(f"      Action space: {train_env.action_space}")
    
    # ── STEP 5: CREATE PPO AGENT ─────────────────────────────────────────
    print("[5/7] Creating PPO agent...")
    
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy coefficient for exploration
        verbose=0,
        tensorboard_log="./logs/rl_training/",
    )
    
    print("      Policy: MlpPolicy")
    print("      Algorithm: PPO")
    print(f"      Total timesteps: {args.timesteps:,}")
    
    # ── STEP 6: TRAIN ────────────────────────────────────────────────────
    print(f"[6/7] Training PPO agent ({args.timesteps:,} timesteps)...")
    print("      This may take 10-30 minutes...\n")
    
    # Create save directory
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=os.path.dirname(args.save_path),
        log_path="./logs/rl_eval/",
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # Progress callback
    progress_callback = ProgressCallback()
    
    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, progress_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user.")
    
    # ── STEP 7: SAVE MODEL ───────────────────────────────────────────────
    print("\n[7/7] Saving trained model...")
    
    model.save(args.save_path)
    print(f"      ✓ Model saved to {args.save_path}.zip")
    
    # ── FINAL EVALUATION ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  TRAINING SUMMARY")
    print("─" * 70)
    
    if progress_callback.episode_rewards:
        print(f"  Episodes completed:     {progress_callback.n_episodes}")
        print(f"  Mean episode reward:    {np.mean(progress_callback.episode_rewards):.2f}")
        print(f"  Std episode reward:     {np.std(progress_callback.episode_rewards):.2f}")
        print(f"  Mean episode length:    {np.mean(progress_callback.episode_lengths):.1f} steps")
    
    # Quick test on validation
    print("\n  Testing on validation set...")
    obs, info = val_env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = val_env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    
    print(f"    Validation reward:      {total_reward:.2f}")
    print(f"    Final capital:          ₹{info['capital']:,.0f}")
    print(f"    P&L:                    ₹{info['capital'] - 100_000:,.0f}")
    print(f"    Trades:                 {info['total_trades']}")
    print(f"    Win rate:               {info['win_rate']:.1%}")
    print(f"    Drawdown:               {info['drawdown']:.1%}")
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Model saved to: {args.save_path}.zip")
    print("\n  Next steps:")
    print("    • Run evaluation: python rl/evaluate_rl.py")
    print("    • Use in backtest: python main.py --rl")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if not SB3_AVAILABLE:
        sys.exit(1)
    
    args = parse_args()
    train_rl_agent(args)
