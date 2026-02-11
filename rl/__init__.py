"""
SNIPER FRAMEWORK - REINFORCEMENT LEARNING MODULE
PPO-based execution agent for trade entry timing and confirmation.

Components:
  - trading_env.py: Gymnasium environment for training
  - train_rl.py: PPO training script  
  - rl_executor.py: Inference interface for backtesting
  - evaluate_rl.py: Evaluation and comparison script
"""

__all__ = ["NiftyTradingEnv", "RLExecutor"]
