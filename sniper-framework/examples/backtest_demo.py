"""Backtest engine demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sniper.backtest.engine import BacktestEngine
from sniper.data.synthetic_provider import SyntheticProvider
from examples.momentum_strategy import MomentumStrategy


def main():
    provider = SyntheticProvider(seed=42)
    engine = BacktestEngine(
        strategy=MomentumStrategy,
        data_provider=provider,
        initial_capital=500000,
    )
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 30)
    results = engine.run(start, end)

    print("Backtest Results")
    print("-" * 40)
    for k, v in results.to_dict().items():
        print(f"  {k}: {v}")
    print("-" * 40)
    print(f"Equity curve points: {len(results.equity_curve)}")


if __name__ == "__main__":
    main()
