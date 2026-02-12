# Sniper Framework

Institutional-grade algorithmic trading framework for Indian derivatives (NSE).

## Features

- **Phase 1**: Event-driven core (EventBus, BaseStrategy, Algorithm)
- **Phase 2**: Data layer (Synthetic, CSV, YFinance) + Indicators (EMA, SMA, RSI, MACD, ATR, BollingerBands)
- **Phase 3**: Regime detection (Simple, HMM, Shock), Meta-labeling, Risk (Kelly, drawdown, VaR), Execution (cost, slippage), Paper broker with fills
- **Phase 4**: Backtest engine with results (accuracy, Sharpe, max drawdown, verdict), CPCV/DSR/walk-forward/Monte Carlo stubs

## Installation

```bash
pip install -e .
# Optional: pip install -e ".[yfinance]" for live data
```

## Quick Start

```python
from sniper.core.base_strategy import BaseStrategy
from sniper.core.event_bus import EventBus
from sniper.indicators.moving_averages import EMA
from sniper.indicators.oscillators import RSI

class MomentumStrategy(BaseStrategy):
    def initialize(self):
        self.ema_fast = EMA(9)
        self.ema_slow = EMA(21)
        self.rsi = RSI(14)
        self.subscribe("NIFTY", "1d")
    
    def on_data(self, symbol, bar):
        self.ema_fast.update(bar)
        self.ema_slow.update(bar)
        self.rsi.update(bar)
        if len(self.ema_fast.values) >= 21 and self.ema_fast[-1] > self.ema_slow[-1]:
            self.emit_signal(direction=1, confidence=0.7, symbol=symbol)
```

Run backtest:

```python
from sniper.backtest.engine import BacktestEngine
from sniper.data.synthetic_provider import SyntheticProvider

engine = BacktestEngine(strategy=MomentumStrategy, data_provider=SyntheticProvider(seed=42))
results = engine.run(datetime(2024,1,1), datetime(2024,6,30))
print(results.to_dict())  # accuracy, sharpe_ratio, max_drawdown, verdict
```
