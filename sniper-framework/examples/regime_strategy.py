"""Strategy with regime detection and meta-labeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sniper.core.event_bus import EventBus
from sniper.core.base_strategy import BaseStrategy
from sniper.core.algorithm import Algorithm
from sniper.data.subscription_manager import SubscriptionManager
from sniper.data.synthetic_provider import SyntheticProvider
from sniper.risk.risk_manager import RiskManager
from sniper.execution.executor import Executor
from sniper.broker.paper_broker import PaperBroker
from sniper.indicators.moving_averages import EMA
from sniper.indicators.oscillators import RSI
from sniper.indicators.volatility import ATR
from sniper.regime.simple_detector import SimpleRegimeDetector
from sniper.signals.meta_labeler import MetaLabeler


class RegimeStrategy(BaseStrategy):
    """EMA + RSI + regime filter + meta-labeling."""

    def initialize(self) -> None:
        self.ema_fast = EMA(9)
        self.ema_slow = EMA(21)
        self.rsi = RSI(14)
        self.atr = ATR(14)
        self.regime_detector = SimpleRegimeDetector(atr_period=14)
        self.meta_labeler = MetaLabeler(threshold=0.60)
        self.subscribe("NIFTY", "1d")
        self.signals_emitted = 0

    def on_data(self, symbol: str, bar) -> None:
        self.ema_fast.update(bar)
        self.ema_slow.update(bar)
        self.rsi.update(bar)
        self.atr.update(bar)
        regime = self.regime_detector.detect(bar)

        if len(self.ema_fast.values) < 21:
            return

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        rsi_val = self.rsi[-1]

        raw_signal = 0
        if ema_f > ema_s and rsi_val > 50 and regime != "high_vol":
            raw_signal = 1
        elif ema_f < ema_s and rsi_val < 50 and regime != "high_vol":
            raw_signal = -1

        if raw_signal == 0:
            return

        if not self.meta_labeler.passes(bar, raw_signal, regime):
            return

        self.emit_signal(
            direction=raw_signal,
            confidence=self.meta_labeler.score(bar, raw_signal, regime),
            symbol=symbol,
            metadata={"entry_price": float(bar.get("close", 0)), "regime": regime},
        )
        self.signals_emitted += 1


def main():
    event_bus = EventBus()
    provider = SyntheticProvider(seed=42)
    sub_manager = SubscriptionManager(event_bus, provider)
    strategy = RegimeStrategy(event_bus)
    risk_manager = RiskManager(event_bus)
    executor = Executor()
    broker = PaperBroker(event_bus=event_bus)

    algorithm = Algorithm(
        strategy=strategy,
        subscription_manager=sub_manager,
        risk_manager=risk_manager,
        executor=executor,
        broker=broker,
        event_bus=event_bus,
    )

    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 31)
    print("Running regime strategy backtest...")
    algorithm.run_backtest(start, end)

    print(f"Signals emitted: {strategy.signals_emitted}")
    print(f"Orders placed: {len(broker.orders)}")
    print(f"Fills: {len(broker.fills)}")


if __name__ == "__main__":
    main()
