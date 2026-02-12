"""Unit tests for Phase 3 components."""

from datetime import datetime

import pandas as pd

from sniper.core.event_bus import EventBus, EventType
from sniper.regime.simple_detector import SimpleRegimeDetector
from sniper.regime.shock_detector import ShockDetector
from sniper.signals.meta_labeler import MetaLabeler
from sniper.signals.signal_filter import SignalFilter
from sniper.risk.risk_manager import RiskManager
from sniper.execution.cost_model import NSECostModel
from sniper.execution.slippage_model import SlippageModel
from sniper.broker.paper_broker import PaperBroker
from sniper.broker.exceptions import BrokerError, OrderRejectedError


def _bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": 1000})


def test_simple_regime_detector():
    det = SimpleRegimeDetector()
    for i in range(50):
        det.detect(_bar(100, 101, 99, 100 + i * 0.01))
    assert det.get_regime() in ("low_vol", "normal", "high_vol")


def test_shock_detector():
    det = ShockDetector(return_threshold=0.02)
    det.detect(_bar(100, 101, 99, 100))
    det.detect(_bar(100, 101, 99, 103))
    assert det.in_shock()


def test_meta_labeler():
    ml = MetaLabeler(threshold=0.60)
    bar = _bar(100, 102, 99, 101)
    score = ml.score(bar, 1, "normal")
    assert 0 <= score <= 1
    assert ml.passes(bar, 1, "high_vol") is False


def test_signal_filter():
    sf = SignalFilter(min_confidence=0.5)
    assert sf.passes({"confidence": 0.7}) is True
    assert sf.passes({"confidence": 0.3}) is False


def test_risk_manager():
    bus = EventBus()
    rm = RiskManager(bus, initial_capital=500000)
    assert rm.can_trade()
    size = rm.compute_position_size({"confidence": 0.7})
    assert "lots" in size
    assert "quantity" in size


def test_cost_model():
    cm = NSECostModel()
    r = cm.compute(25, 100.0, True)
    assert "total" in r
    assert r["notional"] == 2500


def test_slippage():
    slip = SlippageModel(mode="fixed", value=0.001)
    p = slip.apply(100.0, 25)
    assert p > 100


def test_paper_broker_fill():
    bus = EventBus()
    fills = []
    bus.subscribe(EventType.ORDER_FILLED, lambda e: fills.append(e))
    broker = PaperBroker(event_bus=bus)
    order = {"symbol": "NIFTY", "direction": 1, "quantity": 25, "price": 100.0}
    oid = broker.place_order(order)
    assert oid.startswith("paper_")
    assert len(fills) == 1
    assert fills[0].data["fill_price"] == 100.0
