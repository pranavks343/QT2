"""Unit tests for indicators."""

import pandas as pd
import numpy as np

from sniper.indicators.moving_averages import EMA, SMA
from sniper.indicators.oscillators import RSI, MACD, Stochastic
from sniper.indicators.volatility import ATR, BollingerBands


def _bar(open_, high, low, close, volume=1000):
    return pd.Series({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_ema():
    ema = EMA(period=3)
    ema.update(_bar(100, 102, 99, 101))
    ema.update(_bar(101, 103, 100, 102))
    ema.update(_bar(102, 104, 101, 103))
    assert len(ema.values) == 3
    assert ema[-1] > 0
    assert ema[-1] == ema.values[-1]


def test_sma():
    sma = SMA(period=3)
    sma.update(_bar(100, 102, 99, 100))
    sma.update(_bar(100, 102, 99, 101))
    sma.update(_bar(100, 102, 99, 102))
    assert sma[-1] == 101.0  # (100+101+102)/3


def test_rsi():
    rsi = RSI(period=14)
    for i in range(20):
        rsi.update(_bar(100, 101, 99, 100 + (i % 2)))
    assert 0 <= rsi[-1] <= 100


def test_atr():
    atr = ATR(period=3)
    atr.update(_bar(100, 105, 99, 102))
    atr.update(_bar(102, 108, 101, 105))
    atr.update(_bar(105, 110, 104, 106))
    assert atr[-1] > 0


def test_bollinger_bands():
    bb = BollingerBands(period=5)
    for i in range(10):
        bb.update(_bar(100, 101, 99, 100 + i * 0.1))
    u, m, l = bb.get_bands()
    assert u > m > l


def test_indicator_reset():
    ema = EMA(3)
    ema.update(_bar(100, 101, 99, 100))
    ema.reset()
    assert len(ema.values) == 0
