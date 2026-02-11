"""WebSocket market stream endpoint."""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.websocket.connection_manager import ConnectionManager
from data.base_provider import SyntheticProvider, get_provider
from risk.risk_manager import RiskManager
from rl.rl_executor import RLExecutor
from strategy.alpha_monitor import AlphaDecayMonitor
from strategy.core_engine import run_strategy
from strategy.shock_detector import ShockDetector

router = APIRouter()
manager = ConnectionManager()


async def _safe_send(websocket: WebSocket, msg_type: str, data: Dict[str, Any]) -> None:
    payload = {"type": msg_type, "data": data, "timestamp": datetime.utcnow().isoformat()}
    await websocket.send_json(payload)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _is_nan(value: Any) -> bool:
    return value != value


def _build_candle(row, ts: datetime) -> Dict[str, Any]:
    return {
        "datetime": ts.isoformat(),
        "open": _num(row["open"]),
        "high": _num(row["high"]),
        "low": _num(row["low"]),
        "close": _num(row["close"]),
        "volume": _num(row.get("volume", 0)),
        "ema_fast": _num(row.get("ema_fast", 0)),
        "ema_slow": _num(row.get("ema_slow", 0)),
        "ema_trend": _num(row.get("ema_trend", 0)),
        "rsi": _num(row.get("rsi", 0)),
        "signal": int(row.get("signal", 0) or 0),
        "raw_signal": int(row.get("raw_signal", 0) or 0),
        "regime": str(row.get("regime", "ranging") or "ranging"),
        "ml_score": _num(row.get("ml_score", 0)),
        "shock_detected": False if _is_nan(row.get("shock_detected", False)) else bool(row.get("shock_detected", False)),
        "stop_loss": None if row.get("stop_loss") is None or _is_nan(row.get("stop_loss")) else _num(row.get("stop_loss")),
        "take_profit": None if row.get("take_profit") is None or _is_nan(row.get("take_profit")) else _num(row.get("take_profit")),
    }


async def _fetch_latest_with_retry(symbol: str, timeframe: str, bars: int):
    provider = get_provider()
    for attempt in range(2):
        try:
            if hasattr(provider, "fetch_latest"):
                return provider.fetch_latest(symbol=symbol, timeframe=timeframe, bars=bars)
            return provider.fetch(symbol=symbol, start=None, end=None).tail(bars)
        except Exception:
            if attempt == 0:
                await asyncio.sleep(10)
    return SyntheticProvider().fetch(symbol=symbol, start=None, end=None).tail(bars)


@router.websocket("/ws/{symbol}/{timeframe}")
async def market_stream(websocket: WebSocket, symbol: str, timeframe: str):
    await manager.connect(websocket, symbol, timeframe)

    prev_regime: Optional[str] = None
    last_signal_ts: Optional[str] = None
    shock = ShockDetector()
    alpha_monitor = AlphaDecayMonitor()
    risk_manager = RiskManager()
    rl_executor = RLExecutor()
    try:
        rl_executor.load("models/rl_agent")
    except Exception:
        rl_executor.is_loaded = False

    async def stream_loop() -> None:
        nonlocal prev_regime, last_signal_ts
        while True:
            try:
                raw = await _fetch_latest_with_retry(symbol, timeframe, 200)
                strat = run_strategy(raw.copy()).tail(200)
                strat = shock.detect(strat)
                latest_ts = strat.index[-1]
                latest = strat.iloc[-1]
                candle = _build_candle(latest, latest_ts)

                await _safe_send(websocket, "candle_update", candle)

                if prev_regime is not None and candle["regime"] != prev_regime:
                    await _safe_send(
                        websocket,
                        "regime_change",
                        {"from": prev_regime, "to": candle["regime"], "timestamp": datetime.utcnow().isoformat()},
                    )
                prev_regime = candle["regime"]

                if candle["signal"] != 0 and candle["datetime"] != last_signal_ts:
                    await _safe_send(
                        websocket,
                        "signal_fired",
                        {
                            "direction": "LONG" if candle["signal"] > 0 else "SHORT",
                            "ml_score": candle["ml_score"],
                            "entry_price": candle["close"],
                            "stop_loss": candle.get("stop_loss") or candle["close"] * (0.99 if candle["signal"] > 0 else 1.01),
                            "take_profit": candle.get("take_profit") or candle["close"] * (1.02 if candle["signal"] > 0 else 0.98),
                            "regime": candle["regime"],
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )
                    last_signal_ts = candle["datetime"]

                if candle["shock_detected"]:
                    await _safe_send(
                        websocket,
                        "shock_detected",
                        {"vol_ratio": float(latest.get("vol_ratio", 0)), "bars_cooldown": int(getattr(shock, "cooldown_remaining", 3)), "timestamp": datetime.utcnow().isoformat()},
                    )

                action_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
                rl_action = "NOT_LOADED"
                rl_conf = 0.0
                overrode = False
                if rl_executor.is_loaded:
                    obs = rl_executor.build_observation(strat, len(strat) - 1)
                    act = rl_executor.decide(obs)
                    rl_action = action_map.get(act, "HOLD")
                    rl_conf = 0.5
                    overrode = candle["signal"] != 0 and rl_action == "HOLD"

                await _safe_send(
                    websocket,
                    "rl_decision",
                    {"action": rl_action, "confidence": rl_conf, "overrode_signal": overrode, "timestamp": datetime.utcnow().isoformat()},
                )

                alpha_status = alpha_monitor.status()
                if alpha_status.get("decay_flag", False):
                    await _safe_send(
                        websocket,
                        "alpha_warning",
                        {
                            "rolling_sharpe": alpha_status.get("rolling_sharpe", 0),
                            "baseline_sharpe": alpha_status.get("baseline_sharpe", 0),
                            "rolling_accuracy": alpha_status.get("rolling_accuracy", 0),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                rs = risk_manager.summary()
                await _safe_send(
                    websocket,
                    "risk_update",
                    {
                        "drawdown_pct": float(rs.get("max_drawdown_pct", 0)) / 100,
                        "current_capital": float(rs.get("current_capital", 0)),
                        "daily_pnl_pct": 0.0,
                        "trading_halted": bool(rs.get("trading_halted", False)),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

                await asyncio.sleep(3)
            except WebSocketDisconnect:
                break
            except Exception as exc:
                await _safe_send(websocket, "error", {"message": str(exc), "timestamp": datetime.utcnow().isoformat()})
                await asyncio.sleep(3)

    task = asyncio.create_task(stream_loop())

    try:
        hist = await _fetch_latest_with_retry(symbol, timeframe, 200)
        hist = run_strategy(hist.copy()).tail(200)
        candles = [_build_candle(row, idx) for idx, row in hist.iterrows()]
        await _safe_send(websocket, "init", {"candles": candles, "symbol": symbol, "timestamp": datetime.utcnow().isoformat()})
        await task
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        await manager.disconnect(websocket, symbol, timeframe)
