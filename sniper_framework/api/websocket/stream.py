"""
WebSocket Streaming Endpoint
Real-time market data stream with full strategy pipeline
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.base_provider import get_provider
from strategy.core_engine import run_strategy
from risk.risk_manager import RiskManager
from api.websocket.connection_manager import ConnectionManager


router = APIRouter()
manager = ConnectionManager()


# ─── HELPER CLASSES ─────────────────────────────────────────────────────────

class StreamState:
    """Maintains state for a WebSocket stream"""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.previous_regime: Optional[str] = None
        self.last_signal_bar: Optional[int] = None
        self.shock_cooldown: int = 0
        self.risk_manager = RiskManager()
        self.rl_executor = None
        self.load_rl_executor()

    def load_rl_executor(self):
        """Try to load RL executor if available"""
        try:
            from rl.rl_executor import RLExecutor
            self.rl_executor = RLExecutor()
            if not self.rl_executor.load("models/rl_agent.zip"):
                self.rl_executor = None
        except (ImportError, Exception):
            self.rl_executor = None


# ─── WEBSOCKET ENDPOINT ─────────────────────────────────────────────────────

@router.websocket("/ws/{symbol}/{timeframe}")
async def market_stream(websocket: WebSocket, symbol: str, timeframe: str):
    """
    Real-time market data WebSocket stream

    Flow:
    1. Accept connection
    2. Send full historical data as first message
    3. Start streaming loop: every 3 seconds
       - Fetch latest bar
       - Run strategy pipeline
       - Detect changes (regime, signals, shocks)
       - Broadcast updates
    """

    # Initialize connection
    await manager.connect(websocket, symbol, timeframe)
    state = StreamState(symbol, timeframe)

    try:
        # Step 1: Send initial historical data
        print(f"[WS] Sending initial data for {symbol}_{timeframe}")
        await send_initial_data(websocket, symbol, timeframe)

        # Step 2: Start streaming loop
        print(f"[WS] Starting stream loop for {symbol}_{timeframe}")
        await streaming_loop(websocket, state)

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {symbol}_{timeframe}")
    except Exception as e:
        print(f"[WS] Error in stream: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)},
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass
    finally:
        await manager.disconnect(websocket, symbol, timeframe)


# ─── STREAMING FUNCTIONS ────────────────────────────────────────────────────

async def send_initial_data(websocket: WebSocket, symbol: str, timeframe: str):
    """
    Send full historical data as first message
    """
    try:
        provider = get_provider()

        # Fetch historical bars
        if hasattr(provider, 'fetch_latest'):
            df = provider.fetch_latest(symbol, timeframe, bars=200)
        else:
            from datetime import timedelta
            end = datetime.now()
            start = end - timedelta(days=400)
            df = provider.fetch(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            df = df.tail(200)

        if df.empty:
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"No data available for {symbol}"},
                "timestamp": datetime.now().isoformat()
            })
            return

        # Run strategy pipeline
        df = run_strategy(df)

        # Convert to candles list
        candles = []
        for idx, row in df.iterrows():
            candle = {
                "datetime": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']),
                "ema_fast": float(row.get('ema_fast', 0)),
                "ema_slow": float(row.get('ema_slow', 0)),
                "ema_trend": float(row.get('ema_trend', 0)),
                "rsi": float(row.get('rsi', 50)),
                "signal": int(row.get('signal', 0)),
                "raw_signal": int(row.get('raw_signal', 0)),
                "regime": str(row.get('regime', 'ranging')),
                "ml_score": float(row.get('ml_score', 0.5)),
                "shock_detected": bool(row.get('shock_detected', False)),
            }
            candles.append(candle)

        # Send init message
        await websocket.send_json({
            "type": "init",
            "data": {
                "candles": candles,
                "symbol": symbol,
                "timeframe": timeframe
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[WS] Error sending initial data: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": f"Failed to load initial data: {str(e)}"},
            "timestamp": datetime.now().isoformat()
        })


async def streaming_loop(websocket: WebSocket, state: StreamState):
    """
    Main streaming loop - runs every 3 seconds
    """
    retry_count = 0
    max_retries = 3
    backoff_delay = 10

    while True:
        try:
            # Fetch latest bar
            provider = get_provider()

            if hasattr(provider, 'fetch_latest'):
                df = provider.fetch_latest(state.symbol, state.timeframe, bars=200)
            else:
                from datetime import timedelta
                end = datetime.now()
                start = end - timedelta(days=400)
                df = provider.fetch(state.symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                df = df.tail(200)

            if df.empty:
                await asyncio.sleep(3)
                continue

            # Run strategy pipeline
            df = run_strategy(df)

            # Check for shock detector
            shock_detected = False
            if hasattr(df.iloc[-1], 'shock_detected'):
                shock_detected = bool(df.iloc[-1]['shock_detected'])

            # Get latest candle
            latest_row = df.iloc[-1]
            latest_idx = df.index[-1]

            # Build candle data
            candle_data = {
                "datetime": latest_idx.isoformat() if hasattr(latest_idx, 'isoformat') else str(latest_idx),
                "open": float(latest_row['open']),
                "high": float(latest_row['high']),
                "low": float(latest_row['low']),
                "close": float(latest_row['close']),
                "volume": float(latest_row['volume']),
                "ema_fast": float(latest_row.get('ema_fast', 0)),
                "ema_slow": float(latest_row.get('ema_slow', 0)),
                "ema_trend": float(latest_row.get('ema_trend', 0)),
                "rsi": float(latest_row.get('rsi', 50)),
                "signal": int(latest_row.get('signal', 0)),
                "raw_signal": int(latest_row.get('raw_signal', 0)),
                "regime": str(latest_row.get('regime', 'ranging')),
                "ml_score": float(latest_row.get('ml_score', 0.5)),
                "shock_detected": shock_detected,
            }

            # Send candle update
            await websocket.send_json({
                "type": "candle_update",
                "data": candle_data,
                "timestamp": datetime.now().isoformat()
            })

            # Check for regime change
            current_regime = candle_data['regime']
            if state.previous_regime and current_regime != state.previous_regime:
                await websocket.send_json({
                    "type": "regime_change",
                    "data": {
                        "from": state.previous_regime,
                        "to": current_regime,
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now().isoformat()
                })
            state.previous_regime = current_regime

            # Check for confirmed signal
            if candle_data['signal'] != 0 and state.last_signal_bar != len(df):
                direction = "LONG" if candle_data['signal'] == 1 else "SHORT"
                entry_price = candle_data['close']

                # Calculate SL/TP (simplified)
                atr = abs(candle_data['high'] - candle_data['low']) * 1.5  # Approximate
                stop_loss = entry_price - (atr * candle_data['signal'])
                take_profit = entry_price + (atr * 2 * candle_data['signal'])

                await websocket.send_json({
                    "type": "signal_fired",
                    "data": {
                        "direction": direction,
                        "ml_score": candle_data['ml_score'],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "regime": current_regime
                    },
                    "timestamp": datetime.now().isoformat()
                })
                state.last_signal_bar = len(df)

            # Check for shock detection
            if shock_detected and state.shock_cooldown <= 0:
                await websocket.send_json({
                    "type": "shock_detected",
                    "data": {
                        "vol_ratio": 1.8,  # Approximate
                        "bars_cooldown": 3,
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now().isoformat()
                })
                state.shock_cooldown = 3

            # Decrement shock cooldown
            if state.shock_cooldown > 0:
                state.shock_cooldown -= 1

            # RL Decision
            rl_action = "HOLD"
            rl_confidence = 0.5
            rl_overrode = False

            if state.rl_executor:
                try:
                    # Create observation from latest data
                    obs = _create_rl_observation(df)
                    action, confidence = state.rl_executor.decide(obs)
                    rl_action = action
                    rl_confidence = confidence
                    # Check if overrode signal
                    if candle_data['signal'] != 0 and action == "HOLD":
                        rl_overrode = True
                except Exception as e:
                    print(f"[WS] RL decision error: {e}")

            await websocket.send_json({
                "type": "rl_decision",
                "data": {
                    "action": rl_action,
                    "confidence": rl_confidence,
                    "overrode_signal": rl_overrode
                },
                "timestamp": datetime.now().isoformat()
            })

            # Risk update
            risk_summary = state.risk_manager.summary()
            await websocket.send_json({
                "type": "risk_update",
                "data": {
                    "drawdown_pct": risk_summary.get('max_drawdown_pct', 0.0),
                    "current_capital": risk_summary.get('current_capital', 500000),
                    "daily_pnl_pct": 0.0,  # Simplified
                    "trading_halted": risk_summary.get('trading_halted', False)
                },
                "timestamp": datetime.now().isoformat()
            })

            # Check alpha decay
            # (Simplified - in real system would use AlphaDecayMonitor)
            if hasattr(state, 'alpha_monitor'):
                alpha_status = state.alpha_monitor.status()
                if alpha_status.get('decay_detected'):
                    await websocket.send_json({
                        "type": "alpha_warning",
                        "data": {
                            "rolling_sharpe": alpha_status.get('rolling_sharpe', 0.4),
                            "baseline_sharpe": alpha_status.get('baseline_sharpe', 1.8),
                            "rolling_accuracy": alpha_status.get('rolling_accuracy', 0.61)
                        },
                        "timestamp": datetime.now().isoformat()
                    })

            # Reset retry count on success
            retry_count = 0

            # Wait for next tick
            await asyncio.sleep(3)

        except Exception as e:
            print(f"[WS] Error in streaming loop: {e}")
            retry_count += 1

            if retry_count >= max_retries:
                # Send error and exit
                try:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Stream error after {max_retries} retries: {str(e)}"},
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    pass
                break

            # Exponential backoff
            await asyncio.sleep(backoff_delay * retry_count)


def _create_rl_observation(df) -> Dict[str, Any]:
    """
    Create observation dict for RL agent from dataframe
    """
    latest = df.iloc[-1]

    return {
        'close': float(latest['close']),
        'volume': float(latest['volume']),
        'rsi': float(latest.get('rsi', 50)),
        'ema_fast': float(latest.get('ema_fast', 0)),
        'ema_slow': float(latest.get('ema_slow', 0)),
        'regime': str(latest.get('regime', 'ranging')),
        'ml_score': float(latest.get('ml_score', 0.5)),
    }
