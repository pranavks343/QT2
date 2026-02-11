"""FastAPI REST routes for market + analytics."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.engine import BacktestEngine, CPCVValidator
from config import COSTS
from data.base_provider import get_provider
from execution.cost_model import ExecutionCostModel, TradeTicket
from risk.risk_manager import RiskManager
from rl.rl_executor import RLExecutor
from strategy.core_engine import run_strategy

router = APIRouter()

SYMBOLS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]


class SymbolsResponse(BaseModel):
    symbols: List[str]


class Candle(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema_fast: float = 0
    ema_slow: float = 0
    ema_trend: float = 0
    rsi: float = 0
    signal: int = 0
    raw_signal: int = 0
    regime: str = "ranging"
    ml_score: float = 0
    shock_detected: bool = False
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    bars: int
    candles: List[Candle]
    current_regime: str
    last_updated: str


class BacktestResponse(BaseModel):
    total_trades: int
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    dsr_confidence: float
    total_pnl: float
    total_costs: float
    verdict: str
    accuracy_by_regime: Dict[str, float]
    cpcv: Dict[str, Any]


class RLStatusResponse(BaseModel):
    last_action: str
    confidence: float = 0
    overrode_signal: bool = False
    timestamp: Optional[str] = None


class CostSummaryResponse(BaseModel):
    entry_cost: float
    exit_cost: float
    total_cost: float
    gross_pnl: float
    net_pnl: float
    breakeven_move: float


class RiskSummaryResponse(BaseModel):
    initial_capital: float
    current_capital: float
    total_return_pct: float
    peak_capital: float
    max_drawdown_pct: float
    total_trades: int
    wins: int
    losses: int
    accuracy: float
    open_positions: int
    trading_halted: bool
    halt_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str


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


def _to_candles(df) -> List[Candle]:
    out: List[Candle] = []
    for idx, row in df.iterrows():
        out.append(
            Candle(
                datetime=idx.isoformat(),
                open=_num(row["open"]),
                high=_num(row["high"]),
                low=_num(row["low"]),
                close=_num(row["close"]),
                volume=_num(row.get("volume", 0)),
                ema_fast=_num(row.get("ema_fast", 0)),
                ema_slow=_num(row.get("ema_slow", 0)),
                ema_trend=_num(row.get("ema_trend", 0)),
                rsi=_num(row.get("rsi", 0)),
                signal=int(row.get("signal", 0) or 0),
                raw_signal=int(row.get("raw_signal", 0) or 0),
                regime=str(row.get("regime", "ranging") or "ranging"),
                ml_score=_num(row.get("ml_score", 0)),
                shock_detected=False if _is_nan(row.get("shock_detected", False)) else bool(row.get("shock_detected", False)),
                stop_loss=None if row.get("stop_loss") is None or _is_nan(row.get("stop_loss")) else _num(row.get("stop_loss")),
                take_profit=None if row.get("take_profit") is None or _is_nan(row.get("take_profit")) else _num(row.get("take_profit")),
            )
        )
    return out


@router.get("/symbols", response_model=SymbolsResponse)
async def symbols() -> SymbolsResponse:
    return SymbolsResponse(symbols=SYMBOLS)


@router.get("/data", response_model=MarketDataResponse)
async def data(symbol: str, bars: int = 200, timeframe: str = "5m") -> MarketDataResponse:
    provider = get_provider()
    if hasattr(provider, "fetch_latest"):
        df = provider.fetch_latest(symbol=symbol, timeframe=timeframe, bars=bars)
    else:
        df = provider.fetch(symbol=symbol, start=None, end=None).tail(bars)
    enriched = run_strategy(df.copy())
    candles = _to_candles(enriched.tail(bars))
    return MarketDataResponse(
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        candles=candles,
        current_regime=candles[-1].regime if candles else "ranging",
        last_updated=datetime.utcnow().isoformat(),
    )


@router.get("/backtest", response_model=BacktestResponse)
async def backtest(symbol: str = "NIFTY", bars: int = Query(2000, ge=300)) -> BacktestResponse:
    provider = get_provider()
    if hasattr(provider, "fetch_latest"):
        df = provider.fetch_latest(symbol=symbol, timeframe="5m", bars=bars)
    else:
        df = provider.fetch(symbol=symbol, start=None, end=None).tail(bars)
    strategy_df = run_strategy(df.copy())

    engine = BacktestEngine(risk_manager=RiskManager())
    results = engine.run(strategy_df)
    cpcv_stats = CPCVValidator().run(strategy_df)
    verdict = results.verdict()["overall"].replace("✅ ", "").replace("⚠️ ", "").replace("❌ ", "")

    return BacktestResponse(
        total_trades=results.total_trades(),
        accuracy=round(results.accuracy(), 4),
        sharpe_ratio=round(results.sharpe_ratio(), 4),
        max_drawdown=round(float(results.max_drawdown()), 4),
        profit_factor=round(float(results.profit_factor()), 4),
        dsr_confidence=round(float(results.deflated_sharpe_ratio()), 4),
        total_pnl=round(float(results.total_pnl()), 2),
        total_costs=round(float(results.total_costs()), 2),
        verdict=verdict,
        accuracy_by_regime=results.accuracy_by_regime(),
        cpcv={
            "mean_sharpe": round(float(cpcv_stats.get("mean_sharpe", 0)), 4),
            "std_sharpe": round(float(cpcv_stats.get("std_sharpe", 0)), 4),
            "consistent": bool(cpcv_stats.get("is_robust", False)),
        },
    )


@router.get("/rl-status", response_model=RLStatusResponse)
async def rl_status() -> RLStatusResponse:
    executor = RLExecutor()
    if not executor.load("models/rl_agent"):
        return RLStatusResponse(last_action="NOT_LOADED")
    return RLStatusResponse(
        last_action="HOLD",
        confidence=0.5,
        overrode_signal=False,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/costs", response_model=CostSummaryResponse)
async def costs(
    entry: float,
    exit: float,
    lots: int = Query(1, ge=1),
    direction: int = Query(1, ge=-1, le=1),
) -> CostSummaryResponse:
    if direction not in (-1, 1):
        raise HTTPException(status_code=422, detail="direction must be 1 (LONG) or -1 (SHORT)")

    model = ExecutionCostModel()
    ticket = TradeTicket(
        direction=direction,
        entry_price=entry,
        exit_price=exit,
        lots=lots,
        lot_size=COSTS["lot_size_nifty"],
        is_option=True,
    )
    c = model.compute(ticket)
    gross = (exit - entry) * lots * COSTS["lot_size_nifty"] if direction == 1 else (entry - exit) * lots * COSTS["lot_size_nifty"]
    net = gross - c.total_cost
    return CostSummaryResponse(
        entry_cost=round(c.entry_stt + c.entry_brokerage + c.entry_exchange + c.entry_sebi + c.entry_stamp_duty + c.entry_slippage, 4),
        exit_cost=round(c.exit_stt + c.exit_brokerage + c.exit_exchange + c.exit_sebi + c.exit_stamp_duty + c.exit_slippage + c.gst, 4),
        total_cost=round(c.total_cost, 4),
        gross_pnl=round(gross, 4),
        net_pnl=round(net, 4),
        breakeven_move=round(c.total_cost / (lots * COSTS["lot_size_nifty"]), 4) if lots > 0 else 0.0,
    )


@router.get("/risk-summary", response_model=RiskSummaryResponse)
async def risk_summary() -> RiskSummaryResponse:
    return RiskSummaryResponse(**RiskManager().summary())


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())
