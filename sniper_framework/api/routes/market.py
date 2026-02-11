"""
Market API Routes
Provides REST endpoints for market data, backtesting, and system status
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import RISK, COSTS
from data.base_provider import get_provider
from strategy.core_engine import run_strategy
from backtest.engine import BacktestEngine, CPCVValidator
from risk.risk_manager import RiskManager
from execution.cost_model import ExecutionCostModel


router = APIRouter()


# ─── PYDANTIC MODELS ────────────────────────────────────────────────────────

class SymbolsResponse(BaseModel):
    symbols: List[str] = Field(..., description="List of available trading symbols")


class CandleData(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_trend: Optional[float] = None
    rsi: Optional[float] = None
    signal: Optional[int] = None
    raw_signal: Optional[int] = None
    regime: Optional[str] = None
    ml_score: Optional[float] = None
    shock_detected: Optional[bool] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    bars: int
    candles: List[CandleData]
    current_regime: Optional[str] = None
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
    cpcv: Optional[Dict[str, Any]] = None


class RLStatusResponse(BaseModel):
    last_action: str
    confidence: float
    overrode_signal: bool
    timestamp: str


class CostBreakdownResponse(BaseModel):
    entry_costs: float
    exit_costs: float
    total_costs: float
    net_pnl: float
    net_pnl_pct: float
    details: Dict[str, Any]


class RiskSummaryResponse(BaseModel):
    current_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    accuracy: float
    trading_halted: bool
    halt_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# ─── ENDPOINTS ──────────────────────────────────────────────────────────────

@router.get("/symbols", response_model=SymbolsResponse)
async def get_symbols():
    """
    Returns list of available trading symbols
    """
    symbols = [
        "NIFTY",
        "BANKNIFTY",
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN"
    ]
    return SymbolsResponse(symbols=symbols)


@router.get("/data", response_model=MarketDataResponse)
async def get_market_data(
    symbol: str = Query(..., description="Trading symbol"),
    bars: int = Query(200, ge=50, le=5000, description="Number of bars to fetch"),
    timeframe: str = Query("5m", description="Timeframe (1m, 5m, 15m, 1d)")
):
    """
    Fetch market data and run full strategy pipeline
    Returns OHLCV with all indicators, signals, and regime detection
    """
    try:
        # Get data provider
        provider = get_provider()

        # For yfinance provider, use fetch_latest if available
        if hasattr(provider, 'fetch_latest'):
            df = provider.fetch_latest(symbol, timeframe, bars)
        else:
            # Fallback to standard fetch with date range
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=bars * 2)  # Approximate
            df = provider.fetch(
                symbol=symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d")
            )
            df = df.tail(bars)

        if df.empty:
            raise HTTPException(status_code=404, message=f"No data found for {symbol}")

        # Run strategy pipeline
        df = run_strategy(df)

        # Convert to response format
        candles = []
        for idx, row in df.iterrows():
            candle = CandleData(
                datetime=idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                ema_fast=float(row.get('ema_fast', 0)),
                ema_slow=float(row.get('ema_slow', 0)),
                ema_trend=float(row.get('ema_trend', 0)),
                rsi=float(row.get('rsi', 50)),
                signal=int(row.get('signal', 0)),
                raw_signal=int(row.get('raw_signal', 0)),
                regime=str(row.get('regime', 'ranging')),
                ml_score=float(row.get('ml_score', 0.5)),
                shock_detected=bool(row.get('shock_detected', False)),
                stop_loss=float(row.get('stop_loss')) if 'stop_loss' in row and row['stop_loss'] else None,
                take_profit=float(row.get('take_profit')) if 'take_profit' in row and row['take_profit'] else None,
            )
            candles.append(candle)

        current_regime = str(df.iloc[-1].get('regime', 'ranging')) if len(df) > 0 else 'ranging'

        return MarketDataResponse(
            symbol=symbol,
            timeframe=timeframe,
            bars=len(candles),
            candles=candles,
            current_regime=current_regime,
            last_updated=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")


@router.get("/backtest", response_model=BacktestResponse)
async def run_backtest(
    symbol: str = Query("NIFTY", description="Trading symbol"),
    bars: int = Query(2000, ge=1000, le=10000, description="Number of bars for backtest")
):
    """
    Run full backtest pipeline with CPCV validation
    """
    try:
        # Fetch data
        provider = get_provider()
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=bars * 2)

        df = provider.fetch(
            symbol=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )
        df = df.tail(bars)

        if len(df) < bars * 0.5:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Got {len(df)} bars, expected at least {int(bars * 0.5)}"
            )

        # Run strategy
        df = run_strategy(df)

        # Run backtest
        risk = RiskManager()
        engine = BacktestEngine(risk_manager=risk)
        results = engine.run(df)

        # Run CPCV validation
        cpcv_validator = CPCVValidator()
        cpcv_results = cpcv_validator.run(df)

        # Extract accuracy by regime
        accuracy_by_regime = {}
        if hasattr(results, 'regime_stats') and results.regime_stats:
            for regime, stats in results.regime_stats.items():
                if 'accuracy' in stats:
                    accuracy_by_regime[regime] = float(stats['accuracy'])

        # Determine verdict
        verdict = "FULLY QUALIFIED"
        if results.sharpe_ratio < 1.5:
            verdict = "UNDERPERFORMING"
        elif results.accuracy < 0.80:
            verdict = "LOW ACCURACY"
        elif results.max_drawdown > 0.10:
            verdict = "EXCESSIVE DRAWDOWN"

        return BacktestResponse(
            total_trades=results.total_trades,
            accuracy=results.accuracy,
            sharpe_ratio=results.sharpe_ratio,
            max_drawdown=results.max_drawdown,
            profit_factor=results.profit_factor,
            dsr_confidence=results.dsr_confidence if hasattr(results, 'dsr_confidence') else 0.95,
            total_pnl=results.total_pnl,
            total_costs=results.total_costs if hasattr(results, 'total_costs') else 0,
            verdict=verdict,
            accuracy_by_regime=accuracy_by_regime,
            cpcv=cpcv_results if cpcv_results else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@router.get("/rl-status", response_model=RLStatusResponse)
async def get_rl_status():
    """
    Get current RL executor status and last decision
    """
    try:
        from rl.rl_executor import RLExecutor

        rl_executor = RLExecutor()
        if not rl_executor.load("models/rl_agent.zip"):
            return RLStatusResponse(
                last_action="NOT_LOADED",
                confidence=0.0,
                overrode_signal=False,
                timestamp=datetime.now().isoformat()
            )

        # Get last decision if available
        last_action = getattr(rl_executor, 'last_action', 'HOLD')
        confidence = getattr(rl_executor, 'last_confidence', 0.5)
        overrode_signal = getattr(rl_executor, 'last_override', False)

        return RLStatusResponse(
            last_action=str(last_action),
            confidence=float(confidence),
            overrode_signal=bool(overrode_signal),
            timestamp=datetime.now().isoformat()
        )

    except ImportError:
        return RLStatusResponse(
            last_action="NOT_LOADED",
            confidence=0.0,
            overrode_signal=False,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL status error: {str(e)}")


@router.get("/costs", response_model=CostBreakdownResponse)
async def calculate_costs(
    entry: float = Query(..., description="Entry price"),
    exit: float = Query(..., description="Exit price"),
    lots: int = Query(1, ge=1, le=100, description="Number of lots"),
    direction: int = Query(1, description="Direction (1=LONG, -1=SHORT)")
):
    """
    Calculate execution costs for a trade
    """
    try:
        cost_model = ExecutionCostModel()

        # Calculate costs
        breakdown = cost_model.calculate_trade_cost(
            entry_price=entry,
            exit_price=exit,
            lots=lots,
            direction=direction
        )

        gross_pnl = (exit - entry) * direction * lots * COSTS['lot_size_nifty']
        net_pnl = gross_pnl - breakdown.total_costs
        net_pnl_pct = (net_pnl / (entry * lots * COSTS['lot_size_nifty'])) * 100

        return CostBreakdownResponse(
            entry_costs=breakdown.entry_costs,
            exit_costs=breakdown.exit_costs,
            total_costs=breakdown.total_costs,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            details={
                "brokerage": breakdown.brokerage if hasattr(breakdown, 'brokerage') else 0,
                "stt": breakdown.stt if hasattr(breakdown, 'stt') else 0,
                "exchange_fee": breakdown.exchange_fee if hasattr(breakdown, 'exchange_fee') else 0,
                "gst": breakdown.gst if hasattr(breakdown, 'gst') else 0,
                "slippage": breakdown.slippage if hasattr(breakdown, 'slippage') else 0,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost calculation error: {str(e)}")


@router.get("/risk-summary", response_model=RiskSummaryResponse)
async def get_risk_summary():
    """
    Get current risk manager summary
    """
    try:
        risk = RiskManager()
        summary = risk.summary()

        return RiskSummaryResponse(
            current_capital=summary.get('current_capital', RISK['capital']),
            total_return_pct=summary.get('total_return_pct', 0.0),
            max_drawdown_pct=summary.get('max_drawdown_pct', 0.0),
            total_trades=summary.get('total_trades', 0),
            accuracy=summary.get('accuracy', 0.0),
            trading_halted=summary.get('trading_halted', False),
            halt_reason=summary.get('halt_reason', None)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk summary error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )
