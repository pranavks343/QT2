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


class PaperOrderRequest(BaseModel):
    symbol: str
    qty: int = Field(..., ge=1)
    side: str
    order_type: str = "market"
    time_in_force: str = "day"
    market_price: float = Field(..., gt=0)
    limit_price: Optional[float] = None
    client_order_id: Optional[str] = None


class PaperOrderResponse(BaseModel):
    id: str
    symbol: str
    qty: int
    side: str
    status: str
    submitted_at: str
    filled_at: Optional[str] = None
    filled_avg_price: Optional[float] = None
    client_order_id: Optional[str] = None


class PositionResponse(BaseModel):
    symbol: str
    qty: int
    avg_entry_price: float
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
