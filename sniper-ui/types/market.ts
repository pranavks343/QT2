export type RegimeType = 'trending_bull' | 'trending_bear' | 'ranging' | 'high_vol'
export type ActionType = 'LONG' | 'SHORT' | 'HOLD' | 'NOT_LOADED'
export type DirectionType = 'LONG' | 'SHORT'

export type Candle = {
  datetime: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  ema_fast: number
  ema_slow: number
  ema_trend: number
  rsi: number
  signal: number
  raw_signal: number
  regime: RegimeType
  ml_score: number
  shock_detected: boolean
  stop_loss?: number
  take_profit?: number
}

export type WSMessage =
  | { type: 'init'; data: { candles: Candle[]; symbol: string }; timestamp: string }
  | { type: 'candle_update'; data: Candle; timestamp: string }
  | { type: 'regime_change'; data: { from: RegimeType; to: RegimeType; timestamp: string }; timestamp: string }
  | { type: 'signal_fired'; data: { direction: DirectionType; ml_score: number; entry_price: number; stop_loss: number; take_profit: number; regime: RegimeType }; timestamp: string }
  | { type: 'shock_detected'; data: { vol_ratio: number; bars_cooldown: number; timestamp: string }; timestamp: string }
  | { type: 'rl_decision'; data: { action: ActionType; confidence: number; overrode_signal: boolean }; timestamp: string }
  | { type: 'alpha_warning'; data: { rolling_sharpe: number; baseline_sharpe: number; rolling_accuracy: number }; timestamp: string }
  | { type: 'risk_update'; data: { drawdown_pct: number; current_capital: number; daily_pnl_pct: number; trading_halted: boolean }; timestamp: string }
  | { type: 'error'; data: { message: string }; timestamp: string }

export type RiskSummary = {
  current_capital: number
  total_return_pct: number
  max_drawdown_pct: number
  total_trades: number
  accuracy: number
  trading_halted: boolean
  halt_reason: string | null
}

export type RLStatus = {
  last_action: ActionType
  confidence: number
  overrode_signal: boolean
  timestamp: string
}
