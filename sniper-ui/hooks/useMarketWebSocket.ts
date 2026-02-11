"use client"

import { useState, useEffect, useRef, useCallback } from 'react'
import type { Candle, WSMessage, RegimeType, ActionType } from '@/types/market'

type SignalData = {
  direction: string
  ml_score: number
  entry_price: number
  stop_loss: number
  take_profit: number
  regime: RegimeType
}

type RLDecisionData = {
  action: ActionType
  confidence: number
  overrode_signal: boolean
}

type RiskData = {
  drawdown_pct: number
  current_capital: number
  daily_pnl_pct: number
  trading_halted: boolean
}

export function useMarketWebSocket(symbol: string, timeframe: string) {
  const [candles, setCandles] = useState<Candle[]>([])
  const [latestCandle, setLatestCandle] = useState<Candle | null>(null)
  const [lastSignal, setLastSignal] = useState<SignalData | null>(null)
  const [lastRLDecision, setLastRLDecision] = useState<RLDecisionData | null>(null)
  const [regime, setRegime] = useState<RegimeType>('ranging')
  const [riskStatus, setRiskStatus] = useState<RiskData | null>(null)
  const [shockActive, setShockActive] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL}/ws/${symbol}/${timeframe}`
    console.log(`[WS] Connecting to ${wsUrl}`)

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('[WS] Connected')
      setIsConnected(true)
      setError(null)
      reconnectAttempts.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)

        switch (message.type) {
          case 'init':
            console.log(`[WS] Received ${message.data.candles.length} initial candles`)
            setCandles(message.data.candles)
            if (message.data.candles.length > 0) {
              const latest = message.data.candles[message.data.candles.length - 1]
              setLatestCandle(latest)
              setRegime(latest.regime)
            }
            break

          case 'candle_update':
            setLatestCandle(message.data)
            setRegime(message.data.regime)
            setCandles(prev => {
              // Update last candle if same datetime, otherwise append
              if (prev.length > 0 && prev[prev.length - 1].datetime === message.data.datetime) {
                return [...prev.slice(0, -1), message.data]
              }
              return [...prev, message.data]
            })
            break

          case 'regime_change':
            console.log(`[WS] Regime changed: ${message.data.from} → ${message.data.to}`)
            setRegime(message.data.to)
            break

          case 'signal_fired':
            console.log(`[WS] Signal fired: ${message.data.direction}`)
            setLastSignal(message.data)
            // Clear after 8 seconds
            setTimeout(() => setLastSignal(null), 8000)
            break

          case 'shock_detected':
            console.log('[WS] Shock detected')
            setShockActive(true)
            // Clear after cooldown
            setTimeout(() => setShockActive(false), message.data.bars_cooldown * 3000)
            break

          case 'rl_decision':
            setLastRLDecision(message.data)
            break

          case 'risk_update':
            setRiskStatus(message.data)
            break

          case 'alpha_warning':
            console.warn('[WS] Alpha decay warning', message.data)
            break

          case 'error':
            console.error('[WS] Server error:', message.data.message)
            setError(message.data.message)
            break

          default:
            console.warn('[WS] Unknown message type:', message)
        }
      } catch (err) {
        console.error('[WS] Failed to parse message:', err)
      }
    }

    ws.onerror = (event) => {
      console.error('[WS] Error:', event)
      setIsConnected(false)
      setError('WebSocket connection error')
    }

    ws.onclose = () => {
      console.log('[WS] Connection closed')
      setIsConnected(false)

      // Exponential backoff reconnect
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000)
      reconnectAttempts.current++

      console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`)

      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, delay)
    }

    wsRef.current = ws
  }, [symbol, timeframe])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect])

  return {
    candles,
    latestCandle,
    lastSignal,
    lastRLDecision,
    regime,
    riskStatus,
    shockActive,
    isConnected,
    error,
  }
}
