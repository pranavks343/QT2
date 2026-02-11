'use client'

import { useEffect, useRef, useState } from 'react'

import type { Candle, RegimeType, WSMessage } from '@/types/market'

type SignalData = Extract<WSMessage, { type: 'signal_fired' }>['data']
type RLData = Extract<WSMessage, { type: 'rl_decision' }>['data']
type RiskData = Extract<WSMessage, { type: 'risk_update' }>['data']
type AlphaData = Extract<WSMessage, { type: 'alpha_warning' }>['data']

export function useMarketWebSocket(symbol: string, timeframe: string) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<number>(1000)
  const timerRef = useRef<number | null>(null)

  const [candles, setCandles] = useState<Candle[]>([])
  const [latestCandle, setLatestCandle] = useState<Candle | null>(null)
  const [lastSignal, setLastSignal] = useState<SignalData | null>(null)
  const [lastRLDecision, setLastRLDecision] = useState<RLData | null>(null)
  const [regime, setRegime] = useState<RegimeType>('ranging')
  const [riskStatus, setRiskStatus] = useState<RiskData | null>(null)
  const [alphaWarning, setAlphaWarning] = useState<AlphaData | null>(null)
  const [shockActive, setShockActive] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isReconnecting, setIsReconnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isCancelled = false

    const connect = (): void => {
      const base = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000'
      const ws = new WebSocket(`${base}/ws/${symbol}/${timeframe}`)
      wsRef.current = ws

      ws.onopen = () => {
        if (isCancelled) return
        setIsConnected(true)
        setIsReconnecting(false)
        reconnectRef.current = 1000
      }

      ws.onclose = () => {
        if (isCancelled) return
        setIsConnected(false)
        setIsReconnecting(true)
        window.setTimeout(connect, reconnectRef.current)
        reconnectRef.current = Math.min(reconnectRef.current * 2, 30000)
      }

      ws.onerror = () => {
        setError('WebSocket error')
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WSMessage
        if (msg.type === 'init') {
          setCandles(msg.data.candles)
          if (msg.data.candles.length > 0) {
            setLatestCandle(msg.data.candles[msg.data.candles.length - 1])
          }
          return
        }
        if (msg.type === 'candle_update') {
          setLatestCandle(msg.data)
          setCandles((prev) => {
            const last = prev[prev.length - 1]
            if (last && last.datetime === msg.data.datetime) {
              return [...prev.slice(0, -1), msg.data]
            }
            return [...prev, msg.data].slice(-500)
          })
          return
        }
        if (msg.type === 'regime_change') {
          setRegime(msg.data.to)
          return
        }
        if (msg.type === 'signal_fired') {
          setLastSignal(msg.data)
          if (timerRef.current) window.clearTimeout(timerRef.current)
          timerRef.current = window.setTimeout(() => setLastSignal(null), 8000)
          return
        }
        if (msg.type === 'shock_detected') {
          setShockActive(true)
          window.setTimeout(() => setShockActive(false), msg.data.bars_cooldown * 3000)
          return
        }
        if (msg.type === 'rl_decision') {
          setLastRLDecision(msg.data)
          return
        }
        if (msg.type === 'alpha_warning') {
          setAlphaWarning(msg.data)
          return
        }
        if (msg.type === 'risk_update') {
          setRiskStatus(msg.data)
          return
        }
        if (msg.type === 'error') {
          setError(msg.data.message)
        }
      }
    }

    connect()

    return () => {
      isCancelled = true
      if (timerRef.current) window.clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [symbol, timeframe])

  return {
    candles,
    latestCandle,
    lastSignal,
    lastRLDecision,
    regime,
    riskStatus,
    alphaWarning,
    shockActive,
    isConnected,
    isReconnecting,
    error,
  }
}
