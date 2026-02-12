'use client'

import { useCallback, useEffect, useState } from 'react'

import { fetchData } from '@/lib/api'
import type { Candle } from '@/types/market'

export function useHistoricalData(symbol: string, timeframe: string, bars = 200): {
  candles: Candle[]
  isLoading: boolean
  error: string | null
  refetch: () => Promise<void>
} {
  const [candles, setCandles] = useState<Candle[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refetch = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const rows = await fetchData(symbol, bars, timeframe)
      setCandles(rows)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'failed to fetch')
    } finally {
      setIsLoading(false)
    }
  }, [symbol, timeframe, bars])

  useEffect(() => {
    void refetch()
  }, [refetch])

  return { candles, isLoading, error, refetch }
}
