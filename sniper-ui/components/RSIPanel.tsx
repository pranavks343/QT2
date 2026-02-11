'use client'

import { useEffect, useRef } from 'react'
import { ColorType, createChart, type UTCTimestamp } from 'lightweight-charts'
import type { Candle } from '@/types/market'

function toTime(value: string): UTCTimestamp {
  return Math.floor(new Date(value).getTime() / 1000) as UTCTimestamp
}

export function RSIPanel({ candles }: { candles: Candle[] }): React.JSX.Element {
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      height: 120,
      width: ref.current.clientWidth,
      layout: { background: { type: ColorType.Solid, color: '#0a0c0f' }, textColor: '#c8cdd8' },
      grid: { vertLines: { color: '#1e2430' }, horzLines: { color: '#1e2430' } },
    })
    const line = chart.addLineSeries({ color: '#00d4ff' })
    line.setData(candles.map((c) => ({ time: toTime(c.datetime), value: c.rsi })))
    line.createPriceLine({ price: 30, color: '#666' })
    line.createPriceLine({ price: 70, color: '#666' })
    return () => chart.remove()
  }, [candles])
  return <div ref={ref} className='h-[120px] w-full' />
}
