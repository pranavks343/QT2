'use client'

import { useEffect, useRef } from 'react'
import { ColorType, createChart, LineStyle, type IChartApi, type ISeriesApi, type UTCTimestamp } from 'lightweight-charts'

import type { Candle, RegimeType } from '@/types/market'

type Props = { candles: Candle[]; latestCandle: Candle | null; regime: RegimeType; onHover: (c: Candle | null) => void }

function toTime(value: string): UTCTimestamp {
  return Math.floor(new Date(value).getTime() / 1000) as UTCTimestamp
}

export function CandlestickChart({ candles, latestCandle, regime, onHover }: Props): React.JSX.Element {
  const ref = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const emaFastRef = useRef<ISeriesApi<'Line'> | null>(null)
  const emaSlowRef = useRef<ISeriesApi<'Line'> | null>(null)
  const emaTrendRef = useRef<ISeriesApi<'Line'> | null>(null)
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const candlesRef = useRef<Candle[]>([])
  const latestCandleRef = useRef<Candle | null>(null)
  const onHoverRef = useRef(onHover)

  useEffect(() => {
    candlesRef.current = candles
    latestCandleRef.current = latestCandle
    onHoverRef.current = onHover
  }, [candles, latestCandle, onHover])

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      layout: { background: { type: ColorType.Solid, color: '#0a0c0f' }, textColor: '#c8cdd8' },
      grid: { vertLines: { color: '#1e2430' }, horzLines: { color: '#1e2430' } },
      width: ref.current.clientWidth,
      height: 420,
    })
    chartRef.current = chart
    candleRef.current = chart.addCandlestickSeries({ upColor: '#00e676', downColor: '#ff3d5a', borderVisible: false, wickUpColor: '#00e676', wickDownColor: '#ff3d5a' })
    emaFastRef.current = chart.addLineSeries({ color: '#ffd54f', lineWidth: 1 })
    emaSlowRef.current = chart.addLineSeries({ color: '#ff8a65', lineWidth: 1 })
    emaTrendRef.current = chart.addLineSeries({ color: '#ce93d8', lineWidth: 1, lineStyle: LineStyle.Dashed })
    volumeRef.current = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' })

    const crosshairHandler = (param: { time?: UTCTimestamp }): void => {
      if (!param.time) {
        onHoverRef.current(latestCandleRef.current)
        return
      }
      const hit = candlesRef.current.find((c) => toTime(c.datetime) === param.time) ?? null
      onHoverRef.current(hit)
    }

    chart.subscribeCrosshairMove(crosshairHandler)

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.resize(ref.current.clientWidth, 420)
    })
    ro.observe(ref.current)

    return () => {
      ro.disconnect()
      chart.unsubscribeCrosshairMove(crosshairHandler)
      chart.remove()
    }
  }, [])

  useEffect(() => {
    if (!candleRef.current) return
    candleRef.current.setData(candles.map((c) => ({ time: toTime(c.datetime), open: c.open, high: c.high, low: c.low, close: c.close })))
    emaFastRef.current?.setData(candles.map((c) => ({ time: toTime(c.datetime), value: c.ema_fast })))
    emaSlowRef.current?.setData(candles.map((c) => ({ time: toTime(c.datetime), value: c.ema_slow })))
    emaTrendRef.current?.setData(candles.map((c) => ({ time: toTime(c.datetime), value: c.ema_trend })))
    volumeRef.current?.setData(candles.map((c) => ({ time: toTime(c.datetime), value: c.volume, color: c.close >= c.open ? '#00e676' : '#ff3d5a' })))
  }, [candles])

  useEffect(() => {
    if (!latestCandle || !candleRef.current) return
    const time = toTime(latestCandle.datetime)
    candleRef.current.update({ time, open: latestCandle.open, high: latestCandle.high, low: latestCandle.low, close: latestCandle.close })
    emaFastRef.current?.update({ time, value: latestCandle.ema_fast })
    emaSlowRef.current?.update({ time, value: latestCandle.ema_slow })
    emaTrendRef.current?.update({ time, value: latestCandle.ema_trend })
    volumeRef.current?.update({ time, value: latestCandle.volume, color: latestCandle.close >= latestCandle.open ? '#00e676' : '#ff3d5a' })
  }, [latestCandle])

  useEffect(() => {
    const color = regime === 'trending_bull' ? '#00e676' : regime === 'trending_bear' ? '#ff3d5a' : regime === 'high_vol' ? '#ff9800' : '#ffd54f'
    chartRef.current?.applyOptions({ crosshair: { vertLine: { color }, horzLine: { color } } })
  }, [regime])

  return <div ref={ref} className='h-[420px] w-full' />
}
