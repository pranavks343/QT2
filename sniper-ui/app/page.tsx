'use client'

import { useMemo, useState } from 'react'

import { AlphaMonitor } from '@/components/AlphaMonitor'
import { CandlestickChart } from '@/components/CandlestickChart'
import { ConnectionStatus } from '@/components/ConnectionStatus'
import { OHLCVHeader } from '@/components/OHLCVHeader'
import { RLStatus } from '@/components/RLStatus'
import { RSIPanel } from '@/components/RSIPanel'
import { RegimeBadge } from '@/components/RegimeBadge'
import { RiskMonitor } from '@/components/RiskMonitor'
import { ShockIndicator } from '@/components/ShockIndicator'
import { SignalAlert } from '@/components/SignalAlert'
import { SymbolSelector } from '@/components/SymbolSelector'
import { useHistoricalData } from '@/hooks/useHistoricalData'
import { useMarketWebSocket } from '@/hooks/useMarketWebSocket'
import type { Candle } from '@/types/market'

export default function Page(): React.JSX.Element {
  const [symbol, setSymbol] = useState('NIFTY')
  const [timeframe, setTimeframe] = useState('5m')
  const [hovered, setHovered] = useState<Candle | null>(null)
  const historical = useHistoricalData(symbol, timeframe)
  const live = useMarketWebSocket(symbol, timeframe)

  const candles = useMemo(() => (live.candles.length > 0 ? live.candles : historical.candles), [live.candles, historical.candles])
  const activeCandle = hovered ?? live.latestCandle ?? candles[candles.length - 1] ?? null

  return (
    <main className='flex h-screen flex-col gap-2 overflow-hidden p-3'>
      <div className='flex items-center justify-between gap-2'>
        <div className='font-semibold'>SNIPER</div>
        <OHLCVHeader candle={activeCandle} />
        <RegimeBadge regime={live.regime} />
        <SymbolSelector symbol={symbol} timeframe={timeframe} onSymbol={setSymbol} onFrame={setTimeframe} />
        <ConnectionStatus isConnected={live.isConnected} isReconnecting={live.isReconnecting} />
      </div>
      <div className='flex-1 overflow-hidden rounded border border-zinc-800'>
        <CandlestickChart candles={candles} latestCandle={live.latestCandle} regime={live.regime} onHover={setHovered} />
      </div>
      <div className='rounded border border-zinc-800'>
        <RSIPanel candles={candles} />
      </div>
      <div className='grid grid-cols-4 gap-2'>
        <RLStatus rl={live.lastRLDecision} />
        <RiskMonitor risk={live.riskStatus} />
        <AlphaMonitor alpha={live.alphaWarning} />
        <ShockIndicator active={live.shockActive} />
      </div>
      <SignalAlert signal={live.lastSignal} />
    </main>
  )
}
