'use client'

import type { Candle } from '@/types/market'

export function OHLCVHeader({ candle }: { candle: Candle | null }): React.JSX.Element {
  if (!candle) return <div className='text-xs text-zinc-400'>No candle</div>
  const oc = candle.close >= candle.open ? 'text-emerald-400' : 'text-red-400'
  return (
    <div className='font-mono text-xs'>
      O: <span className={oc}>{candle.open.toFixed(2)}</span> H: <span className='text-emerald-400'>{candle.high.toFixed(2)}</span> L: <span className='text-red-400'>{candle.low.toFixed(2)}</span> C: <span className={oc}>{candle.close.toFixed(2)}</span> V: <span>{candle.volume.toFixed(0)}</span>
    </div>
  )
}
