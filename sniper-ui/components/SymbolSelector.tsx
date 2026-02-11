'use client'

import { Select } from '@/components/ui/select'

const SYMBOLS = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN']
const FRAMES = ['1m', '5m', '15m', '1d']

export function SymbolSelector({ symbol, timeframe, onSymbol, onFrame }: { symbol: string; timeframe: string; onSymbol: (v: string) => void; onFrame: (v: string) => void }): React.JSX.Element {
  return <div className='flex items-center gap-2'><Select value={symbol} onChange={onSymbol} options={SYMBOLS} /><Select value={timeframe} onChange={onFrame} options={FRAMES} /></div>
}
