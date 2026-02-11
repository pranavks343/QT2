'use client'

import { Card } from '@/components/ui/card'
import type { WSMessage } from '@/types/market'

type RiskData = Extract<WSMessage, { type: 'risk_update' }>['data']

export function RiskMonitor({ risk }: { risk: RiskData | null }): React.JSX.Element {
  const dd = risk?.drawdown_pct ?? 0
  const ddColor = dd < 0.05 ? 'text-emerald-400' : dd < 0.08 ? 'text-yellow-400' : 'text-red-400'
  return <Card><div className='text-sm'>Capital: ₹{(risk?.current_capital ?? 0).toFixed(0)}</div><div className={ddColor}>Drawdown: {(dd * 100).toFixed(2)}%</div><div>Daily P&L: {((risk?.daily_pnl_pct ?? 0) * 100).toFixed(2)}%</div>{risk?.trading_halted ? <div className='mt-1 text-red-400'>HALTED</div> : null}</Card>
}
