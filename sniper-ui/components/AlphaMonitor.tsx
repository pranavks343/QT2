'use client'

import { Card } from '@/components/ui/card'
import type { WSMessage } from '@/types/market'

type AlphaData = Extract<WSMessage, { type: 'alpha_warning' }>['data']

export function AlphaMonitor({ alpha }: { alpha: AlphaData | null }): React.JSX.Element {
  return <Card><div className='text-sm'>Rolling Sharpe: {alpha?.rolling_sharpe?.toFixed(2) ?? '-'}</div><div>Baseline: {alpha?.baseline_sharpe?.toFixed(2) ?? '-'}</div><div>Rolling Accuracy: {alpha ? `${(alpha.rolling_accuracy * 100).toFixed(1)}%` : '-'}</div>{alpha ? <div className='mt-1 text-yellow-300'>ALPHA DECAY DETECTED</div> : null}</Card>
}
