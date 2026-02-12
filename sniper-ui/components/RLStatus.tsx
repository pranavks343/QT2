'use client'

import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import type { WSMessage } from '@/types/market'

type RLData = Extract<WSMessage, { type: 'rl_decision' }>['data']

export function RLStatus({ rl }: { rl: RLData | null }): React.JSX.Element {
  const action = rl?.action ?? 'NOT_LOADED'
  return <Card><div className='text-sm'>RL: {action}</div><Progress value={(rl?.confidence ?? 0) * 100} />{rl?.overrode_signal ? <Badge className='mt-2 border-yellow-500 text-yellow-400'>Overrode Signal</Badge> : null}</Card>
}
