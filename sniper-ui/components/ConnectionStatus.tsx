'use client'

import { Badge } from '@/components/ui/badge'

export function ConnectionStatus({ isConnected, isReconnecting }: { isConnected: boolean; isReconnecting: boolean }): React.JSX.Element {
  const label = isConnected ? 'LIVE' : isReconnecting ? 'RECONNECTING...' : 'DISCONNECTED'
  const color = isConnected ? 'bg-emerald-500' : isReconnecting ? 'bg-yellow-500' : 'bg-red-500'
  return <Badge className='gap-2 border-zinc-700'><span className={`h-2 w-2 rounded-full ${color} animate-pulse`} />{label}</Badge>
}
