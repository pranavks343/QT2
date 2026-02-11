'use client'

import { Badge } from '@/components/ui/badge'

export function ShockIndicator({ active }: { active: boolean }): React.JSX.Element | null {
  if (!active) return null
  return <Badge className='border-orange-500 text-orange-300 animate-pulse'>⚡ SHOCK DETECTED</Badge>
}
