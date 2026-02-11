'use client'
import { Badge } from '@/components/ui/badge'
import type { RegimeType } from '@/types/market'

const labels: Record<RegimeType, string> = { trending_bull: 'BULL TREND', trending_bear: 'BEAR TREND', ranging: 'RANGING', high_vol: 'HIGH VOL' }

export function RegimeBadge({ regime }: { regime: RegimeType }): React.JSX.Element {
  const cls = regime === 'trending_bull' ? 'border-emerald-500 text-emerald-400 animate-pulse' : regime === 'trending_bear' ? 'border-red-500 text-red-400 animate-pulse' : regime === 'high_vol' ? 'border-orange-500 text-orange-400 animate-pulse' : 'border-yellow-500 text-yellow-400'
  return <Badge className={cls}>{labels[regime]}</Badge>
}
