'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>): React.JSX.Element {
  return <div className={cn('rounded-xl border border-zinc-800 bg-zinc-900/70 p-3', className)} {...props} />
}
