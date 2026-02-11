'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'

export function Button({ className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>): React.JSX.Element {
  return <button className={cn('rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm', className)} {...props} />
}
