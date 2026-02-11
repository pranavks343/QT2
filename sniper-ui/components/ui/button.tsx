'use client'

import * as React from 'react'

import { cn } from '@/lib/utils'

export function Button({ className, type = 'button', ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>): React.JSX.Element {
  return <button type={type} className={cn('rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm', className)} {...props} />
}
