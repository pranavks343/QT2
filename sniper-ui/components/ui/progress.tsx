'use client'

import * as React from 'react'

export function Progress({ value }: { value: number }): React.JSX.Element {
  return (
    <div className='h-2 w-full rounded bg-zinc-800'>
      <div className='h-2 rounded bg-blue-500' style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }} />
    </div>
  )
}
