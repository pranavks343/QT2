'use client'

import * as React from 'react'

export function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }): React.JSX.Element {
  return (
    <select
      className='h-9 rounded-md border border-zinc-700 bg-zinc-900 px-2 text-sm text-zinc-100'
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  )
}
