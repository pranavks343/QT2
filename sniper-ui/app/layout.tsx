import type { Metadata } from 'next'

import './globals.css'

export const metadata: Metadata = {
  title: 'Sniper UI',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>): React.JSX.Element {
  return (
    <html lang='en' className='dark'>
      <body className='font-mono bg-[#0a0c0f] text-zinc-100'>{children}</body>
    </html>
  )
}
