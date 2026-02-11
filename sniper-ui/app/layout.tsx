import type { Metadata } from 'next'
import { IBM_Plex_Mono } from 'next/font/google'

import './globals.css'

const mono = IBM_Plex_Mono({ subsets: ['latin'], weight: ['400', '500', '600'] })

export const metadata: Metadata = {
  title: 'Sniper UI',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>): React.JSX.Element {
  return (
    <html lang='en' className='dark'>
      <body className={`${mono.className} bg-[#0a0c0f] text-zinc-100`}>{children}</body>
    </html>
  )
}
