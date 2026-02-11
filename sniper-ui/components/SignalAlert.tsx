'use client'

import { useEffect, useRef } from 'react'

import { Card } from '@/components/ui/card'
import type { WSMessage } from '@/types/market'

type SignalData = Extract<WSMessage, { type: 'signal_fired' }>['data']

export function SignalAlert({ signal }: { signal: SignalData | null }): React.JSX.Element | null {
  const audioContextRef = useRef<AudioContext | null>(null)

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        void audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!signal) return

    if (audioContextRef.current === null) {
      audioContextRef.current = new AudioContext()
    }

    const ctx = audioContextRef.current
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.frequency.value = signal.direction === 'LONG' ? 880 : 440
    gain.gain.value = 0.03

    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start()
    osc.stop(ctx.currentTime + 0.1)

    return () => {
      osc.disconnect()
      gain.disconnect()
    }
  }, [signal])

  if (!signal) return null
  return (
    <Card className={`fixed bottom-4 right-4 w-80 transition-transform ${signal.direction === 'LONG' ? 'border-emerald-500' : 'border-red-500'}`}>
      <div className='text-sm font-semibold'>{signal.direction} SIGNAL</div>
      <div className='text-xs'>Entry: {signal.entry_price.toFixed(2)} SL: {signal.stop_loss.toFixed(2)} TP: {signal.take_profit.toFixed(2)}</div>
    </Card>
  )
}
