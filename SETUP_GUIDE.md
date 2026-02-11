# Sniper Trading Framework - Complete Setup Guide

## ✅ COMPLETED: Backend (FastAPI)

The backend is fully implemented and ready to run:

### Backend Structure
```
sniper_framework/
├── api/
│   ├── __init__.py
│   ├── main.py                          # FastAPI app entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   └── market.py                    # All REST endpoints
│   └── websocket/
│       ├── __init__.py
│       ├── connection_manager.py        # WebSocket connection handling
│       └── stream.py                    # Real-time streaming logic
└── run_server.py                        # Server runner script
```

### Backend Features
- ✅ REST API endpoints for market data, backtesting, RL status, costs, risk
- ✅ WebSocket streaming with real-time candle updates
- ✅ Regime change detection and broadcasting
- ✅ Signal fired notifications
- ✅ Shock detection alerts
- ✅ RL decision broadcasting
- ✅ Risk monitoring updates
- ✅ Alpha decay warnings
- ✅ YFinance provider with fetch_latest() for streaming
- ✅ Auto-reconnect and error handling

### Start the Backend
```bash
cd sniper_framework
python run_server.py
```

Server will run at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/{symbol}/{timeframe}

---

## 🔧 FRONTEND SETUP (Next.js + Shadcn UI)

### Step 1: Install Dependencies

```bash
cd sniper-ui
npm install
```

### Step 2: Install Additional Packages

```bash
npm install lightweight-charts @radix-ui/react-select @radix-ui/react-separator @radix-ui/react-tabs @radix-ui/react-tooltip class-variance-authority clsx tailwind-merge lucide-react
```

### Step 3: Create Global Styles

Create `app/globals.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

@layer base {
  :root {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --border: 217.2 32.6% 17.5%;
  }

  * {
    @apply border-border;
  }

  body {
    @apply bg-background text-foreground font-mono;
    font-family: 'IBM Plex Mono', monospace;
  }
}
```

### Step 4: Create Root Layout

Create `app/layout.tsx`:

```typescript
import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Sniper Trading Framework',
  description: 'Real-time quantitative trading dashboard',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  )
}
```

### Step 5: Create Type Definitions

Create `types/market.ts`:

```typescript
export type RegimeType = 'trending_bull' | 'trending_bear' | 'ranging' | 'high_vol'
export type ActionType = 'LONG' | 'SHORT' | 'HOLD' | 'NOT_LOADED'
export type DirectionType = 'LONG' | 'SHORT'

export type Candle = {
  datetime: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  ema_fast: number
  ema_slow: number
  ema_trend: number
  rsi: number
  signal: number
  raw_signal: number
  regime: RegimeType
  ml_score: number
  shock_detected: boolean
  stop_loss?: number
  take_profit?: number
}

export type WSMessage =
  | { type: 'init'; data: { candles: Candle[]; symbol: string } }
  | { type: 'candle_update'; data: Candle }
  | { type: 'regime_change'; data: { from: RegimeType; to: RegimeType; timestamp: string } }
  | { type: 'signal_fired'; data: { direction: DirectionType; ml_score: number; entry_price: number; stop_loss: number; take_profit: number; regime: RegimeType } }
  | { type: 'shock_detected'; data: { vol_ratio: number; bars_cooldown: number; timestamp: string } }
  | { type: 'rl_decision'; data: { action: ActionType; confidence: number; overrode_signal: boolean } }
  | { type: 'alpha_warning'; data: { rolling_sharpe: number; baseline_sharpe: number; rolling_accuracy: number } }
  | { type: 'risk_update'; data: { drawdown_pct: number; current_capital: number; daily_pnl_pct: number; trading_halted: boolean } }
  | { type: 'error'; data: { message: string } }
```

### Step 6: Create API Client

Create `lib/api.ts`:

```typescript
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function fetchSymbols(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/api/symbols`)
  const data = await res.json()
  return data.symbols
}

export async function fetchMarketData(symbol: string, bars: number, timeframe: string) {
  const res = await fetch(`${BASE_URL}/api/data?symbol=${symbol}&bars=${bars}&timeframe=${timeframe}`)
  return res.json()
}
```

### Step 7: Create WebSocket Hook

Create `hooks/useMarketWebSocket.ts`:

```typescript
"use client"

import { useState, useEffect, useRef } from 'react'
import type { Candle, WSMessage, RegimeType, ActionType } from '@/types/market'

export function useMarketWebSocket(symbol: string, timeframe: string) {
  const [candles, setCandles] = useState<Candle[]>([])
  const [latestCandle, setLatestCandle] = useState<Candle | null>(null)
  const [regime, setRegime] = useState<RegimeType>('ranging')
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL}/ws/${symbol}/${timeframe}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      const message: WSMessage = JSON.parse(event.data)

      switch (message.type) {
        case 'init':
          setCandles(message.data.candles)
          break
        case 'candle_update':
          setLatestCandle(message.data)
          setCandles(prev => [...prev.slice(0, -1), message.data])
          break
        case 'regime_change':
          setRegime(message.data.to)
          break
      }
    }

    ws.onerror = () => setIsConnected(false)
    ws.onclose = () => setIsConnected(false)

    wsRef.current = ws

    return () => {
      ws.close()
    }
  }, [symbol, timeframe])

  return { candles, latestCandle, regime, isConnected }
}
```

### Step 8: Create Simple Dashboard

Create `app/page.tsx`:

```typescript
"use client"

import { useMarketWebSocket } from '@/hooks/useMarketWebSocket'

export default function Home() {
  const { candles, latestCandle, regime, isConnected } = useMarketWebSocket('NIFTY', '5m')

  return (
    <div className="min-h-screen p-8">
      <h1 className="text-3xl font-bold mb-4">Sniper Trading Dashboard</h1>

      <div className="mb-4">
        Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
      </div>

      <div className="mb-4">
        Regime: <span className="font-bold">{regime}</span>
      </div>

      <div className="mb-4">
        Candles loaded: {candles.length}
      </div>

      {latestCandle && (
        <div className="bg-gray-800 p-4 rounded">
          <h2 className="text-xl mb-2">Latest Candle</h2>
          <div>Close: ₹{latestCandle.close.toFixed(2)}</div>
          <div>RSI: {latestCandle.rsi.toFixed(2)}</div>
          <div>ML Score: {latestCandle.ml_score.toFixed(3)}</div>
        </div>
      )}
    </div>
  )
}
```

### Step 9: Run the Frontend

```bash
cd sniper-ui
npm run dev
```

Frontend will run at: http://localhost:3000

---

## 🚀 QUICK START (Both Servers)

Create `run_dev.sh` in project root:

```bash
#!/bin/bash
cd sniper_framework && python run_server.py &
cd sniper-ui && npm run dev &
wait
```

Make it executable and run:

```bash
chmod +x run_dev.sh
./run_dev.sh
```

---

## 📋 REMAINING WORK

The basic structure is ready. To complete the full dashboard as specified:

1. ✅ Backend - COMPLETE
2. ⚠️  Frontend - Basic structure created, needs:
   - CandlestickChart component with lightweight-charts
   - RSIPanel component
   - SignalAlert component with audio
   - RegimeBadge component with animations
   - RLStatus component
   - RiskMonitor component
   - AlphaMonitor component
   - ShockIndicator component
   - ConnectionStatus component
   - SymbolSelector component

All backend features are functional. Frontend components can be added incrementally.

---

## 🧪 TESTING

### Test Backend
```bash
# Health check
curl http://localhost:8000/api/health

# Get symbols
curl http://localhost:8000/api/symbols

# Get market data
curl "http://localhost:8000/api/data?symbol=NIFTY&bars=100&timeframe=5m"

# API docs
open http://localhost:8000/docs
```

### Test WebSocket
Use browser console:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/NIFTY/5m')
ws.onmessage = (e) => console.log(JSON.parse(e.data))
```

---

## 📦 DEPENDENCIES

### Backend (Python)
All dependencies in `requirements.txt`:
- fastapi
- uvicorn[standard]
- websockets
- python-multipart
- pydantic
- yfinance
- pandas, numpy, scipy
- scikit-learn, hmmlearn
- stable-baselines3, gymnasium, torch

### Frontend (Node.js)
All dependencies in `package.json`:
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- lightweight-charts
- Radix UI components

---

## 🎯 NEXT STEPS

1. Install frontend dependencies: `cd sniper-ui && npm install`
2. Start backend: `cd sniper_framework && python run_server.py`
3. Start frontend: `cd sniper-ui && npm run dev`
4. Open http://localhost:3000
5. Add remaining components incrementally

The system is fully functional with the basic setup. Advanced UI components can be added as needed!
