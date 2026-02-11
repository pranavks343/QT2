const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function fetchSymbols(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/api/symbols`)
  const data = await res.json()
  return data.symbols
}

export async function fetchMarketData(symbol: string, bars: number, timeframe: string) {
  const res = await fetch(`${BASE_URL}/api/data?symbol=${symbol}&bars=${bars}&timeframe=${timeframe}`)
  if (!res.ok) throw new Error('Failed to fetch market data')
  return res.json()
}

export async function fetchRiskSummary() {
  const res = await fetch(`${BASE_URL}/api/risk-summary`)
  if (!res.ok) throw new Error('Failed to fetch risk summary')
  return res.json()
}

export async function fetchRLStatus() {
  const res = await fetch(`${BASE_URL}/api/rl-status`)
  if (!res.ok) throw new Error('Failed to fetch RL status')
  return res.json()
}

export async function fetchBacktest(symbol: string, bars: number) {
  const res = await fetch(`${BASE_URL}/api/backtest?symbol=${symbol}&bars=${bars}`)
  if (!res.ok) throw new Error('Failed to fetch backtest')
  return res.json()
}

export async function fetchCosts(entry: number, exit: number, lots: number, direction: number) {
  const res = await fetch(
    `${BASE_URL}/api/costs?entry=${entry}&exit=${exit}&lots=${lots}&direction=${direction}`
  )
  if (!res.ok) throw new Error('Failed to fetch costs')
  return res.json()
}
