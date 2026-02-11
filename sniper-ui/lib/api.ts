import type { Candle, RLStatus, RiskSummary } from '@/types/market'

const BASE = process.env.NEXT_PUBLIC_API_URL

function getBaseUrl(): string {
  if (!BASE) {
    throw new Error('NEXT_PUBLIC_API_URL is not configured')
  }
  return BASE
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`)
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}) for ${path}`)
  }
  return (await res.json()) as T
}

export async function fetchSymbols(): Promise<string[]> {
  const data = await getJson<{ symbols: string[] }>('/api/symbols')
  return data.symbols
}

export async function fetchData(symbol: string, bars: number, timeframe: string): Promise<Candle[]> {
  const data = await getJson<{ candles: Candle[] }>(`/api/data?symbol=${symbol}&bars=${bars}&timeframe=${timeframe}`)
  return data.candles
}

export async function fetchRiskSummary(): Promise<RiskSummary> {
  return getJson<RiskSummary>('/api/risk-summary')
}

export async function fetchRLStatus(): Promise<RLStatus> {
  return getJson<RLStatus>('/api/rl-status')
}

export async function fetchBacktest(symbol: string, bars: number): Promise<Record<string, unknown>> {
  return getJson<Record<string, unknown>>(`/api/backtest?symbol=${symbol}&bars=${bars}`)
}

export async function fetchCosts(entry: number, exit: number, lots: number, direction: number): Promise<Record<string, unknown>> {
  return getJson<Record<string, unknown>>(`/api/costs?entry=${entry}&exit=${exit}&lots=${lots}&direction=${direction}`)
}
