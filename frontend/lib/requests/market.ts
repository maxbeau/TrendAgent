import { apiClient } from '../api-client';

export interface OhlcPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface OverlayPoint {
  time: string;
  value: number;
}

export interface BandPoint {
  time: string;
  upper: number;
  lower: number;
}

export interface OhlcResponse {
  ticker: string;
  candles: OhlcPoint[];
  ma20: OverlayPoint[];
  ma50: OverlayPoint[];
  ma200: OverlayPoint[];
  bands: BandPoint[];
  source?: string;
}

export async function fetchOhlc(ticker: string, limit = 200) {
  const { data } = await apiClient.get<OhlcResponse>('/market/ohlc', { params: { ticker, limit } });
  return data;
}
