import { useMemo } from 'react';

import { useAionStore } from '@/store/aion-store';

export interface LiveQuote {
  close: number;
  change: number;
  pct: number;
  dateLabel?: string;
}

export function useLiveQuote(): LiveQuote | null {
  const candles = useAionStore((state) => state.ohlc?.candles);

  return useMemo(() => {
    if (!candles?.length) return null;
    const last = candles[candles.length - 1];
    const prev = candles.length > 1 ? candles[candles.length - 2] : null;
    const close = Number(last.close);
    const prevClose = prev ? Number(prev.close) : null;
    if (!Number.isFinite(close)) return null;
    const change = prevClose ? close - prevClose : 0;
    const pct = prevClose ? (change / prevClose) * 100 : 0;
    return {
      close,
      change,
      pct,
      dateLabel: typeof last.time === 'string' ? last.time : undefined,
    };
  }, [candles]);
}
