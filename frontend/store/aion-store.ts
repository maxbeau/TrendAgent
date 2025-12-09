import { create } from 'zustand';

import { normalizeAionResult } from '@/lib/aion-normalizer';
import type { FactorMetaResponse } from '@/lib/requests/factor-meta';
import type { OhlcResponse } from '@/lib/requests/market';
import type { FullReportResponse } from '@/lib/requests/report';
import type { AionAnalysisResult } from '@/types/aion';

const parseTimestamp = (value?: string) => {
  if (!value) return null;
  const normalized = /([zZ]|[+-]\d\d:\d\d)$/.test(value) ? value : `${value}Z`;
  const ts = Date.parse(normalized);
  return Number.isFinite(ts) ? ts : null;
};

interface AionStoreState {
  analysis?: AionAnalysisResult;
  ohlc?: OhlcResponse;
  factorMeta?: FactorMetaResponse;
  setAnalysis: (analysis?: AionAnalysisResult) => void;
  hydrate: (payload: FullReportResponse) => void;
  clear: () => void;
}

export const useAionStore = create<AionStoreState>((set) => ({
  analysis: undefined,
  ohlc: undefined,
  factorMeta: undefined,

  setAnalysis: (analysis) =>
    set((state) => {
      if (!analysis) {
        return state.analysis ? { ...state, analysis: undefined } : state;
      }
      const incomingTs = parseTimestamp(analysis.calculated_at);
      const currentTs = parseTimestamp(state.analysis?.calculated_at);
      if (incomingTs !== null && currentTs !== null && incomingTs <= currentTs) {
        return state;
      }
      return { ...state, analysis };
    }),

  hydrate: (payload) => {
    const normalizedAnalysis = normalizeAionResult(payload.analysis);
    const incomingTs = parseTimestamp(normalizedAnalysis.calculated_at);
    set((state) => {
      const currentTs = parseTimestamp(state.analysis?.calculated_at);
      if (incomingTs !== null && currentTs !== null && incomingTs <= currentTs) {
        return state;
      }
      return {
        analysis: normalizedAnalysis,
        ohlc: payload.ohlc,
        factorMeta: payload.factor_meta,
      };
    });
  },

  clear: () => set({ analysis: undefined, ohlc: undefined, factorMeta: undefined }),
}));
