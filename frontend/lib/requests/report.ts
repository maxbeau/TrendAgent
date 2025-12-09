import { apiClient } from '../api-client';
import type { EngineScoreApi } from '@/types/aion';
import type { FactorMetaResponse } from './factor-meta';
import type { OhlcResponse } from './market';

export interface FullReportResponse {
  analysis: EngineScoreApi;
  ohlc: OhlcResponse;
  factor_meta: FactorMetaResponse;
}

export async function fetchFullReport(ticker: string, limit = 200) {
  const { data } = await apiClient.get<FullReportResponse>(`/report/v2/${ticker}`, {
    params: { limit },
  });
  return data;
}
