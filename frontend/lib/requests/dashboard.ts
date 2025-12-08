import { apiClient } from '../api-client';
import type { EngineScoreApi } from '@/types/aion';

export interface DashboardSummaryItem extends EngineScoreApi {
  created_at?: string;
}

export interface DashboardSummaryResponse {
  latest_scores: DashboardSummaryItem[];
}

export async function fetchDashboardSummary() {
  const response = await apiClient.get<DashboardSummaryResponse>('/dashboard/summary');
  return response.data;
}
