import { apiClient } from '../api-client';

export interface FactorMetaItem {
  factor_key: string;
  formula_text?: string;
  description?: string;
  source?: string;
  updated_at?: string | null;
}

export interface FactorMetaResponse {
  factors: FactorMetaItem[];
}

export async function fetchFactorMeta() {
  const { data } = await apiClient.get<FactorMetaResponse>('/meta/factors');
  return data;
}
