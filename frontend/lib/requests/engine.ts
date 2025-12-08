import { apiClient } from '../api-client';
import type { TaskResponse } from '@/types/aion';

export interface CalculatePayload {
  ticker: string;
  model_version?: string;
}

export async function triggerAionCalculation(payload: CalculatePayload) {
  const { data } = await apiClient.post<TaskResponse>('/engine/calculate', payload);
  return data;
}

export async function fetchAionStatus(taskId: string) {
  const { data } = await apiClient.get<TaskResponse>(`/engine/status/${taskId}`);
  return data;
}
