'use client';

import { useMutation, useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';

import type { AionAnalysisResult, TaskResponse } from '@/types/aion';
import { fetchAionStatus, triggerAionCalculation, type CalculatePayload } from '@/lib/requests/engine';
import { normalizeAionResult } from '@/lib/aion-normalizer';

interface UseAionEngineState {
  start: (input: CalculatePayload) => void;
  taskId: string | null;
  status?: TaskResponse['status'];
  result?: AionAnalysisResult;
  isIdle: boolean;
  isCalculating: boolean;
  isSuccess: boolean;
  isError: boolean;
  isLoading: boolean;
  error?: string;
}

export function useAionEngine(): UseAionEngineState {
  const [taskId, setTaskId] = useState<string | null>(null);

  const calculateMutation = useMutation({
    mutationFn: (input: CalculatePayload) => triggerAionCalculation(input),
    onSuccess: (data) => setTaskId(data.task_id),
    onError: () => setTaskId(null),
  });

  const statusQuery = useQuery({
    queryKey: ['engine-status', taskId],
    queryFn: () => fetchAionStatus(taskId ?? ''),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status) return 1000;
      return status === 'pending' || status === 'processing' ? 1000 : false;
    },
    refetchOnWindowFocus: false,
  });

  const status = statusQuery.data?.status;
  const raw = statusQuery.data?.data;
  const result = raw ? normalizeAionResult(raw) : undefined;

  const isSuccess = status === 'completed';
  const isError = status === 'failed' || calculateMutation.isError || statusQuery.isError;
  const isCalculating = Boolean(taskId) && !isSuccess && !isError;
  const isIdle = !taskId && !calculateMutation.isPending;

  const errorMessage = useMemo(() => {
    const mutationError = calculateMutation.error as Error | undefined;
    const queryError = statusQuery.error as Error | undefined;
    return mutationError?.message ?? queryError?.message;
  }, [calculateMutation.error, statusQuery.error]);

  const start = (input: CalculatePayload) => calculateMutation.mutate(input);

  return {
    start,
    taskId,
    status,
    result,
    isIdle,
    isCalculating,
    isSuccess,
    isError,
    isLoading: calculateMutation.isPending || statusQuery.isLoading,
    error: errorMessage,
  };
}
