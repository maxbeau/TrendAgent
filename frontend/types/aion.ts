export type FactorKey =
  | 'macro'
  | 'industry'
  | 'fundamental'
  | 'technical'
  | 'flow'
  | 'sentiment'
  | 'catalyst'
  | 'volatility';

export interface FactorDetail {
  score: number | null;
  status: 'live' | 'delayed' | 'static';
  summary: string;
  key_evidence?: string[];
  components?: Record<string, unknown>;
  sources?: Array<{ title?: string; url?: string; source?: string }>;
}

// 后端 /engine/status 返回的原始字段
export interface EngineScoreApi {
  id: string;
  task_id: string;
  ticker: string;
  total_score: number;
  model_version: string;
  action_card: string;
  factors: Record<
    string,
    {
      score: number | null;
      status?: 'live' | 'delayed' | 'static';
      weight?: number;
      summary?: string;
      key_evidence?: string[];
      sources?: Array<{ title?: string; url?: string; source?: string }>;
      components?: Record<string, unknown>;
      errors?: string[];
    }
  >;
  weight_denominator?: number;
  calculated_at?: string;
}

export interface AionAnalysisResult {
  ticker: string;
  calculated_at?: string;
  model_version: string;
  total_score: number; // 1.00 - 5.00
  signal: 'STRONG_BUY' | 'BUY' | 'WAIT' | 'SELL' | 'SHORT';
  action_card: string;
  action_plan?: {
    target_price: number;
    stop_loss: number;
    suggested_strategy: string;
    risk_sizing: 'Standard' | 'Half' | 'Quarter';
  };
  factors: Partial<Record<FactorKey, FactorDetail>>;
}

export interface TaskResponse {
  task_id?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number; // 0-100
  data?: EngineScoreApi;
  error?: string;
}
