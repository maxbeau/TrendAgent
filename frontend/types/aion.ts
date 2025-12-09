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

// --- 趋势路径与情景分析 ---
export interface TrendScenario {
  type: 'base_case' | 'bull_case' | 'bear_case' | 'alt_case';
  label: string;
  probability: number;
  direction: 'bullish' | 'bearish' | 'neutral' | 'volatile';
  description: string;
  support: number[];
  resistance: number[];
  timeframe_notes: string;
  catalysts?: string[];
}

// --- 关键变量触发机制 ---
export interface KeyVariable {
  name: string;
  threshold: string;
  impact: 'bullish' | 'bearish' | 'neutral';
  suggestion: string;
}

// --- 策略矩阵 (正股 + 期权) ---
export interface StrategyLeg {
  action: 'buy' | 'sell';
  type: 'call' | 'put';
  strike: string;
  expiration?: string;
}

export interface OptionStrategy {
  name: string;
  description?: string;
  legs: StrategyLeg[];
  expiration_notes?: string;
  rationale?: string;
}

export interface StockStrategy {
  entry_zone: string;
  add_conditions: string[];
  reduce_conditions: string[];
  profit_target: string;
}

// --- 扩展因子详情 (支持权重) ---
export interface FactorDetailExtended extends FactorDetail {
  weight?: number;
  weighted_score?: number;
}

// --- 风控与赔率 ---
export interface RiskManagement {
  initial_position: string;
  max_exposure: string;
  add_rule: string;
  stop_loss_rule: string;
  odds_rating?: string;
  win_rate_rr?: string;
}

export interface ExecutionNotes {
  observation_cycle: string[];
  signals_to_watch: string[];
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
  scenarios?: TrendScenario[];
  key_variables?: KeyVariable[];
  stock_strategy?: StockStrategy;
  option_strategies?: OptionStrategy[];
  risk_management?: RiskManagement;
  execution_notes?: ExecutionNotes;
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
  factors: Partial<Record<FactorKey, FactorDetailExtended>>;
  scenarios?: TrendScenario[];
  key_variables?: KeyVariable[];
  stock_strategy?: StockStrategy;
  option_strategies?: OptionStrategy[];
  risk_management?: RiskManagement;
  execution_notes?: ExecutionNotes;
}

export interface TaskResponse {
  task_id?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number; // 0-100
  data?: EngineScoreApi;
  error?: string;
}
