import type { AionAnalysisResult, EngineScoreApi, FactorKey } from '@/types/aion';

const factorKeyAliases: Record<FactorKey, string[]> = {
  macro: ['macro', 'macro_economy', 'f1'],
  industry: ['industry', 'sector', 'f2'],
  fundamental: ['fundamental', 'valuation', 'quality', 'f3'],
  technical: ['technical', 'price_action', 'momentum', 'f4'],
  flow: ['flow', 'flow_liquidity', 'positioning', 'f5'],
  sentiment: ['sentiment', 'emotion', 'crowd', 'f6'],
  catalyst: ['catalyst', 'event', 'trigger', 'f7'],
  volatility: ['volatility', 'risk', 'derivatives', 'f8'],
};

const factorCodeMap: Record<string, FactorKey> = {
  F1: 'macro',
  F2: 'industry',
  F3: 'fundamental',
  F4: 'technical',
  F5: 'flow',
  F6: 'sentiment',
  F7: 'catalyst',
  F8: 'volatility',
};

const allFactors: FactorKey[] = [
  'macro',
  'industry',
  'fundamental',
  'technical',
  'flow',
  'sentiment',
  'catalyst',
  'volatility',
];

function matchFactorKey(rawKey: string): FactorKey | undefined {
  const direct = factorCodeMap[rawKey.toUpperCase()];
  if (direct) return direct;
  const lowered = rawKey.toLowerCase();
  return (Object.keys(factorKeyAliases) as FactorKey[]).find((key) =>
    factorKeyAliases[key].some((alias) => lowered.includes(alias)),
  );
}

export function actionCardToSignal(action: string, score: number): AionAnalysisResult['signal'] {
  const normalized = (action ?? '').toLowerCase();
  if (normalized.includes('strong buy')) return 'STRONG_BUY';
  if (normalized.includes('buy')) return 'BUY';
  if (normalized.includes('wait') || normalized.includes('hold')) return 'WAIT';
  if (normalized.includes('reduce') || normalized.includes('sell')) return 'SELL';
  if (normalized.includes('short') || normalized.includes('avoid')) return 'SHORT';
  if (score >= 4.5) return 'STRONG_BUY';
  if (score >= 4.0) return 'BUY';
  if (score >= 3.0) return 'WAIT';
  if (score >= 2.0) return 'SELL';
  return 'SHORT';
}

function normalizeFactorStatus(raw?: string): 'live' | 'delayed' | 'static' {
  const normalized = (raw ?? '').toLowerCase();
  if (normalized === 'ok' || normalized === 'live') return 'live';
  if (normalized === 'degraded' || normalized === 'delayed') return 'delayed';
  return 'static';
}

function summarizeFactor(score: number | null, errors?: string[]) {
  const scoreText = typeof score === 'number' ? score.toFixed(2) : '—';
  const firstError = errors?.[0];
  return firstError ? `Score ${scoreText} · ${firstError}` : `Score ${scoreText}`;
}

function normalizeKeyEvidence(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter((item) => item.trim().length > 0);
  }
  if (typeof value === 'string' && value.trim()) return [value.trim()];
  return [];
}

function normalizeSources(value: unknown): Array<{ title?: string; url?: string; source?: string }> {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (item && typeof item === 'object') {
        return {
          title: (item as any).title ?? (item as any).headline,
          url: (item as any).url ?? (item as any).article_url,
          source: (item as any).source ?? (item as any).published_utc ?? (item as any).published_at,
        };
      }
      if (typeof item === 'string') return { title: item };
      return null;
    })
    .filter(Boolean) as Array<{ title?: string; url?: string; source?: string }>;
}

export function normalizeAionResult(data: EngineScoreApi): AionAnalysisResult {
  const factors: AionAnalysisResult['factors'] = {};

  // 先填充所有因子默认值，避免前端展示为空。
  allFactors.forEach((key) => {
    factors[key] = { score: null, status: 'static', summary: '等待模型结果...', key_evidence: [] };
  });

  Object.entries(data.factors || {}).forEach(([rawKey, value]) => {
    const key = matchFactorKey(rawKey);
    if (!key) return;
    const preferredSummary =
      typeof value.summary === 'string' && value.summary.trim().length ? value.summary : summarizeFactor(value.score, value.errors);
    factors[key] = {
      score: value.score,
      status: normalizeFactorStatus(value.status),
      summary: preferredSummary,
      key_evidence: normalizeKeyEvidence(value.key_evidence),
      components: value.components,
      sources: normalizeSources(value.sources),
    };
  });

  return {
    ticker: data.ticker,
    calculated_at: data.calculated_at,
    model_version: data.model_version,
    total_score: data.total_score,
    signal: actionCardToSignal(data.action_card ?? '', data.total_score),
    action_card: data.action_card,
    factors,
  };
}
