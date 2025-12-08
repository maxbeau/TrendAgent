import { safeNumber } from './numbers';

export type VolTone = 'bullish' | 'bearish' | 'neutral' | 'warning' | 'muted';

export type RawExpectedMove = {
  iv?: Record<string, unknown> | null;
  hv?: Record<string, unknown> | null;
};

export type ExpectedMoveRange = {
  lower: number | null;
  upper: number | null;
  spot: number | null;
  days: number;
  basis: 'IV' | 'HV';
  movePct: number | null;
  moveAbs: number | null;
};

const IV_HV_THRESHOLD = 0.05;

export function describeIvHvDelta(delta: number | null): { text: string; tone: VolTone } {
  if (delta === null) return { text: '等待波动率计算 (IV vs HV)', tone: 'muted' };
  if (delta > IV_HV_THRESHOLD) return { text: '隐含波动率高于历史波动率 · 期权偏贵', tone: 'warning' };
  if (delta < -IV_HV_THRESHOLD) return { text: '隐含波动率低于历史波动率 · 期权偏便宜', tone: 'bullish' };
  return { text: '隐含波动率接近历史波动率 · 中性', tone: 'neutral' };
}

export function ivHvBadgeLabel(delta: number | null): string {
  if (delta === null) return 'IV vs HV · —';
  const pct = delta * 100;
  const formatted = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
  return `IV vs HV · ${formatted}`;
}

export function pickExpectedMove(raw?: RawExpectedMove | null): ExpectedMoveRange | null {
  if (!raw) return null;
  const basis: 'IV' | 'HV' | null = raw.iv ? 'IV' : raw.hv ? 'HV' : null;
  if (!basis) return null;
  const source = (basis === 'IV' ? raw.iv : raw.hv) ?? null;
  if (!source || typeof source !== 'object') return null;
  const lower = safeNumber((source as any).lower);
  const upper = safeNumber((source as any).upper);
  const spot = safeNumber((source as any).spot);
  const dayValue = safeNumber((source as any).days);
  const days = dayValue !== null ? Math.max(1, Math.round(dayValue)) : 30;
  const movePct = safeNumber((source as any).move_pct);
  const moveAbs = safeNumber((source as any).move_abs);
  return {
    lower,
    upper,
    spot,
    days,
    basis,
    movePct,
    moveAbs,
  };
}

export function formatExpectedMoveRange(range: ExpectedMoveRange | null): string | null {
  if (!range || range.lower === null || range.upper === null) return null;
  return `${range.days}日 1σ 区间 $${range.lower.toFixed(2)} - $${range.upper.toFixed(2)}（基于${range.basis}）`;
}

