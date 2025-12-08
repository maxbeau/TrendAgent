import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { AionAnalysisResult } from '@/types/aion';

type LiveQuote = { close: number; change: number; pct: number } | null;

interface MarketSnapshotProps {
  ticker: string;
  liveQuote: LiveQuote;
  ivHvDelta: number | null;
  factors?: AionAnalysisResult['factors'];
  actionCard?: string;
}

function toneClass(tone: 'bullish' | 'bearish' | 'neutral' | 'warning' | 'muted') {
  if (tone === 'bullish') return 'text-bullish';
  if (tone === 'bearish') return 'text-bearish';
  if (tone === 'warning') return 'text-warning';
  if (tone === 'neutral') return 'text-slate-200';
  return 'text-slate-400';
}

function formatPrice(liveQuote: LiveQuote) {
  if (!liveQuote) return '等待行情数据';
  const price = `$${liveQuote.close.toFixed(2)}`;
  const change = `${liveQuote.change >= 0 ? '+' : ''}${liveQuote.change.toFixed(2)}`;
  const pct = `${liveQuote.pct >= 0 ? '+' : ''}${liveQuote.pct.toFixed(2)}%`;
  return `${price} · 日内 ${change} (${pct})`;
}

function describeVol(ivHvDelta: number | null) {
  if (ivHvDelta === null) return { text: '等待波动率计算 (IV vs HV)', tone: 'muted' as const };
  if (ivHvDelta > 0.05) return { text: '隐含波动率高于历史波动率 · 期权偏贵', tone: 'warning' as const };
  if (ivHvDelta < -0.05) return { text: '隐含波动率低于历史波动率 · 期权偏便宜', tone: 'bullish' as const };
  return { text: '隐含波动率接近历史波动率 · 中性', tone: 'neutral' as const };
}

function safeNumber(val: unknown): number | null {
  if (typeof val === 'number' && Number.isFinite(val)) return val;
  if (typeof val === 'string') {
    const num = Number(val);
    return Number.isFinite(num) ? num : null;
  }
  return null;
}

function describeVolume(volumeZ: number | null) {
  if (volumeZ === null) return '成交量等待更新';
  if (volumeZ >= 1.5) return '成交量维持高位';
  if (volumeZ >= 0.5) return '成交量略高于均值';
  if (volumeZ <= -0.8) return '成交量显著低于均值';
  return '成交量接近均值';
}

function describePcr(pcr: number | null) {
  if (pcr === null) return '期权市场情绪等待更新';
  if (pcr > 1.2) return 'Put/Call 偏高 · 防守情绪上升';
  if (pcr < 0.8) return 'Put/Call 偏低 · 看涨情绪占优';
  return 'Put/Call 中性区间';
}

function SnapshotItem({
  title,
  badge,
  emoji,
  lines,
  tone = 'neutral',
}: {
  title: string;
  badge: string;
  emoji: string;
  lines: string[];
  tone?: 'bullish' | 'bearish' | 'neutral' | 'warning' | 'muted';
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{title}</p>
          <p className={cn('text-sm font-medium leading-relaxed', toneClass(tone))}>
            {emoji} {lines[0]}
          </p>
        </div>
        <Badge variant="outline">{badge}</Badge>
      </div>
      {lines.slice(1).map((line) => (
        <p key={line} className="mt-1 text-xs text-slate-400">
          {line}
        </p>
      ))}
    </div>
  );
}

export function MarketSnapshot({ ticker, liveQuote, ivHvDelta, factors, actionCard }: MarketSnapshotProps) {
  const priceLine = formatPrice(liveQuote);

  const volComponents = factors?.volatility?.components as { iv_vs_hv?: unknown } | undefined;
  const volDelta = safeNumber(ivHvDelta ?? volComponents?.iv_vs_hv);
  const expectedMove = (volComponents as { expected_move?: { iv?: Record<string, unknown>; hv?: Record<string, unknown> } } | undefined)?.expected_move;
  const volDesc = describeVol(volDelta);
  const volRangeLine = (() => {
    const pick = expectedMove?.iv ?? expectedMove?.hv;
    if (!pick) return null;
    const lower = safeNumber((pick as any).lower);
    const upper = safeNumber((pick as any).upper);
    if (lower === null || upper === null) return null;
    const days = (pick as any).days ?? 30;
    const basis = expectedMove?.iv ? 'IV' : 'HV';
    return `${days}日 1σ 区间 $${lower.toFixed(2)} - $${upper.toFixed(2)}（基于${basis}）`;
  })();

  const technicalComponents = factors?.technical?.components as { volume_z?: unknown } | undefined;
  const volumeZ = safeNumber(technicalComponents?.volume_z);

  const flowComponents = factors?.flow?.components as { put_call?: { put_call_ratio?: unknown }; institutional_count?: unknown } | undefined;
  const pcr = safeNumber(flowComponents?.put_call?.put_call_ratio);
  const instCount = safeNumber(flowComponents?.institutional_count);

  const flowLinePrimary = `${describeVolume(volumeZ)} · ${describePcr(pcr)}`;
  const flowLineSecondary = instCount !== null ? `机构持仓记录数：${instCount}` : '机构持仓等待更新';

  const industrySummary = factors?.industry?.summary;
  const catalystSummary = factors?.catalyst?.summary;
  const narrativePrimary =
    industrySummary || catalystSummary
      ? [industrySummary, catalystSummary].filter(Boolean).join(' / ')
      : '等待模型生成行业与催化叙事';
  const narrativeSecondary = actionCard ? `当前决策卡片：${actionCard}` : '运行 AION 引擎后展示决策卡片';

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>市场快照</CardTitle>
          <CardDescription>基础行情 · 波动率 · 资金与成交 · 行业与叙事</CardDescription>
        </div>
        <Badge variant="outline">Ticker · {ticker}</Badge>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-4">
          <SnapshotItem title="基础行情" badge="价格" emoji="📌" lines={[priceLine]} />
        <SnapshotItem
          title="波动率"
          badge="IV vs HV 差值"
          emoji="🔄"
          lines={[volDesc.text, volRangeLine ?? '基于 AION Volatility 因子 (IV-HV)']}
          tone={volDesc.tone}
        />
          <SnapshotItem
            title="资金与成交"
            badge="成交与期权情绪"
            emoji="🔍"
            lines={[flowLinePrimary, flowLineSecondary]}
            tone="neutral"
          />
          <SnapshotItem
            title="行业与叙事"
            badge="行业 & 催化摘要"
            emoji="🧭"
            lines={[narrativePrimary, narrativeSecondary]}
            tone="neutral"
          />
        </div>
      </CardContent>
    </Card>
  );
}
