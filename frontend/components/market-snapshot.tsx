import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { safeNumber } from '@/lib/numbers';
import { describeIvHvDelta, formatExpectedMoveRange, pickExpectedMove, type RawExpectedMove } from '@/lib/volatility';
import { cn } from '@/lib/utils';
import { useLiveQuote } from '@/hooks/use-live-quote';
import { useAionStore } from '@/store/aion-store';
import type { AionAnalysisResult } from '@/types/aion';

type LiveQuote = { close: number; change: number; pct: number; dateLabel?: string } | null;

interface MarketSnapshotProps {
  ticker?: string;
  liveQuote?: LiveQuote;
  ivHvDelta?: number | null;
  factors?: AionAnalysisResult['factors'];
  actionCard?: string;
  isLoading?: boolean;
}

type InstitutionalTrend = {
  source?: string;
  timeline?: Array<{
    period?: string;
    holder_count?: number;
    total_value?: number | null;
    total_shares?: number | null;
  }>;
  latest_holder_count?: number;
  latest_period?: string | null;
  previous_period?: string | null;
  qoq_change_value?: number | null;
  qoq_change_shares?: number | null;
  trend_metric?: number | null;
};

type InstitutionalSources = {
  fmp?: boolean;
  yfinance?: boolean;
};

type FlowComponents = {
  put_call?: { put_call_ratio?: unknown };
  institutional_count?: unknown;
  institutional_trend?: InstitutionalTrend;
  institutional_sources?: InstitutionalSources;
};

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

function formatSignedPercent(value: number | null) {
  if (value === null) return '—';
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
}

function formatCompactUsd(value: number | null) {
  if (value === null) return '—';
  const abs = Math.abs(value);
  const units =
    abs >= 1e9
      ? { divisor: 1e9, suffix: 'B' }
      : abs >= 1e6
        ? { divisor: 1e6, suffix: 'M' }
        : abs >= 1e3
          ? { divisor: 1e3, suffix: 'K' }
          : { divisor: 1, suffix: '' };
  const precision = units.divisor === 1 ? 0 : 1;
  return `$${(value / units.divisor).toFixed(precision)}${units.suffix}`;
}

function describeInstitutionalTrendSummary(trend?: InstitutionalTrend | null) {
  const change =
    safeNumber(trend?.qoq_change_value) ??
    safeNumber(trend?.qoq_change_shares) ??
    safeNumber(trend?.trend_metric);
  if (change === null) {
    return { text: '机构增减趋势等待更新', tone: 'muted' as const };
  }
  let tone: 'bullish' | 'bearish' | 'neutral' | 'warning' | 'muted' = 'neutral';
  if (change >= 0.1) tone = 'bullish';
  else if (change <= -0.1) tone = 'bearish';
  const direction = change >= 0 ? '机构持仓回升' : '机构持仓下降';
  return {
    text: `${direction} ${formatSignedPercent(change)}`,
    tone,
  };
}

function describeInstitutionalSource(trend?: InstitutionalTrend | null, sources?: InstitutionalSources) {
  if (!trend) return '机构趋势等待更新';
  if (sources?.fmp) return '首选数据源：FMP';
  if (sources?.yfinance) return '首选数据源：yfinance（FMP 数据暂缺）';
  return '尚未连接可用的机构数据源';
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

type SnapshotRichLine = {
  label: string;
  value: string;
  highlight?: boolean;
};

function SnapshotItem({
  title,
  badge,
  emoji,
  lines = [],
  richLines = [],
  tone = 'neutral',
}: {
  title: string;
  badge: string;
  emoji: string;
  lines?: string[];
  richLines?: SnapshotRichLine[];
  tone?: 'bullish' | 'bearish' | 'neutral' | 'warning' | 'muted';
}) {
  const primaryText = lines?.[0] ?? richLines?.[0]?.value ?? '等待更新';
  const extraLines = (lines ?? []).slice(1);

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{title}</p>
          <p className={cn('text-sm font-medium leading-relaxed', toneClass(tone))}>
            {emoji} {primaryText}
          </p>
        </div>
        <Badge variant="outline">{badge}</Badge>
      </div>
      {extraLines.map((line) => (
        <p key={line} className="mt-1 text-xs text-slate-400">
          {line}
        </p>
      ))}
      {richLines?.length ? (
        <div className="mt-3 space-y-1">
          {richLines.map((line) => (
            <div key={`${line.label}-${line.value}`} className="flex items-center justify-between text-xs">
              <span className="uppercase tracking-[0.18em] text-slate-500">{line.label}</span>
              <span
                className={cn(
                  'text-right text-slate-300',
                  line.highlight ? 'font-medium text-slate-100' : '',
                )}
              >
                {line.value}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function InstitutionalTrendTimeline({ trend }: { trend?: InstitutionalTrend | null }) {
  const timeline = Array.isArray(trend?.timeline) ? trend.timeline.slice(0, 4) : [];
  if (!timeline.length) return null;
  const sourceLabel = trend?.source ? trend.source.toUpperCase() : '多源';
  return (
    <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
        <p className="font-medium text-slate-200">机构持仓趋势</p>
        <span>数据源 · {sourceLabel}</span>
      </div>
      <div className="mt-3 grid grid-cols-3 text-xs uppercase tracking-[0.18em] text-slate-500">
        <span>季度</span>
        <span>机构数</span>
        <span>持仓规模</span>
      </div>
      <div className="mt-2 space-y-1">
        {timeline.map((entry, idx) => {
          const holders =
            typeof entry.holder_count === 'number'
              ? entry.holder_count.toLocaleString('en-US')
              : '—';
          const totalValue = formatCompactUsd(safeNumber(entry.total_value));
          return (
            <div
              key={entry.period ?? idx}
              className="grid grid-cols-3 rounded-lg bg-white/5 px-2 py-1 text-sm font-mono text-slate-200"
            >
              <span>{entry.period ?? '—'}</span>
              <span>{holders}</span>
              <span>{totalValue}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function MarketSnapshot({
  ticker,
  liveQuote,
  ivHvDelta,
  factors,
  actionCard,
  isLoading,
}: MarketSnapshotProps = {}) {
  const analysis = useAionStore((state) => state.analysis);
  const derivedLiveQuote = useLiveQuote();
  const loading = Boolean(isLoading);
  const resolvedTicker = ticker ?? 'NVDA';
  const resolvedFactors = loading ? undefined : factors ?? analysis?.factors;
  const resolvedActionCard = loading ? undefined : actionCard ?? analysis?.action_card;
  const volComponents = resolvedFactors?.volatility?.components as { iv_vs_hv?: unknown; expected_move?: RawExpectedMove } | undefined;
  const storeIvHvDelta = safeNumber(volComponents?.iv_vs_hv);
  const volDelta = safeNumber((ivHvDelta ?? storeIvHvDelta) ?? null);
  const expectedMove = pickExpectedMove(volComponents?.expected_move);
  const volDesc = describeIvHvDelta(volDelta);
  const volRangeLine = formatExpectedMoveRange(expectedMove);

  const technicalComponents = resolvedFactors?.technical?.components as { volume_z?: unknown } | undefined;
  const volumeZ = safeNumber(technicalComponents?.volume_z);

  const flowComponents = resolvedFactors?.flow?.components as FlowComponents | undefined;
  const pcr = safeNumber(flowComponents?.put_call?.put_call_ratio);
  const institutionalTrend = flowComponents?.institutional_trend;
  const instCount = safeNumber(flowComponents?.institutional_count ?? institutionalTrend?.latest_holder_count);
  const trendSummary = describeInstitutionalTrendSummary(institutionalTrend);
  const sourceSummary = describeInstitutionalSource(institutionalTrend, flowComponents?.institutional_sources);

  const flowLinePrimary = `${describeVolume(volumeZ)} · ${describePcr(pcr)}`;
  const flowLineSecondary = instCount !== null ? `机构持仓记录数：${instCount}` : '机构持仓等待更新';
  const flowRichLines: SnapshotRichLine[] = [
    { label: '机构持仓', value: flowLineSecondary },
    { label: '机构行为', value: trendSummary.text, highlight: trendSummary.tone === 'bullish' || trendSummary.tone === 'bearish' },
    { label: '来源', value: sourceSummary },
  ];

  const industrySummary = resolvedFactors?.industry?.summary;
  const catalystSummary = resolvedFactors?.catalyst?.summary;
  const narrativePrimary =
    industrySummary || catalystSummary
      ? [industrySummary, catalystSummary].filter(Boolean).join(' · ')
      : '等待模型生成行业与催化叙事';

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>市场快照</CardTitle>
          <CardDescription>基础行情 · 波动率 · 资金与成交 · 行业与叙事</CardDescription>
        </div>
        <Badge variant="outline">Ticker · {resolvedTicker}</Badge>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="grid gap-3 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, idx) => (
              <Card key={idx} className="border-white/10 bg-white/5 p-4">
                <div className="flex items-center justify-between">
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-5 w-12" />
                </div>
                <Skeleton className="mt-3 h-4 w-1/2" />
                <Skeleton className="mt-2 h-3 w-full" />
                <Skeleton className="mt-2 h-3 w-5/6" />
              </Card>
            ))}
            <div className="col-span-full mt-2">
              <Skeleton className="h-16 w-full rounded-xl" />
            </div>
          </div>
        ) : (
          <>
            <div className="grid gap-3 lg:grid-cols-3">
              <SnapshotItem title="基础行情" badge="价格" emoji="📌" lines={[formatPrice(liveQuote ?? derivedLiveQuote)]} />
              <SnapshotItem
                title="波动率"
                badge="IV vs HV 差值"
                emoji="🔄"
                lines={[volDesc.text]}
                richLines={[
                  {
                    label: '波动区间',
                    value: volRangeLine ?? '等待波动区间数据',
                    highlight: Boolean(volRangeLine),
                  },
                ]}
                tone={volDesc.tone}
              />
              <SnapshotItem
                title="资金与成交"
                badge="成交与期权情绪"
                emoji="🔍"
                lines={[flowLinePrimary]}
                richLines={flowRichLines}
                tone={trendSummary.tone}
              />
            </div>

            <div className="mt-3 rounded-xl border-l-2 border-violet-400 bg-white/5 p-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-slate-500">
                <span>🧭 行业与叙事</span>
                {resolvedActionCard ? <Badge variant="outline" className="text-[10px] border-violet-400/30 text-violet-200">{resolvedActionCard}</Badge> : null}
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-200">
                {narrativePrimary}
              </p>
            </div>

            <InstitutionalTrendTimeline trend={institutionalTrend} />
          </>
        )}
      </CardContent>
    </Card>
  );
}
