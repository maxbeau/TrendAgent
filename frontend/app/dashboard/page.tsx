'use client';
import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'next/navigation';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { ActionCard } from '@/components/action-card';
import { AionRadar } from '@/components/charts/aion-radar';
import { StructureChart } from '@/components/charts/structure-chart';
import { FactorGrid } from '@/components/factor-grid';
import { useAionEngine } from '@/hooks/use-aion-engine';
import { normalizeAionResult } from '@/lib/aion-normalizer';
import { factorLabels } from '@/lib/factor-labels';
import { safeNumber } from '@/lib/numbers';
import { fetchDashboardSummary } from '@/lib/requests/dashboard';
import { cn } from '@/lib/utils';
import { fetchOhlc } from '@/lib/requests/market';
import { fetchFactorMeta } from '@/lib/requests/factor-meta';
import { describeIvHvDelta, formatExpectedMoveRange, ivHvBadgeLabel as buildIvHvBadgeLabel, pickExpectedMove, type RawExpectedMove } from '@/lib/volatility';
import type { FactorKey } from '@/types/aion';
import { RefreshCcw } from 'lucide-react';

const prettyKey = (key: string) => key.replace(/^.+?\./, '').replace(/_/g, ' ').toUpperCase();

type NormalizedEventComponent = {
  label?: string;
  description?: string;
  date?: string;
  formattedDate?: string | null;
  source?: string;
};

export default function DashboardPage() {
  const searchParams = useSearchParams();
  const [factorDialog, setFactorDialog] = useState<{
    key: FactorKey;
    summary?: string;
    key_evidence?: string[];
    sources?: Array<{ title?: string; url?: string; source?: string }>;
    components?: Record<string, unknown>;
  } | null>(null);
  const ticker = searchParams?.get('ticker')?.toUpperCase() ?? 'NVDA';
  const { start, result, isCalculating } = useAionEngine();
  const {
    data: summaryData,
    isLoading: summaryLoading,
    isError: summaryError,
    error: summaryErrorObj,
    isFetching: summaryFetching,
    refetch: refetchSummary,
  } = useQuery({
    queryKey: ['dashboard-summary'],
    queryFn: fetchDashboardSummary,
    staleTime: 60_000,
  });
  const {
    data: factorMeta,
    isLoading: metaLoading,
    isError: metaError,
    error: metaErrorObj,
  } = useQuery({
    queryKey: ['factor-meta'],
    queryFn: fetchFactorMeta,
    staleTime: 300_000,
  });
  const {
    data: ohlcData,
    isLoading: ohlcLoading,
    isError: ohlcError,
    error: ohlcErrorObj,
    isFetching: ohlcFetching,
    refetch: refetchOhlc,
  } = useQuery({
    queryKey: ['ohlc', ticker],
    queryFn: () => fetchOhlc(ticker),
    staleTime: 60_000,
    retry: 1,
  });
  const summaryResult = useMemo(() => {
    const scores = summaryData?.latest_scores ?? [];
    const matched = scores.find((entry) => entry.ticker?.toUpperCase() === ticker);
    return matched ? normalizeAionResult(matched) : undefined;
  }, [summaryData?.latest_scores, ticker]);
  const displayResult = result ?? summaryResult;
  const formulaMap = useMemo(() => {
    const map: Partial<Record<FactorKey, string>> = {};
    factorMeta?.factors?.forEach((item) => {
      const key = item.factor_key as FactorKey;
      if (key) map[key] = item.formula_text || '';
    });
    return map;
  }, [factorMeta?.factors]);
  const modelLabel = displayResult?.model_version ?? 'AION';
  const radarData = useMemo(
    () =>
      displayResult?.factors
        ? (Object.entries(displayResult.factors) as [FactorKey, { score: number }][]).map(([key, factor]) => ({
            factor: factorLabels[key],
            score: factor.score ?? 0,
          }))
        : [],
    [displayResult?.factors],
  );
  const structureData = useMemo(
    () => ({
      candles: ohlcData?.candles ?? [],
      ma20: ohlcData?.ma20 ?? [],
      ma50: ohlcData?.ma50 ?? [],
      ma200: ohlcData?.ma200 ?? [],
      bands: ohlcData?.bands ?? [],
    }),
    [ohlcData],
  );
  const ohlcErrorMessage = ohlcError ? (ohlcErrorObj as Error | undefined)?.message ?? '无法加载行情数据' : null;
  const summaryErrorMessage = summaryError
    ? (summaryErrorObj as Error | undefined)?.message ?? '无法加载最新评分数据'
    : null;
  const metaErrorMessage = metaError ? (metaErrorObj as Error | undefined)?.message ?? '无法加载因子元数据' : null;
  const lastSyncedValue = useMemo(() => {
    const raw = displayResult?.calculated_at;
    if (!raw) return '—';
    const normalizedRaw = /([zZ]|[+-]\d\d:\d\d)$/.test(raw) ? raw : `${raw}Z`;
    const dt = new Date(normalizedRaw);
    if (Number.isNaN(dt.getTime())) return '—';
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    // Format timestamp in the viewer's timezone but keep origin normalized to UTC
    const formatted = dt.toLocaleString(undefined, {
      timeZone: tz,
      timeZoneName: 'short',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
    return formatted;
  }, [displayResult?.calculated_at]);
  const isRefreshing = isCalculating || summaryFetching || ohlcFetching;
  const handleRefresh = () => {
    if (isCalculating) return;
    refetchSummary();
    refetchOhlc();
    start({ ticker });
  };

  const renderSummary = (text?: string) => {
    if (!text) return '暂无摘要，可重新运行引擎获取最新结果。';
    const parts = text
      .split(/[;；]/)
      .map((p) => p.trim())
      .filter(Boolean);
    if (!parts.length) return text;
    return (
      <div className="space-y-1">
        {parts.map((part) => (
          <p key={part}>{part}</p>
        ))}
      </div>
    );
  };
  const liveQuote = useMemo(() => {
    const candles = ohlcData?.candles;
    if (!candles?.length) return null;
    const last = candles[candles.length - 1];
    const prev = candles.length > 1 ? candles[candles.length - 2] : null;
    const close = Number(last.close);
    const prevClose = prev ? Number(prev.close) : null;
    if (!Number.isFinite(close)) return null;
    const change = prevClose ? close - prevClose : 0;
    const pct = prevClose ? (change / prevClose) * 100 : 0;
    const dateLabel =
      typeof last.time === 'string'
        ? last.time
        : typeof last.time === 'number'
          ? new Date(last.time * 1000).toISOString().split('T')[0]
          : '';
    return { close, change, pct, dateLabel };
  }, [ohlcData?.candles]);
  const priceTone =
    liveQuote && liveQuote.change !== 0 ? (liveQuote.change > 0 ? 'text-bullish' : 'text-bearish') : 'text-slate-300';
  const pctLabel =
    liveQuote && Number.isFinite(liveQuote.pct) ? `${liveQuote.pct >= 0 ? '+' : ''}${liveQuote.pct.toFixed(2)}%` : '—';
  const ivHvDelta = useMemo(() => {
    const components = displayResult?.factors?.volatility?.components as { iv_vs_hv?: unknown } | undefined;
    return safeNumber(components?.iv_vs_hv);
  }, [displayResult?.factors?.volatility?.components]);
  const expectedMove = useMemo(() => {
    const components = displayResult?.factors?.volatility?.components as { expected_move?: RawExpectedMove } | undefined;
    return pickExpectedMove(components?.expected_move);
  }, [displayResult?.factors?.volatility?.components]);
  const ivHvBadgeText = useMemo(() => buildIvHvBadgeLabel(ivHvDelta), [ivHvDelta]);
  const expectedRangeText = useMemo(() => formatExpectedMoveRange(expectedMove), [expectedMove]);
  const volumeZ = useMemo(() => {
    const components = displayResult?.factors?.technical?.components as { volume_z?: unknown } | undefined;
    return safeNumber(components?.volume_z);
  }, [displayResult?.factors?.technical?.components]);
  const putCall = useMemo(() => {
    const components = displayResult?.factors?.flow?.components as { put_call?: { put_call_ratio?: unknown } } | undefined;
    return safeNumber(components?.put_call?.put_call_ratio);
  }, [displayResult?.factors?.flow?.components]);
  const narrativeText = useMemo(() => {
    const catalyst = displayResult?.factors?.catalyst?.summary;
    const industry = displayResult?.factors?.industry?.summary;
    const candidate = catalyst || industry || '';
    if (!candidate || candidate.includes('等待模型结果')) return '等待模型生成行业/催化叙事';
    return candidate.replace(/^Score\s*[:]*\s*[0-9.]+\s*(?:·|:)?\s*/i, '').trim();
  }, [displayResult?.factors?.catalyst?.summary, displayResult?.factors?.industry?.summary]);
  const volSnapshot = useMemo(() => {
    const desc = describeIvHvDelta(ivHvDelta);
    const base = `Vol: ${desc.text}`;
    return expectedRangeText ? `${base} · ${expectedRangeText}` : base;
  }, [ivHvDelta, expectedRangeText]);
  const flowSnapshot = useMemo(() => {
    const volumePart =
      volumeZ === null ? '成交量等待更新' : volumeZ >= 1.5 ? '成交量高位' : volumeZ >= 0.5 ? '成交量略高' : volumeZ <= -0.8 ? '成交量偏低' : '成交量中性';
    const pcrPart =
      putCall === null ? 'Put/Call 等待更新' : putCall > 1.2 ? 'Put/Call 偏高（防守）' : putCall < 0.8 ? 'Put/Call 偏低（看涨）' : 'Put/Call 中性';
    return `Flow: ${volumePart} · ${pcrPart}`;
  }, [volumeZ, putCall]);
  const factorComponents = factorDialog?.components as Record<string, any> | undefined;
  const isCatalystDialog = factorDialog?.key === 'catalyst';
  const toNum = (val: unknown) => {
    const num = Number(val);
    return Number.isFinite(num) ? num : null;
  };
  const formatEventDate = (value?: string) => {
    if (!value) return null;
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(date);
  };
  const normalizeEventComponent = (raw: any): NormalizedEventComponent | null => {
    if (!raw || typeof raw !== 'object') return null;
    const label = typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : undefined;
    const description = typeof raw.description === 'string' && raw.description.trim() ? raw.description.trim() : undefined;
    const date = typeof raw.date === 'string' && raw.date.trim() ? raw.date.trim() : undefined;
    const source = typeof raw.source === 'string' && raw.source.trim() ? raw.source.trim() : undefined;
    const formattedDate = formatEventDate(date);
    if (!label && !description && !date) return null;
    return { label, description, date, formattedDate, source };
  };
  const weightContext = useMemo(() => {
    const weights = factorComponents?.weights_used;
    if (!weights || typeof weights !== 'object') return null;
    const map: Record<string, number> = {};
    Object.entries(weights).forEach(([k, v]) => {
      const num = toNum(v);
      if (num !== null) map[k] = num;
    });
    if (!Object.keys(map).length) return null;
    return { map, denom: toNum(factorComponents?.weight_denominator) };
  }, [factorComponents]);
  const weightSummary = useMemo(() => {
    if (!weightContext) return null;
    const denomNum = weightContext.denom;
    const text = Object.entries(weightContext.map)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k, base]) => {
        const pct = denomNum && denomNum > 0 ? (base / denomNum) * 100 : base * 100;
        return `${prettyKey(k)} ${pct.toFixed(0)}%`;
      })
      .join(' · ');
    const denomText = denomNum !== null ? `（权重基数 ${denomNum.toFixed(1)}）` : '';
    return `${text} ${denomText}`.trim();
  }, [weightContext]);
  const factorScoreSummary = useMemo(() => {
    const factorScores = factorComponents?.factor_scores;
    if (!factorScores || typeof factorScores !== 'object') return null;
    const entries = Object.entries(factorScores)
      .filter(([, v]) => typeof v === 'number')
      .map(([k, v]) => ({ key: k, score: Number(v) }));
    return entries.length ? entries : null;
  }, [factorComponents]);

  const factorScoreList = useMemo(() => {
    if (!factorScoreSummary) return [];
    const weights = weightContext?.map || {};
    const denomNum = weightContext?.denom ?? null;
    return factorScoreSummary.map((item) => {
      const weightNum = weights[item.key];
      const weight = Number.isFinite(weightNum)
        ? denomNum !== null && denomNum > 0
          ? (weightNum / denomNum) * 100
          : weightNum * 100
        : null;
      return {
        label: prettyKey(item.key),
        score: item.score,
        weight,
      };
    });
  }, [factorScoreSummary, weightContext]);
  const selectedEvent = useMemo(() => normalizeEventComponent(factorComponents?.selected_event), [factorComponents]);
  const eventCandidates = useMemo(() => {
    const raw = factorComponents?.event_candidates;
    if (!Array.isArray(raw)) return [];
    return raw
      .map((item) => normalizeEventComponent(item))
      .filter((item): item is NormalizedEventComponent => Boolean(item));
  }, [factorComponents]);
  const eventSourceCount = useMemo(() => {
    const val = toNum(factorComponents?.event_source_count);
    return val !== null ? Math.max(0, Math.round(val)) : null;
  }, [factorComponents]);
  const componentDetails = useMemo(() => {
    if (!factorComponents) return [];
    const exclude = new Set([
      'weights_used',
      'factor_scores',
      'weight_denominator',
      'reasons',
      'citations',
      'asof_date',
      'confidence',
      'selected_event',
      'event_candidates',
      'event_source_count',
    ]);
    const formatValue = (key: string, val: any) => {
      if (val === null || val === undefined) return null;
      if (typeof val === 'number') {
        const num = Number.isFinite(val) ? val : null;
        if (num === null) return null;
        const lower = key.toLowerCase();
        if (lower.includes('pct') || lower.includes('ratio')) return `${(num * 100).toFixed(2)}%`;
        if (lower.includes('vol') || lower.includes('iv') || lower.includes('hv')) return `${num.toFixed(4)}`;
        return num.toFixed(2);
      }
      if (typeof val === 'string') return val;
      if (typeof val === 'object' && val !== null) {
        // 优先对常见结构化字段做扁平化
        if ('lower' in val && 'upper' in val) {
          const l = Number((val as any).lower);
          const u = Number((val as any).upper);
          if (Number.isFinite(l) && Number.isFinite(u)) return `${l.toFixed(2)} ~ ${u.toFixed(2)}`;
        }
        if ('put_call_ratio' in val) {
          const pcr = Number((val as any).put_call_ratio);
          return Number.isFinite(pcr) ? `PCR ${pcr.toFixed(2)}` : null;
        }
        if ('spot' in val && 'expiration' in val) {
          const spot = Number((val as any).spot);
          const exp = (val as any).expiration;
          const spotText = Number.isFinite(spot) ? `$${spot.toFixed(2)}` : '—';
          return `Spot ${spotText} · Exp ${exp ?? '—'}`;
        }
      }
      try {
        return JSON.stringify(val);
      } catch {
        return String(val);
      }
    };
    return Object.entries(factorComponents)
      .filter(([key]) => !exclude.has(key))
      .map(([key, val]) => ({ key: prettyKey(key), value: formatValue(key, val) }))
      .filter((item) => item.value !== null && item.value !== undefined && item.value !== '');
  }, [factorComponents]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-obsidian-950 via-obsidian-950 to-obsidian-900">
      <div className="page-shell space-y-6 py-8">
        <section className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-3">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-500">TrendAgent · {modelLabel}</p>
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex items-baseline gap-3">
                <h1 className="text-4xl font-semibold tracking-tight">{ticker}</h1>
              </div>
              <Badge variant="warning">{ivHvBadgeText}</Badge>
            </div>
            <p className="text-sm text-slate-400">
              Live Price{' '}
              <span className={cn('font-mono', priceTone)}>
                {liveQuote ? `$${liveQuote.close.toFixed(2)}` : '—'}
              </span>{' '}
              · Daily Δ <span className={cn('font-mono', priceTone)}>{pctLabel}</span>
            </p>
            <p className="text-sm text-slate-500">
              {volSnapshot} · {flowSnapshot} · {narrativeText}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    'h-2 w-2 rounded-full',
                    isRefreshing ? 'bg-amber-300 animate-pulse' : 'bg-emerald-400',
                  )}
                />
                <div className="leading-tight">
                  <p className="text-[10px] uppercase tracking-[0.26em] text-slate-500">Last Synced</p>
                  <p className="font-mono text-sm text-slate-100">{lastSyncedValue}</p>
                </div>
              </div>
              <button
                onClick={handleRefresh}
                disabled={isCalculating}
                className="group inline-flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-100 transition hover:border-violet-400/40 hover:text-violet-50 disabled:cursor-not-allowed disabled:opacity-60"
                aria-label="Refresh latest data"
                title="刷新最新数据"
              >
                <RefreshCcw className={cn('h-4 w-4', isRefreshing ? 'animate-spin' : 'group-hover:rotate-180')} />
              </button>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <ActionCard
            result={displayResult}
            fallbackTicker={ticker}
            isCalculating={isCalculating}
            isLoading={(summaryLoading || summaryFetching) && !displayResult}
          />

          <Card>
            <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>AION Radar</CardTitle>
                <CardDescription>八维因子雷达图</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {summaryLoading || summaryFetching ? (
                <Skeleton className="h-[320px] w-full rounded-xl" />
              ) : summaryErrorMessage ? (
                <div className="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm text-amber-100">
                  {summaryErrorMessage}
                </div>
              ) : radarData.length ? (
                <AionRadar data={radarData} />
              ) : (
                <p className="text-sm text-slate-300">暂无雷达数据，请运行 AION 或检查后台同步。</p>
              )}
            </CardContent>
          </Card>
        </section>

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">因子证据网格</h2>
              <p className="text-sm text-slate-400">八大因子状态灯 + 摘要</p>
              {metaErrorMessage ? <p className="text-xs text-warning">{metaErrorMessage}</p> : null}
            </div>
            <Badge variant="outline">AION Factors</Badge>
          </div>
          <FactorGrid
            result={displayResult}
            onSelect={(key, data) =>
              setFactorDialog({
                key,
                summary: data?.summary,
                key_evidence: data?.key_evidence,
                sources: data?.sources,
                components: data?.components,
              })
            }
            formulas={formulaMap}
            metaLoading={metaLoading}
          />
        </section>

        <section className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>智能研报</CardTitle>
              <CardDescription>Scenario Analysis · Bull / Base / Bear</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-slate-200">
              <div className="rounded-lg border border-white/10 bg-white/5 p-4 text-slate-300">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">状态</p>
                <p className="mt-2">尚未接入智能研报数据源，等待后端接口对接后展示 Bull / Base / Bear 场景。</p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 p-4 text-slate-300">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">提示</p>
                <p className="mt-2">可点击右上角运行 AION 引擎，生成基础因子评分；研报内容将在后续版本补充。</p>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6">
          <Card>
            <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>结构化 K 线</CardTitle>
                <CardDescription>蜡烛 + 均线 + 波动率带</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {ohlcLoading || ohlcFetching ? (
                <Skeleton className="h-[320px] w-full rounded-xl" />
              ) : ohlcError ? (
                <div className="rounded-lg border border-bearish/30 bg-bearish/10 p-3 text-sm text-bearish">
                  {ohlcErrorMessage}
                </div>
              ) : (
                <StructureChart
                  candles={structureData.candles}
                  ma20={structureData.ma20}
                  ma50={structureData.ma50}
                  ma200={structureData.ma200}
                  bands={structureData.bands}
                />
              )}
            </CardContent>
          </Card>
        </section>

        <Dialog open={Boolean(factorDialog)} onOpenChange={(open) => !open && setFactorDialog(null)}>
          <DialogContent className="flex max-h-[85vh] flex-col overflow-hidden">
            <DialogHeader>
              <DialogTitle>
                因子详情 · {factorDialog ? factorLabels[factorDialog.key] : ''}
              </DialogTitle>
              <DialogDescription>AION 因子输出</DialogDescription>
            </DialogHeader>
            <div className="mt-4 flex-1 overflow-y-auto pr-2 min-h-0">
              <div className="space-y-3 text-sm text-slate-200">
                {renderSummary(factorDialog?.summary)}
                <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">关键证据</p>
                  {factorDialog?.key_evidence?.length ? (
                    <ul className="list-disc space-y-1 pl-4 text-xs text-slate-100">
                      {factorDialog.key_evidence.map((item) => (
                        <li key={item} className="leading-snug">
                          {item}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-slate-400">暂无证据摘要，可重新运行引擎。</p>
                  )}
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">算法 & 权重</p>
                  <p className="text-sm text-slate-100">
                    {factorDialog ? formulaMap[factorDialog.key] ?? '算法待配置' : '—'}
                  </p>
                  <p className="text-xs text-slate-300">
                    当前权重：{weightSummary ?? '暂无权重数据'}
                  </p>
                  {factorScoreList.length ? (
                    <div className="space-y-1 text-xs text-slate-300">
                      {factorScoreList.map((item) => (
                        <div key={item.label} className="flex items-center justify-between">
                          <span>{item.label}</span>
                          <span className="font-mono text-slate-100">
                            {item.score.toFixed(1)}{item.weight !== null ? ` · ${item.weight.toFixed(0)}%` : ''}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-400">暂无子因子得分，可重新运行引擎。</p>
                  )}
                </div>
                {isCatalystDialog ? (
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">近期催化事件</p>
                    {selectedEvent ? (
                      <div className="rounded border border-white/10 bg-black/20 p-2 text-xs text-slate-200">
                        <p className="text-sm font-medium text-slate-100">{selectedEvent.label ?? '事件'}</p>
                        {selectedEvent.description ? (
                          <p className="text-xs text-slate-300">{selectedEvent.description}</p>
                        ) : null}
                        <p className="text-[11px] text-slate-500">
                          {selectedEvent.formattedDate ?? selectedEvent.date ?? '日期待定'}
                          {selectedEvent.source ? ` · 来源 ${selectedEvent.source}` : ''}
                        </p>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-400">暂无结构化事件，可刷新或等待更多新闻。</p>
                    )}
                    {eventCandidates.length ? (
                      <div className="space-y-1 text-xs text-slate-300">
                        {eventCandidates.map((candidate, idx) => (
                          <div key={`${candidate.label ?? 'event'}-${candidate.date ?? idx}`} className="flex flex-col rounded bg-black/10 p-2">
                            <span className="text-[11px] uppercase tracking-wide text-slate-500">
                              {candidate.label ?? 'EVENT'}
                            </span>
                            {candidate.description ? <span className="text-slate-200">{candidate.description}</span> : null}
                            <span className="text-[11px] text-slate-500">
                              {candidate.formattedDate ?? candidate.date ?? '日期待定'}
                              {candidate.source ? ` · ${candidate.source}` : ''}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    {eventSourceCount ? (
                      <p className="text-[11px] text-slate-500">事件来源映射 {eventSourceCount} 条</p>
                    ) : null}
                  </div>
                ) : null}
                <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">原始数据</p>
                  {componentDetails.length ? (
                    <div className="space-y-1 text-xs text-slate-300">
                      {componentDetails.map((item) => (
                        <div key={item.key} className="flex items-start justify-between gap-3">
                          <span className="min-w-[120px] text-slate-400">{item.key}</span>
                          <span className="font-mono text-slate-100 break-all text-right">{item.value}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-400">暂无结构化数据。</p>
                  )}
                </div>
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">来源</p>
                  <div className="max-h-32 overflow-y-auto space-y-2 pr-1">
                    {factorDialog?.sources?.length ? (
                      factorDialog.sources.map((src) => (
                        <div key={src.url ?? src.title} className="rounded-lg border border-white/10 bg-white/5 p-2">
                          <p className="text-slate-100">{src.title}</p>
                          <p className="text-xs text-slate-400">{src.source}</p>
                          {src.url ? (
                            <a className="text-xs text-violet-200" href={src.url} target="_blank" rel="noreferrer">
                              打开原文
                            </a>
                          ) : null}
                        </div>
                      ))
                    ) : (
                      <p className="text-xs text-slate-400">暂无引用新闻源。</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
