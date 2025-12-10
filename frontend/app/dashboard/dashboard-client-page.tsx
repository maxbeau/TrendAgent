'use client';

import { AxiosError } from 'axios';
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useRouter, useSearchParams } from 'next/navigation';
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
import { TrendScenario } from '@/components/aion/trend-scenario';
import { KeyVariableTable } from '@/components/aion/key-variables';
import { StrategyMatrix } from '@/components/aion/strategy-matrix';
import { AionRadar } from '@/components/charts/aion-radar';
import { StructureChart } from '@/components/charts/structure-chart';
import { FactorGrid } from '@/components/factor-grid';
import { MarketSnapshot } from '@/components/market-snapshot';
import { useAionEngine } from '@/hooks/use-aion-engine';
import { useLiveQuote } from '@/hooks/use-live-quote';
import { factorLabels } from '@/lib/factor-labels';
import { safeNumber } from '@/lib/numbers';
import { fetchFullReport } from '@/lib/requests/report';
import { cn } from '@/lib/utils';
import { describeIvHvDelta, formatExpectedMoveRange, ivHvBadgeLabel as buildIvHvBadgeLabel, pickExpectedMove, type RawExpectedMove } from '@/lib/volatility';
import type { AionAnalysisResult, FactorKey } from '@/types/aion';
import { RefreshCcw } from 'lucide-react';
import { useAionStore } from '@/store/aion-store';

const prettyKey = (key: string) => key.replace(/^.+?\./, '').replace(/_/g, ' ').toUpperCase();
const CACHE_TTL_MINUTES = 60;
const CACHE_TTL_MS = CACHE_TTL_MINUTES * 60 * 1000;
const isFresh = (analysis?: AionAnalysisResult) => {
  if (!analysis?.calculated_at) return false;
  const normalized = /([zZ]|[+-]\d\d:\d\d)$/.test(analysis.calculated_at)
    ? analysis.calculated_at
    : `${analysis.calculated_at}Z`;
  const ts = Date.parse(normalized);
  if (!Number.isFinite(ts)) return false;
  const ageMs = Date.now() - ts;
  return ageMs < CACHE_TTL_MINUTES * 60 * 1000;
};

type NormalizedEventComponent = {
  label?: string;
  description?: string;
  date?: string;
  formattedDate?: string | null;
  source?: string;
};

export default function DashboardClientPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const urlTicker = searchParams?.get('ticker')?.toUpperCase() ?? 'NVDA';
  const analysis = useAionStore((state) => state.analysis);
  const ohlc = useAionStore((state) => state.ohlc);
  const factorMeta = useAionStore((state) => state.factorMeta);
  const hydrate = useAionStore((state) => state.hydrate);
  const setAnalysis = useAionStore((state) => state.setAnalysis);
  const clear = useAionStore((state) => state.clear);
  const ticker = analysis?.ticker || urlTicker;
  const [tickerInput, setTickerInput] = useState(urlTicker);
  const [factorDialog, setFactorDialog] = useState<{
    key: FactorKey;
    summary?: string;
    key_evidence?: string[];
    sources?: Array<{ title?: string; url?: string; source?: string }>;
    components?: Record<string, unknown>;
  } | null>(null);
  const lastRefreshTime = useRef<number>(0);
  const autoStartRef = useRef<Record<string, number>>({});

  const {
    data: reportData,
    isLoading: reportLoading,
    isFetching: reportFetching,
    isError: reportError,
    error: reportErrorObj,
    refetch: refetchReport,
  } = useQuery({
    queryKey: ['full-report', urlTicker],
    queryFn: () => fetchFullReport(urlTicker),
    enabled: typeof window !== 'undefined',
    staleTime: 60_000,
  });
  useEffect(() => {
    if (reportData) hydrate(reportData);
  }, [reportData, hydrate]);
  const reportErrorMessage = reportError
    ? ((reportErrorObj as Error | undefined)?.message ?? '无法加载仪表盘报告')
    : null;
  const currentAnalysis = useMemo(
    () => (analysis?.ticker?.toUpperCase() === urlTicker ? analysis : undefined),
    [analysis, urlTicker],
  );
  const hasFreshCache = useMemo(() => isFresh(currentAnalysis), [currentAnalysis]);

  useEffect(() => {
    setTickerInput(urlTicker);
    clear();
    delete autoStartRef.current[urlTicker];
  }, [urlTicker, clear]);

  const handleTickerSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const normalized = tickerInput.trim().toUpperCase().replace(/\s+/g, '');
      if (!normalized || normalized === urlTicker) return;
      const params = new URLSearchParams(searchParams.toString());
      params.set('ticker', normalized);
      router.push(`/dashboard?${params.toString()}`);
    },
    [router, searchParams, tickerInput, urlTicker],
  );

  const { start, result, isCalculating, isSuccess } = useAionEngine();
  const reportStatus = (reportErrorObj as AxiosError | undefined)?.response?.status;
  useEffect(() => {
    const lastAutoStart = autoStartRef.current[urlTicker];
    const reportNotFound = reportStatus === 404;
    const missingOrStale = !hasFreshCache;
    const shouldStart = !reportLoading && !isCalculating && missingOrStale && (reportNotFound || !currentAnalysis);
    const cooldownPassed = !lastAutoStart || Date.now() - lastAutoStart > CACHE_TTL_MS;
    if (shouldStart && cooldownPassed) {
      autoStartRef.current[urlTicker] = Date.now();
      start({ ticker: urlTicker });
    }
  }, [currentAnalysis, hasFreshCache, isCalculating, reportLoading, reportStatus, start, urlTicker]);
  // 引擎结果同步 - 直接调用
  useEffect(() => {
    if (result) {
      setAnalysis(result);
    }
  }, [result, setAnalysis]);

  useEffect(() => {
    if (isSuccess) {
      refetchReport();
    }
  }, [isSuccess, refetchReport]);

  const liveQuote = useLiveQuote();
  const priceTone =
    liveQuote && liveQuote.change !== 0 ? (liveQuote.change > 0 ? 'text-bullish' : 'text-bearish') : 'text-slate-300';
  const pctLabel =
    liveQuote && Number.isFinite(liveQuote.pct) ? `${liveQuote.pct >= 0 ? '+' : ''}${liveQuote.pct.toFixed(2)}%` : '—';

  const formulaMap = useMemo(() => {
    const map: Partial<Record<FactorKey, string>> = {};
    factorMeta?.factors?.forEach((item) => {
      const key = item.factor_key as FactorKey;
      if (key) map[key] = item.formula_text || '';
    });
    return map;
  }, [factorMeta?.factors]);

  // factorLabels 现在是静态的，无需额外处理

  const modelLabel = analysis?.model_version ?? 'AION';
  const radarData = useMemo(() => {
    if (!analysis?.factors) return [];
    
    const entries = (Object.entries(analysis.factors) as [FactorKey, { score: number }][]).map(([key, factor]) => ({
      factor: factorLabels[key],
      score: factor.score ?? 0,
    }));
    
    return entries;
  }, [analysis?.factors]);
  const structureData = useMemo(
    () => ({
      candles: ohlc?.candles ?? [],
      ma20: ohlc?.ma20 ?? [],
      ma50: ohlc?.ma50 ?? [],
      ma200: ohlc?.ma200 ?? [],
      bands: ohlc?.bands ?? [],
    }),
    [ohlc],
  );

  const lastSyncedValue = useMemo(() => {
    const raw = analysis?.calculated_at;
    if (!raw) return '—';
    const normalizedRaw = /([zZ]|[+-]\d\d:\d\d)$/.test(raw) ? raw : `${raw}Z`;
    const dt = new Date(normalizedRaw);
    if (Number.isNaN(dt.getTime())) return '—';
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return dt.toLocaleString(undefined, {
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
  }, [analysis?.calculated_at]);

  const ivHvDelta = useMemo(() => {
    const components = analysis?.factors?.volatility?.components as { iv_vs_hv?: unknown } | undefined;
    return safeNumber(components?.iv_vs_hv);
  }, [analysis?.factors?.volatility?.components]);
  const expectedMove = useMemo(() => {
    const components = analysis?.factors?.volatility?.components as { expected_move?: RawExpectedMove } | undefined;
    return pickExpectedMove(components?.expected_move);
  }, [analysis?.factors?.volatility?.components]);
  const ivHvBadgeText = useMemo(() => buildIvHvBadgeLabel(ivHvDelta), [ivHvDelta]);
  const expectedRangeText = useMemo(() => formatExpectedMoveRange(expectedMove), [expectedMove]);

  const volumeZ = useMemo(() => {
    const components = analysis?.factors?.technical?.components as { volume_z?: unknown } | undefined;
    return safeNumber(components?.volume_z);
  }, [analysis?.factors?.technical?.components]);
  const putCall = useMemo(() => {
    const components = analysis?.factors?.flow?.components as { put_call?: { put_call_ratio?: unknown } } | undefined;
    return safeNumber(components?.put_call?.put_call_ratio);
  }, [analysis?.factors?.flow?.components]);
  const narrativeText = useMemo(() => {
    const catalyst = analysis?.factors?.catalyst?.summary;
    const industry = analysis?.factors?.industry?.summary;
    const candidate = catalyst || industry || '';
    if (!candidate || candidate.includes('等待模型结果')) return '等待模型生成行业/催化叙事';
    return candidate.replace(/^Score\s*[:]*\s*[0-9.]+\s*(?:·|:)?\s*/i, '').trim();
  }, [analysis?.factors?.catalyst?.summary, analysis?.factors?.industry?.summary]);
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

  const formulaLoading = reportLoading && !factorMeta;
  const ohlcLoading = reportLoading && !ohlc;
  const ohlcErrorMessage = reportErrorMessage;
  const isRefreshing = isCalculating || reportFetching;
  const handleRefresh = useCallback(() => {
    if (isCalculating) return;
    const now = Date.now();
    if (lastRefreshTime.current && now - lastRefreshTime.current < 1000) {
      return;
    }
    lastRefreshTime.current = now;
    refetchReport();
    start({ ticker: urlTicker });
  }, [isCalculating, refetchReport, start, urlTicker]);

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
      <nav className="sticky top-0 z-20 border-b border-white/5 bg-obsidian-950/80 backdrop-blur">
        <div className="page-shell flex flex-col gap-3 py-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="border-violet-400/30 bg-white/5 text-slate-100">
              TrendAgent
            </Badge>
            <span className="text-sm text-slate-400">多 ticker 快速切换</span>
          </div>
          <form
            onSubmit={handleTickerSubmit}
            className="flex w-full flex-col gap-2 md:w-auto md:flex-row md:items-center md:gap-3"
          >
            <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 md:min-w-[320px]">
              <span className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Ticker</span>
              <input
                value={tickerInput}
                onChange={(event) => setTickerInput(event.target.value)}
                className="w-full bg-transparent text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none"
                placeholder="如：NVDA / AAPL / TSLA"
                aria-label="输入要搜索的 ticker"
              />
            </div>
            <button
              type="submit"
              className="inline-flex items-center justify-center rounded-xl bg-violet-500 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-violet-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-violet-300"
            >
              搜索
            </button>
          </form>
        </div>
      </nav>
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
            {reportErrorMessage ? <p className="text-xs text-warning">{reportErrorMessage}</p> : null}
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
          <ActionCard isCalculating={isCalculating} isLoading={reportLoading && !analysis} />

          <Card>
            <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>AION Radar</CardTitle>
                <CardDescription>八维因子雷达图</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {reportLoading && !radarData.length ? (
                <Skeleton className="h-[320px] w-full rounded-xl" />
              ) : reportErrorMessage ? (
                <div className="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm text-amber-100">
                  {reportErrorMessage}
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
              {reportErrorMessage ? <p className="text-xs text-warning">{reportErrorMessage}</p> : null}
            </div>
            <Badge variant="outline">AION Factors</Badge>
          </div>
          <FactorGrid
            onSelect={(key, data) =>
              setFactorDialog({
                key,
                summary: data?.summary,
                key_evidence: data?.key_evidence,
                sources: data?.sources,
                components: data?.components,
              })
            }
            metaLoading={formulaLoading}
          />
        </section>

        <section className="grid gap-6">
          <TrendScenario />
          <KeyVariableTable />
        </section>

        <section className="grid gap-6">
          <StrategyMatrix />
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
              {ohlcLoading ? (
                <Skeleton className="h-[320px] w-full rounded-xl" />
              ) : ohlcErrorMessage ? (
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

        <section>
          <Card>
            <CardHeader>
              <CardTitle>市场快照</CardTitle>
              <CardDescription>基础行情 · 波动率 · 资金与成交 · 行业与叙事</CardDescription>
            </CardHeader>
            <CardContent>
              <MarketSnapshot ticker={ticker} />
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
