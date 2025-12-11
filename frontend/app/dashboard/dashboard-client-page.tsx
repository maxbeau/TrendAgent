'use client';

import { AxiosError } from 'axios';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter, useSearchParams } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { ActionCard } from '@/components/action-card';
import { TrendScenario } from '@/components/aion/trend-scenario';
import { KeyVariableTable } from '@/components/aion/key-variables';
import { StrategyMatrix } from '@/components/aion/strategy-matrix';
import { AionRadar } from '@/components/charts/aion-radar';
import { StructureChart } from '@/components/charts/structure-chart';
import { FactorGrid } from '@/components/factor-grid';
import { MarketSnapshot } from '@/components/market-snapshot';
import { DashboardHeader } from '@/components/dashboard/dashboard-header';
import { DashboardNav } from '@/components/dashboard/dashboard-nav';
import { FactorDetailsDialog } from '@/components/dashboard/factor-details-dialog';
import { Badge } from '@/components/ui/badge';
import { useAionEngine } from '@/hooks/use-aion-engine';
import { useLiveQuote } from '@/hooks/use-live-quote';
import { factorLabels } from '@/lib/factor-labels';
import { safeNumber } from '@/lib/numbers';
import { fetchFullReport, type FullReportResponse } from '@/lib/requests/report';
import { ivHvBadgeLabel as buildIvHvBadgeLabel, pickExpectedMove, type RawExpectedMove } from '@/lib/volatility';
import type { AionAnalysisResult, FactorKey } from '@/types/aion';
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
  const ticker = urlTicker;
  const [factorDialog, setFactorDialog] = useState<{
    key: FactorKey;
    summary?: string;
    key_evidence?: string[];
    sources?: Array<{ title?: string; url?: string; source?: string }>;
    components?: Record<string, unknown>;
  } | null>(null);
  const lastRefreshTime = useRef<number>(0);
  const autoStartRef = useRef<Record<string, number>>({});
  const queryClient = useQueryClient();
  const cachedReport = queryClient.getQueryData<FullReportResponse>(['full-report', urlTicker]);
  const [isMounted, setIsMounted] = useState(false);

  const currentAnalysis = useMemo(
    () => (analysis?.ticker?.toUpperCase() === urlTicker ? analysis : undefined),
    [analysis, urlTicker],
  );
  const cachedAnalysis = cachedReport?.analysis as unknown as AionAnalysisResult | undefined;
  const hasFreshCache = useMemo(
    () => isFresh(currentAnalysis) || isFresh(cachedAnalysis),
    [cachedAnalysis, currentAnalysis],
  );
  const shouldFetchReport = useMemo(
    () =>
      !hasFreshCache ||
      (!currentAnalysis && !cachedAnalysis) ||
      (!ohlc && !cachedReport?.ohlc) ||
      (!factorMeta && !cachedReport?.factor_meta),
    [cachedAnalysis, cachedReport?.factor_meta, cachedReport?.ohlc, currentAnalysis, factorMeta, hasFreshCache, ohlc],
  );
  useEffect(() => {
    setIsMounted(true);
  }, []);
  const queryEnabled = isMounted && shouldFetchReport;

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
    enabled: queryEnabled,
    staleTime: CACHE_TTL_MS,
    initialData: cachedReport,
    refetchOnWindowFocus: false,
  });
  useEffect(() => {
    if (reportData) hydrate(reportData);
  }, [reportData, hydrate]);
  const reportErrorMessage = reportError
    ? ((reportErrorObj as Error | undefined)?.message ?? '无法加载仪表盘报告')
    : null;

  useEffect(() => {
    clear();
    delete autoStartRef.current[urlTicker];
  }, [urlTicker, clear]);

  const handleTickerSubmit = useCallback(
    (nextTicker: string) => {
      const normalized = nextTicker.trim().toUpperCase().replace(/\s+/g, '');
      if (!normalized || normalized === urlTicker) return;
      const params = new URLSearchParams(searchParams.toString());
      params.set('ticker', normalized);
      router.push(`/dashboard?${params.toString()}`);
    },
    [router, searchParams, urlTicker],
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
  const isRefreshing = isCalculating || reportFetching;
  const displayAnalysis = isRefreshing ? undefined : currentAnalysis;
  const displayOhlc = isRefreshing ? undefined : ohlc;
  const displayFactorMeta = isRefreshing ? undefined : factorMeta;

  const liveQuote = useLiveQuote();
  const priceTone =
    liveQuote && liveQuote.change !== 0 ? (liveQuote.change > 0 ? 'text-bullish' : 'text-bearish') : 'text-slate-300';
  const pctLabel =
    liveQuote && Number.isFinite(liveQuote.pct) ? `${liveQuote.pct >= 0 ? '+' : ''}${liveQuote.pct.toFixed(2)}%` : '—';

  const formulaMap = useMemo(() => {
    const map: Partial<Record<FactorKey, string>> = {};
    displayFactorMeta?.factors?.forEach((item) => {
      const key = item.factor_key as FactorKey;
      if (key) map[key] = item.formula_text || '';
    });
    return map;
  }, [displayFactorMeta?.factors]);

  // factorLabels 现在是静态的，无需额外处理

  const modelLabel = displayAnalysis?.model_version ?? 'AION';
  const radarData = useMemo(() => {
    if (!displayAnalysis?.factors) return [];
    
    const entries = (Object.entries(displayAnalysis.factors) as [FactorKey, { score: number }][]).map(([key, factor]) => ({
      factor: factorLabels[key],
      score: factor.score ?? 0,
    }));
    
    return entries;
  }, [displayAnalysis?.factors]);
  const structureData = useMemo(
    () => ({
      candles: displayOhlc?.candles ?? [],
      ma20: displayOhlc?.ma20 ?? [],
      ma50: displayOhlc?.ma50 ?? [],
      ma200: displayOhlc?.ma200 ?? [],
      bands: displayOhlc?.bands ?? [],
    }),
    [displayOhlc],
  );

  const lastSyncedValue = useMemo(() => {
    if (!isMounted) return '—';
    const raw = displayAnalysis?.calculated_at;
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
  }, [displayAnalysis?.calculated_at, isMounted]);

  const ivHvDelta = useMemo(() => {
    const components = displayAnalysis?.factors?.volatility?.components as { iv_vs_hv?: unknown } | undefined;
    return safeNumber(components?.iv_vs_hv);
  }, [displayAnalysis?.factors?.volatility?.components]);
  const expectedMove = useMemo(() => {
    const components = displayAnalysis?.factors?.volatility?.components as { expected_move?: RawExpectedMove } | undefined;
    return pickExpectedMove(components?.expected_move);
  }, [displayAnalysis?.factors?.volatility?.components]);
  const ivHvBadgeText = useMemo(() => buildIvHvBadgeLabel(ivHvDelta), [ivHvDelta]);

  const formulaLoading = isRefreshing || (reportLoading && !displayFactorMeta);
  const ohlcLoading = isRefreshing || (reportLoading && !displayOhlc);
  const ohlcErrorMessage = reportErrorMessage;
  const radarLoading = isRefreshing || (reportLoading && !radarData.length);
  const handleRefresh = useCallback(() => {
    if (isRefreshing) return;
    const now = Date.now();
    if (lastRefreshTime.current && now - lastRefreshTime.current < 1000) {
      return;
    }
    lastRefreshTime.current = now;
    start({ ticker: urlTicker });
  }, [isRefreshing, start, urlTicker]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-obsidian-950 via-obsidian-950 to-obsidian-900">
      <DashboardNav modelLabel={modelLabel} initialTicker={ticker} onSubmit={handleTickerSubmit} />
      <div className="page-shell space-y-6 py-8">
        <DashboardHeader
          ticker={ticker}
          ivHvBadgeText={ivHvBadgeText}
          livePrice={liveQuote?.close}
          priceTone={priceTone}
          pctLabel={pctLabel}
          lastSyncedValue={lastSyncedValue}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          reportErrorMessage={reportErrorMessage}
        />

        <section>
          <MarketSnapshot
            ticker={ticker}
            factors={displayAnalysis?.factors}
            actionCard={displayAnalysis?.action_card}
            isLoading={isRefreshing}
          />
        </section>

        <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <ActionCard result={displayAnalysis} isCalculating={isRefreshing} isLoading={reportLoading && !displayAnalysis} />

          <Card>
            <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>AION Radar</CardTitle>
                <CardDescription>八维因子雷达图</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {radarLoading ? (
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
            result={displayAnalysis}
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
            isLoading={isRefreshing}
          />
        </section>

        <section className="grid gap-6">
          <TrendScenario scenarios={displayAnalysis?.scenarios} isLoading={isRefreshing} />
          <KeyVariableTable variables={displayAnalysis?.key_variables} isLoading={isRefreshing} />
        </section>

        <section className="grid gap-6">
          <StrategyMatrix
            stockStrategy={displayAnalysis?.stock_strategy}
            optionStrategies={displayAnalysis?.option_strategies}
            riskManagement={displayAnalysis?.risk_management}
            isLoading={isRefreshing}
          />
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

        <FactorDetailsDialog
          open={Boolean(factorDialog)}
          onOpenChange={(open) => !open && setFactorDialog(null)}
          factorKey={factorDialog?.key}
          factorLabel={factorDialog ? factorLabels[factorDialog.key] : ''}
          data={factorDialog}
          formulaText={factorDialog ? formulaMap[factorDialog.key] : ''}
        />
      </div>
    </div>
  );
}
