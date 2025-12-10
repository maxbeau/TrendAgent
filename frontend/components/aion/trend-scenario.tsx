import { useEffect, useMemo, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import { useAionStore } from '@/store/aion-store';
import type { TrendScenario as TrendScenarioType } from '@/types/aion';

const directionLabels: Record<TrendScenarioType['direction'], { label: string; className: string }> = {
  bullish: { label: '看多', className: 'text-bullish border-bullish/40 bg-bullish/10' },
  bearish: { label: '看空', className: 'text-bearish border-bearish/40 bg-bearish/10' },
  neutral: { label: '中性', className: 'text-slate-200 border-white/20 bg-white/5' },
  volatile: { label: '高波动', className: 'text-warning border-warning/40 bg-warning/10' },
};

const scenarioPriority: Record<TrendScenarioType['type'], number> = {
  base_case: 0,
  bull_case: 1,
  bear_case: 2,
  alt_case: 3,
};

const formatLevel = (value: number) => {
  if (!Number.isFinite(value)) return '—';
  return `$${value % 1 === 0 ? value.toFixed(0) : value.toFixed(2)}`;
};

const formatLevels = (levels: number[]) => {
  if (!levels.length) return '—';
  return levels.map((level) => formatLevel(level)).join(' / ');
};

const toPct = (value?: number) => {
  if (value === undefined || !Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, Math.round(value * 100)));
};

export function TrendScenario({ scenarios, isLoading }: { scenarios?: TrendScenarioType[]; isLoading?: boolean } = {}) {
  const storeScenarios = useAionStore((state) => state.analysis?.scenarios);
  const loading = Boolean(isLoading);
  const ordered = useMemo(
    () =>
      (loading ? [] : scenarios ?? storeScenarios ?? [])
        .slice()
        .sort((a, b) => scenarioPriority[a.type] - scenarioPriority[b.type]),
    [loading, scenarios, storeScenarios],
  );
  const baseCase = useMemo(
    () => ordered.find((item) => item.type === 'base_case') ?? ordered[0],
    [ordered],
  );
  const altCase = useMemo(
    () => ordered.find((item) => item.type !== baseCase?.type),
    [ordered, baseCase?.type],
  );
  const [activeType, setActiveType] = useState<TrendScenarioType['type'] | null>(baseCase?.type ?? null);
  useEffect(() => {
    const next = baseCase?.type ?? ordered[0]?.type ?? null;
    setActiveType(next);
  }, [baseCase?.type, ordered]);

  const activeScenario =
    ordered.find((item) => item.type === activeType) ??
    baseCase ??
    ordered[0] ??
    null;
  const basePct = toPct(baseCase?.probability);
  const altPct = toPct(altCase?.probability);
  const totalForSplit = Math.max(basePct + altPct, 1);
  const baseWidth = (basePct / totalForSplit) * 100;
  const altWidth = (altPct / totalForSplit) * 100;

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>趋势路径 · 情景分析</CardTitle>
          <CardDescription>Base / Bull / Bear / Alt Case · 支撑阻力 · 驱动催化</CardDescription>
        </div>
        <Badge variant="outline">Trend Scenarios</Badge>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-24" />
              </div>
              <Skeleton className="mt-3 h-3 w-full" />
              <div className="mt-3 flex items-center gap-2">
                <Skeleton className="h-7 w-24" />
                <Skeleton className="h-7 w-24" />
              </div>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              {Array.from({ length: 2 }).map((_, idx) => (
                <div key={idx} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <Skeleton className="h-4 w-28" />
                  <Skeleton className="mt-2 h-6 w-16" />
                  <Skeleton className="mt-3 h-2 w-full" />
                  <Skeleton className="mt-3 h-10 w-full" />
                </div>
              ))}
            </div>
          </div>
        ) : ordered.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-300">
            暂无情景分析数据，等待后端接口接入。当前展示为占位信息。
          </div>
        ) : (
          <div className="space-y-4">
            <div className="rounded-2xl border border-white/10 bg-gradient-to-r from-violet-500/10 via-white/0 to-white/0 p-4 shadow-sm">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Base vs Alt</p>
                  <p className="text-sm text-slate-200">先看多空概率，再展开细节</p>
                </div>
                <div className="text-right text-xs text-slate-400">
                  <p>Base Case: {basePct || '—'}%</p>
                  <p>Alt Case: {altPct || '—'}%</p>
                </div>
              </div>
              <div className="mt-3 flex h-3 overflow-hidden rounded-full border border-white/10 bg-white/10">
                <div
                  className="h-full bg-emerald-400"
                  style={{ width: `${baseWidth}%` }}
                  aria-label={`Base Case ${basePct}%`}
                />
                <div
                  className="h-full bg-rose-400"
                  style={{ width: `${altWidth}%` }}
                  aria-label={`Alt Case ${altPct}%`}
                />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-300">
                {baseCase ? (
                  <button
                    type="button"
                    className={cn(
                      'rounded-full border px-3 py-1 uppercase tracking-[0.16em]',
                      activeScenario?.type === baseCase.type
                        ? 'border-emerald-300/60 bg-emerald-500/10 text-emerald-50'
                        : 'border-white/20 bg-white/5 text-slate-200 hover:border-white/40',
                    )}
                    onClick={() => setActiveType(baseCase.type)}
                  >
                    {baseCase.label} {basePct}%
                  </button>
                ) : null}
                {altCase ? (
                  <button
                    type="button"
                    className={cn(
                      'rounded-full border px-3 py-1 uppercase tracking-[0.16em]',
                      activeScenario?.type === altCase.type
                        ? 'border-rose-300/60 bg-rose-500/10 text-rose-50'
                        : 'border-white/20 bg-white/5 text-slate-200 hover:border-white/40',
                    )}
                    onClick={() => setActiveType(altCase.type)}
                  >
                    {altCase.label} {altPct}%
                  </button>
                ) : null}
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              {ordered.map((scenario) => {
                const probability = toPct(scenario.probability);
                const direction = directionLabels[scenario.direction] ?? directionLabels.neutral;
                const isActive = scenario.type === activeScenario?.type;
                return (
                  <button
                    type="button"
                    key={`${scenario.type}-${scenario.label}`}
                    onClick={() => setActiveType(scenario.type)}
                    className={cn(
                      'flex h-full flex-col rounded-2xl border p-4 text-left transition',
                      'bg-gradient-to-br from-white/5 to-white/0',
                      isActive ? 'border-violet-300/50 shadow-lg shadow-violet-500/10' : 'border-white/10 hover:border-white/20',
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="space-y-1">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{scenario.label}</p>
                        <Badge className={direction.className} variant="outline">
                          {direction.label}
                        </Badge>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-slate-500">Probability</p>
                        <p className="text-2xl font-semibold text-slate-50">{probability}%</p>
                      </div>
                    </div>
                    <div className="mt-3 h-2 rounded-full bg-white/10">
                      <div className="h-2 rounded-full bg-violet-500" style={{ width: `${probability}%` }} />
                    </div>
                    <p className="mt-3 text-sm text-slate-200 line-clamp-2">
                      {scenario.description || '等待情景描述'}
                    </p>
                    {isActive ? (
                      <div className="mt-4 space-y-3 text-xs text-slate-400">
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div>
                            <p className="uppercase tracking-[0.18em] text-slate-500">支撑</p>
                            <p className="mt-1 font-mono text-slate-100">{formatLevels(scenario.support ?? [])}</p>
                          </div>
                          <div>
                            <p className="uppercase tracking-[0.18em] text-slate-500">阻力</p>
                            <p className="mt-1 font-mono text-slate-100">{formatLevels(scenario.resistance ?? [])}</p>
                          </div>
                        </div>
                        <div>
                          <p className="uppercase tracking-[0.18em] text-slate-500">时间节奏</p>
                          <p className="mt-1 text-sm text-slate-100">{scenario.timeframe_notes || '时间节奏待更新'}</p>
                        </div>
                        {scenario.catalysts?.length ? (
                          <div>
                            <p className="uppercase tracking-[0.18em] text-slate-500">潜在催化</p>
                            <div className="mt-1 flex flex-wrap gap-2">
                              {scenario.catalysts.map((item) => (
                                <span key={item} className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[11px] text-slate-200">
                                  {item}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <p className="mt-3 text-[11px] uppercase tracking-[0.16em] text-slate-500">
                        点击展开详情 · 先看概率再看故事
                      </p>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
