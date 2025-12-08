'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import type { AionAnalysisResult } from '@/types/aion';

interface ActionCardProps {
  result?: AionAnalysisResult;
  fallbackTicker: string;
  isCalculating: boolean;
  isLoading?: boolean;
}

function scoreTone(scorePct: number) {
  if (scorePct >= 80) return 'text-bullish';
  if (scorePct <= 50) return 'text-bearish';
  return 'text-warning';
}

function signalLabel(signal: AionAnalysisResult['signal']) {
  switch (signal) {
    case 'STRONG_BUY':
      return 'STRONG BUY';
    case 'BUY':
      return 'BUY';
    case 'WAIT':
      return 'WAIT';
    case 'SELL':
      return 'SELL';
    case 'SHORT':
      return 'SHORT';
    default:
      return 'WAIT';
  }
}

function sizingHint(plan?: AionAnalysisResult['action_plan']) {
  if (!plan) return 'Sizing TBD';
  if (plan.risk_sizing === 'Half') return '⚠️ High Volatility: Half Size';
  if (plan.risk_sizing === 'Quarter') return '⚠️ Extreme Volatility: Quarter Size';
  return 'Standard Size';
}

function safeNumber(val: unknown): number | null {
  if (typeof val === 'number' && Number.isFinite(val)) return val;
  if (typeof val === 'string') {
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export function ActionCard({ result, fallbackTicker, isCalculating, isLoading }: ActionCardProps) {
  const hasResult = Boolean(result);
  const showSkeleton = (isLoading || isCalculating) && !hasResult;
  const showPlaceholder = !showSkeleton && !hasResult;
  const totalScoreRaw = result?.total_score ?? 0;
  const totalScorePct = Math.min(Math.max(totalScoreRaw, 0), 5) * 20; // 转换为百分制
  const signal = result?.signal ?? 'WAIT';
  const actionLabel = result?.action_card ?? 'Pending Signal';
  const derivedPlan = (() => {
    const volComponents = result?.factors?.volatility?.components as { expected_move?: { iv?: Record<string, unknown>; hv?: Record<string, unknown> } } | undefined;
    const pick = volComponents?.expected_move?.iv ?? volComponents?.expected_move?.hv;
    if (!pick) return undefined;
    const spot = safeNumber((pick as any).spot);
    const upper = safeNumber((pick as any).upper);
    const lower = safeNumber((pick as any).lower);
    if (upper === null && lower === null && spot === null) return undefined;
    const target_price = upper ?? (spot !== null ? spot * 1.05 : null);
    const stop_loss = lower ?? (spot !== null ? spot * 0.97 : null);
    const suggested_strategy =
      signal === 'STRONG_BUY' || signal === 'BUY'
        ? '买入正股或买入看涨期权'
        : signal === 'WAIT'
          ? '观望，等待确认'
          : signal === 'SELL'
            ? '逢高减仓或卖出认购'
            : '考虑做空或买入保护性看跌';
    const risk_sizing =
      signal === 'STRONG_BUY' || signal === 'BUY' ? 'Standard' : signal === 'WAIT' ? 'Half' : 'Quarter';
    return { target_price, stop_loss, suggested_strategy, risk_sizing };
  })();
  const actionPlan = result?.action_plan ?? derivedPlan;
  const formatPrice = (value?: number | null) => {
    const num = safeNumber(value);
    if (num === null || num <= 0) return '待生成';
    return `$${num.toFixed(2)}`;
  };
  const entryRange = (() => {
    const volComponents = result?.factors?.volatility?.components as { expected_move?: { iv?: Record<string, unknown>; hv?: Record<string, unknown> } } | undefined;
    const pick = volComponents?.expected_move?.iv ?? volComponents?.expected_move?.hv;
    const lower = pick ? safeNumber((pick as any).lower) : null;
    const upper = pick ? safeNumber((pick as any).upper) : null;
    const target = actionPlan?.target_price ?? null;
    const stop = actionPlan?.stop_loss ?? null;
    return {
      lower: lower ?? stop ?? null,
      upper: upper ?? target ?? null,
      basis: pick ? (volComponents?.expected_move?.iv ? 'IV' : 'HV') : null,
      days: pick ? (pick as any).days ?? 30 : null,
    };
  })();
  const formatRange = (lower?: number | null, upper?: number | null, suffix?: string) => {
    const l = safeNumber(lower);
    const u = safeNumber(upper);
    if (l === null && u === null) return '待生成';
    if (l !== null && u !== null) return `$${l.toFixed(2)} - $${u.toFixed(2)}${suffix ?? ''}`;
    if (u !== null) return `$${u.toFixed(2)}${suffix ?? ''}`;
    return `$${l!.toFixed(2)}${suffix ?? ''}`;
  };

  return (
    <Card>
      <CardHeader className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle>核心行动卡</CardTitle>
            <CardDescription>模型决策与风险管理摘要</CardDescription>
          </div>
          <Badge variant="accent">Signal</Badge>
        </div>
        {isCalculating ? <p className="text-xs text-slate-400">Calculating…</p> : null}
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-2">
        {showSkeleton ? (
          <>
            <div className="space-y-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Total Score</p>
              <Skeleton className="h-12 w-32" />
              <div className="flex items-center gap-3 text-sm">
                <Skeleton className="h-8 w-24 rounded-full" />
                <Skeleton className="h-4 w-20" />
              </div>
              <Skeleton className="h-24 w-full rounded-xl" />
            </div>
            <div className="space-y-4 rounded-xl border border-white/10 bg-white/5 p-4">
              <Skeleton className="h-4 w-1/3" />
              <div className="grid grid-cols-2 gap-3 text-sm">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
              <Skeleton className="h-4 w-2/3" />
            </div>
          </>
        ) : showPlaceholder ? (
          <div className="md:col-span-2">
            <div className="space-y-3 rounded-xl border border-white/10 bg-white/5 p-4 text-sm text-slate-200">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Waiting For Data</p>
              <p>暂无模型结果，请先运行 AION 引擎或等待后台同步。</p>
            </div>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Total Score</p>
                  <span className="rounded-full border border-bullish/40 bg-bullish/10 px-2 py-1 text-[11px] uppercase tracking-[0.16em] text-bullish">
                    {signalLabel(signal)}
                  </span>
                </div>
                <div className="mt-2 flex items-baseline gap-3">
                  <div className={cn('relative inline-block pr-14 font-mono text-5xl font-semibold leading-none', scoreTone(totalScorePct))}>
                    <span>{totalScorePct.toFixed(1)}</span>
                    <span className="absolute bottom-1 right-3 text-base text-slate-400">/100</span>
                  </div>
                  <span className="text-xs uppercase tracking-[0.2em] text-bullish">{actionLabel}</span>
                </div>
                <p className="mt-2 text-xs text-slate-400">基于 1-5 分模型换算的百分制，便于快速对比。</p>
              </div>

              <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Sizing & Timing</p>
                <p className="mt-2 text-sm text-amber-100">{sizingHint(actionPlan)}</p>
                <p className="text-xs text-slate-400">根据波动与信号自动给出仓位提示，后端计划接入实时风险限额。</p>
              </div>
            </div>

            <div className="space-y-4 rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Trade Plan</p>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-1">
                  <p className="text-slate-400">Target Price</p>
                  <p className="font-mono text-bullish text-lg">{formatPrice(actionPlan?.target_price)}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-slate-400">Stop Loss</p>
                  <p className="font-mono text-bearish text-lg">{formatPrice(actionPlan?.stop_loss)}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-slate-400">Strategy</p>
                  <p className="font-mono text-slate-100">{actionPlan?.suggested_strategy ?? '待生成'}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-slate-400">Risk Sizing</p>
                  <p className="font-mono text-amber-100">{actionPlan?.risk_sizing ?? '待生成'}</p>
                </div>
              </div>
              <div className="mt-2 rounded-lg border border-white/10 bg-white/5 p-3 text-xs text-slate-300">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Entry Range</p>
                <p className="mt-1 font-mono text-slate-100">
                  {formatRange(entryRange.lower, entryRange.upper, entryRange.basis ? `（${entryRange.days ?? 30}日基于${entryRange.basis}）` : '')}
                </p>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
