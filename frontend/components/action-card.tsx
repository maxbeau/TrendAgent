'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { safeNumber } from '@/lib/numbers';
import { pickExpectedMove, type RawExpectedMove } from '@/lib/volatility';
import { cn } from '@/lib/utils';
import { useAionStore } from '@/store/aion-store';
import type { AionAnalysisResult } from '@/types/aion';

interface ActionCardProps {
  result?: AionAnalysisResult;
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

const signalThemes: Record<
  AionAnalysisResult['signal'],
  { wrap: string; badge: string; text: string; ring: string }
> = {
  STRONG_BUY: {
    wrap: 'border-emerald-400/60 bg-gradient-to-br from-emerald-500/20 via-emerald-400/10 to-emerald-950/20 shadow-[0_0_30px_-5px_rgba(16,185,129,0.3)]',
    badge: 'border-emerald-300/60 bg-emerald-500/20 text-emerald-100 shadow-[0_0_10px_rgba(16,185,129,0.2)]',
    text: 'text-emerald-50 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]',
    ring: 'shadow-[0_0_0_4px_rgba(16,185,129,0.15),0_0_0_8px_rgba(16,185,129,0.05)]',
  },
  BUY: {
    wrap: 'border-emerald-300/40 bg-gradient-to-br from-emerald-400/15 via-emerald-300/5 to-transparent shadow-[0_0_20px_-5px_rgba(52,211,153,0.2)]',
    badge: 'border-emerald-200/50 bg-emerald-400/15 text-emerald-50',
    text: 'text-emerald-50',
    ring: 'shadow-[0_0_0_4px_rgba(52,211,153,0.1)]',
  },
  WAIT: {
    wrap: 'border-amber-200/40 bg-gradient-to-br from-amber-400/15 via-amber-300/5 to-transparent shadow-[0_0_20px_-5px_rgba(251,191,36,0.15)]',
    badge: 'border-amber-200/50 bg-amber-500/15 text-amber-50',
    text: 'text-amber-50',
    ring: 'shadow-[0_0_0_4px_rgba(251,191,36,0.1)]',
  },
  SELL: {
    wrap: 'border-rose-300/40 bg-gradient-to-br from-rose-500/15 via-rose-400/5 to-transparent shadow-[0_0_20px_-5px_rgba(251,113,133,0.2)]',
    badge: 'border-rose-200/50 bg-rose-500/15 text-rose-50',
    text: 'text-rose-50',
    ring: 'shadow-[0_0_0_4px_rgba(251,113,133,0.1)]',
  },
  SHORT: {
    wrap: 'border-rose-400/60 bg-gradient-to-br from-rose-500/20 via-rose-500/10 to-rose-950/20 shadow-[0_0_30px_-5px_rgba(244,63,94,0.3)]',
    badge: 'border-rose-300/60 bg-rose-600/20 text-rose-50 shadow-[0_0_10px_rgba(244,63,94,0.2)]',
    text: 'text-rose-50 drop-shadow-[0_0_8px_rgba(244,63,94,0.5)]',
    ring: 'shadow-[0_0_0_4px_rgba(244,63,94,0.15),0_0_0_8px_rgba(244,63,94,0.05)]',
  },
};

export function ActionCard({ result, isCalculating, isLoading }: ActionCardProps) {
  const storeResult = useAionStore((state) => state.analysis);
  const resolvedResult = result ?? storeResult;
  const hasResult = Boolean(resolvedResult);
  const showSkeleton = isCalculating || (isLoading && !hasResult);
  const showPlaceholder = !showSkeleton && !hasResult;
  const totalScoreRaw = resolvedResult?.total_score ?? 0;
  const totalScorePct = Math.min(Math.max(totalScoreRaw, 0), 5) * 20; // 转换为百分制
  const signal = resolvedResult?.signal ?? 'WAIT';
  const actionLabel = resolvedResult?.action_card ?? 'Pending Signal';
  const volComponents = resolvedResult?.factors?.volatility?.components as { expected_move?: RawExpectedMove } | undefined;
  const expectedMove = pickExpectedMove(volComponents?.expected_move);
  const derivedPlan = (() => {
    if (!expectedMove) return undefined;
    const { spot, upper, lower } = expectedMove;
    if (upper === null && lower === null && spot === null) return undefined;
    const target_price = upper ?? (spot !== null ? spot * 1.05 : null);
    const stop_loss = lower ?? (spot !== null ? spot * 0.97 : null);
    if (target_price === null || stop_loss === null) return undefined;
    const suggested_strategy =
      signal === 'STRONG_BUY' || signal === 'BUY'
        ? '买入正股或买入看涨期权'
        : signal === 'WAIT'
          ? '观望，等待确认'
          : signal === 'SELL'
            ? '逢高减仓或卖出认购'
            : '考虑做空或买入保护性看跌';
    const risk_sizing: "Standard" | "Half" | "Quarter" =
      signal === 'STRONG_BUY' || signal === 'BUY' ? 'Standard' : signal === 'WAIT' ? 'Half' : 'Quarter';
    return { target_price, stop_loss, suggested_strategy, risk_sizing };
  })();
  const actionPlan = resolvedResult?.action_plan ?? derivedPlan;
  const formatPrice = (value?: number | null) => {
    const num = safeNumber(value);
    if (num === null || num <= 0) return '待生成';
    return `$${num.toFixed(2)}`;
  };
  const entryRange = (() => {
    const lower = expectedMove?.lower ?? null;
    const upper = expectedMove?.upper ?? null;
    const target = actionPlan?.target_price ?? null;
    const stop = actionPlan?.stop_loss ?? null;
    return {
      lower: lower ?? stop ?? null,
      upper: upper ?? target ?? null,
      basis: expectedMove?.basis ?? null,
      days: expectedMove?.days ?? null,
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
  const signalTheme = signalThemes[signal] ?? signalThemes.WAIT;

  return (
    <Card>
      <CardHeader className="flex flex-col gap-2">
        <CardTitle>核心行动卡</CardTitle>
        <CardDescription>模型决策与风险管理摘要</CardDescription>
        {isCalculating ? <p className="text-xs text-slate-400">Calculating…</p> : null}
      </CardHeader>
      <CardContent className="space-y-6">
        {showSkeleton ? (
          <div className="grid gap-4 md:grid-cols-2">
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
          </div>
        ) : showPlaceholder ? (
          <div className="md:col-span-2">
            <div className="space-y-3 rounded-xl border border-white/10 bg-white/5 p-4 text-sm text-slate-200">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Waiting For Data</p>
              <p>暂无模型结果，请先运行 AION 引擎或等待后台同步。</p>
            </div>
          </div>
        ) : (
          <>
            <div
              className={cn(
                'relative overflow-hidden rounded-2xl border p-5 transition-all duration-500 hover:scale-[1.01]',
                signalTheme.wrap,
              )}
            >
              <div className="absolute -left-10 top-1/2 h-40 w-40 -translate-y-1/2 rounded-full bg-white/10 blur-3xl" />
              <div className="absolute -right-10 bottom-0 h-32 w-32 rounded-full bg-white/5 blur-3xl" />
              <div className="relative flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-3">
                  <span className={cn('rounded-full border px-3 py-1 text-[11px] uppercase tracking-[0.2em]', signalTheme.badge)}>
                    {signalLabel(signal)}
                  </span>
                  <p className="text-sm uppercase tracking-[0.2em] text-white/80">最新模型信号</p>
                </div>
                <div className="flex items-baseline gap-3">
                  <div
                    className={cn(
                      'relative inline-flex items-baseline gap-2 rounded-lg px-3 py-2 font-mono text-5xl font-semibold leading-none',
                      scoreTone(totalScorePct),
                      signalTheme.ring,
                    )}
                  >
                    <span>{totalScorePct.toFixed(1)}</span>
                    <span className="text-base text-white/70">/100</span>
                  </div>
                  <div className="text-left">
                    <p className="text-xs uppercase tracking-[0.16em] text-white/70">Confidence</p>
                    <p className={cn('text-sm font-semibold', signalTheme.text)}>{actionLabel}</p>
                  </div>
                </div>
              </div>
              <p className="mt-2 text-xs text-white/70">优先让用户先看到「是什么信号」与「有多大把握」，支持实时刷新。</p>
            </div>

            <div className="grid gap-4 lg:grid-cols-[1.05fr_1fr]">
              <div className="space-y-3 rounded-xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Sizing & Timing</p>
                <p className="mt-2 text-sm text-amber-100">{sizingHint(actionPlan)}</p>
                <p className="text-xs text-slate-400">根据波动与信号自动给出仓位提示，后端计划接入实时风险限额。</p>
                <div className="mt-3 rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-slate-300">
                  <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Entry Range</p>
                  <p className="mt-1 font-mono text-slate-100">
                    {formatRange(entryRange.lower, entryRange.upper, entryRange.basis ? `（${entryRange.days ?? 30}日基于${entryRange.basis}）` : '')}
                  </p>
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
                <div className="rounded-lg border border-white/10 bg-white/5 p-3 text-xs text-slate-300">
                  <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Risk Note</p>
                  <p className="mt-1">
                    保持执行纪律：入场-止盈-止损需成套考虑，价格偏离时优先控制仓位。
                  </p>
                </div>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
