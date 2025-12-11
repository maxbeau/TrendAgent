import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { RefreshCcw } from 'lucide-react';

type DashboardHeaderProps = {
  ticker: string;
  ivHvBadgeText?: string | null;
  livePrice?: number;
  priceTone: string;
  pctLabel: string;
  lastSyncedValue: string;
  isRefreshing: boolean;
  onRefresh: () => void;
  reportErrorMessage?: string | null;
};

export function DashboardHeader({
  ticker,
  ivHvBadgeText,
  livePrice,
  priceTone,
  pctLabel,
  lastSyncedValue,
  isRefreshing,
  onRefresh,
  reportErrorMessage,
}: DashboardHeaderProps) {
  return (
    <section className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex items-baseline gap-3">
            <h1 className="text-4xl font-semibold tracking-tight">{ticker}</h1>
          </div>
          {ivHvBadgeText ? <Badge variant="warning">{ivHvBadgeText}</Badge> : null}
        </div>
        <p className="text-sm text-slate-400">
          Live Price{' '}
          <span className={cn('font-mono', priceTone)}>{livePrice ? `$${livePrice.toFixed(2)}` : '—'}</span>{' '}
          · Daily Δ <span className={cn('font-mono', priceTone)}>{pctLabel}</span>
        </p>
        {reportErrorMessage ? <p className="text-xs text-warning">{reportErrorMessage}</p> : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3">
          <div className="flex items-center gap-2">
            <span
              className={cn('h-2 w-2 rounded-full', isRefreshing ? 'bg-amber-300 animate-pulse' : 'bg-emerald-400')}
            />
            <div className="leading-tight">
              <p className="text-[10px] uppercase tracking-[0.26em] text-slate-500">Last Synced</p>
              <p className="font-mono text-sm text-slate-100">{lastSyncedValue}</p>
            </div>
          </div>
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="group inline-flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-100 transition hover:border-violet-400/40 hover:text-violet-50 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Refresh latest data"
            title="刷新最新数据"
          >
            <RefreshCcw className={cn('h-4 w-4', isRefreshing ? 'animate-spin' : 'group-hover:rotate-180')} />
          </button>
        </div>
      </div>
    </section>
  );
}
