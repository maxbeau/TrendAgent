import { useMemo } from 'react';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useLiveQuote } from '@/hooks/use-live-quote';
import { useAionStore } from '@/store/aion-store';
import type {
  ExecutionNotes,
  OptionStrategy,
  RiskManagement,
  StockStrategy,
  StrategyLeg,
} from '@/types/aion';

const formatLeg = (leg: StrategyLeg) => `${leg.action === 'buy' ? '买入' : '卖出'} ${leg.strike} ${leg.type.toUpperCase()}`;

const extractPrice = (raw?: string | number | null) => {
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
  if (typeof raw !== 'string') return null;
  const match = raw.replace(/,/g, '').match(/(\d+(?:\.\d+)?)/);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
};

const parseEntryZone = (zone?: string) => {
  if (!zone || typeof zone !== 'string') return null;
  const parts = zone.split(/[–—-]/).map((part) => extractPrice(part.trim()));
  if (parts.length === 1) return { lower: parts[0], upper: parts[0] };
  const [lower, upper] = parts;
  return { lower: lower ?? null, upper: upper ?? null };
};

const distanceToneClass: Record<'in' | 'below' | 'above' | 'unknown', string> = {
  in: 'border-emerald-400/40 bg-emerald-500/10 text-emerald-100',
  below: 'border-amber-300/40 bg-amber-500/10 text-amber-50',
  above: 'border-rose-300/40 bg-rose-500/10 text-rose-50',
  unknown: 'border-white/10 bg-white/5 text-slate-200',
};

const buildEntryDistance = (price: number | null, zone?: string) => {
  if (!Number.isFinite(price)) return null;
  const parsed = parseEntryZone(zone);
  if (!parsed || (parsed.lower === null && parsed.upper === null)) return null;
  const { lower, upper } = parsed;

  if (lower !== null && price < lower) {
    const pct = ((lower - price) / lower) * 100;
    return { label: `距离买点还有 ${pct.toFixed(1)}%`, tone: 'below' as const };
  }
  if (upper !== null && price > upper) {
    const pct = ((price - upper) / upper) * 100;
    return { label: `已偏离入场区 ${pct.toFixed(1)}%`, tone: 'above' as const };
  }
  if (lower !== null || upper !== null) {
    return { label: '价格在入场区附近，关注执行节奏', tone: 'in' as const };
  }
  return { label: '等待入场区或行情数据更新', tone: 'unknown' as const };
};

export function StrategyMatrix({
  stockStrategy,
  optionStrategies,
  riskManagement,
  executionNotes,
}: {
  stockStrategy?: StockStrategy;
  optionStrategies?: OptionStrategy[];
  riskManagement?: RiskManagement;
  executionNotes?: ExecutionNotes;
} = {}) {
  const analysis = useAionStore((state) => state.analysis);
  const stock = stockStrategy ?? analysis?.stock_strategy;
  const options = optionStrategies ?? analysis?.option_strategies ?? [];
  const risk = riskManagement ?? analysis?.risk_management;
  const execNotes = executionNotes ?? analysis?.execution_notes;
  const hasData = stock || options.length || risk || execNotes;
  const liveQuote = useLiveQuote();
  const entryDistance = useMemo(
    () => buildEntryDistance(liveQuote?.close ?? null, stock?.entry_zone),
    [liveQuote?.close, stock?.entry_zone],
  );

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>策略执行矩阵</CardTitle>
          <CardDescription>正股执行逻辑 + 期权组合 + 风控提示</CardDescription>
        </div>
        <Badge variant="outline">Execution</Badge>
      </CardHeader>
      <CardContent className="space-y-6">
        {!hasData ? (
          <div className="rounded-xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-300">
            暂无策略矩阵数据，可在后端同步完成后展示；当前显示为占位卡片。
          </div>
        ) : (
          <>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">正股策略</p>
                {stock ? (
                  <div className="mt-3 space-y-3 text-sm text-slate-200">
                    <div>
                      <p className="text-xs text-slate-500">入场区间</p>
                      <p className="font-mono text-lg text-slate-100">{stock.entry_zone}</p>
                    </div>
                    {entryDistance ? (
                      <div
                        className={`rounded-lg border px-3 py-2 text-xs ${distanceToneClass[entryDistance.tone] ?? distanceToneClass.unknown}`}
                      >
                        <p className="text-[10px] uppercase tracking-[0.2em] text-white/70">价格锚点</p>
                        <p className="mt-1 font-mono text-sm">{entryDistance.label}</p>
                      </div>
                    ) : null}
                    {stock.add_conditions?.length ? (
                      <div>
                        <p className="text-xs text-slate-500">加仓条件</p>
                        <ul className="mt-1 list-inside list-disc space-y-1 text-slate-300">
                          {stock.add_conditions.map((condition) => (
                            <li key={`add-${condition}`}>{condition}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {stock.reduce_conditions?.length ? (
                      <div>
                        <p className="text-xs text-slate-500">减仓 / 风险提示</p>
                        <ul className="mt-1 list-inside list-disc space-y-1 text-slate-300">
                          {stock.reduce_conditions.map((condition) => (
                            <li key={`reduce-${condition}`}>{condition}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    <div>
                      <p className="text-xs text-slate-500">止盈目标</p>
                      <p className="font-mono text-lg text-emerald-200">{stock.profit_target}</p>
                    </div>
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-slate-400">等待正股策略输出...</p>
                )}
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">期权策略</p>
                {options.length ? (
                  <div className="mt-3 space-y-3">
                    {options.map((strategy) => (
                      <div key={strategy.name} className="rounded-xl border border-white/10 bg-white/5 p-3">
                        <div className="flex items-center justify-between">
                          <p className="font-medium text-slate-100">{strategy.name}</p>
                          {strategy.expiration_notes ? (
                            <span className="text-xs text-slate-400">{strategy.expiration_notes}</span>
                          ) : null}
                        </div>
                        {strategy.description ? (
                          <p className="mt-1 text-xs text-slate-400">{strategy.description}</p>
                        ) : null}
                        <ul className="mt-3 space-y-1 text-sm text-slate-200">
                          {strategy.legs.map((leg, idx) => (
                            <li key={`${strategy.name}-leg-${idx}`} className="font-mono">
                              {formatLeg(leg)}
                              {leg.expiration ? ` · ${leg.expiration}` : ''}
                            </li>
                          ))}
                        </ul>
                        {strategy.rationale ? (
                          <p className="mt-2 text-xs text-slate-400">{strategy.rationale}</p>
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-slate-400">等待期权策略输出...</p>
                )}
              </div>
            </div>
            {(risk || execNotes) && (
              <div className="grid gap-4 lg:grid-cols-2">
                {risk ? (
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">风险管理</p>
                    <div className="mt-3 space-y-2 text-sm text-slate-200">
                      <p>初始仓位：{risk.initial_position}</p>
                      <p>最大敞口：{risk.max_exposure}</p>
                      <p>加仓规则：{risk.add_rule}</p>
                      <p>止损规则：{risk.stop_loss_rule}</p>
                      {risk.odds_rating ? <p>赔率评级：{risk.odds_rating}</p> : null}
                      {risk.win_rate_rr ? <p>胜率 × 盈亏比：{risk.win_rate_rr}</p> : null}
                    </div>
                  </div>
                ) : null}
                {execNotes ? (
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">执行提示</p>
                    <div className="mt-3 space-y-2 text-sm text-slate-200">
                      {execNotes.observation_cycle?.length ? (
                        <div>
                          <p className="text-xs text-slate-500">观察周期</p>
                          <div className="mt-1 flex flex-wrap gap-2">
                            {execNotes.observation_cycle.map((cycle) => (
                              <Badge key={cycle} variant="outline">
                                {cycle}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      {execNotes.signals_to_watch?.length ? (
                        <div>
                          <p className="text-xs text-slate-500">重点信号</p>
                          <ul className="mt-1 list-inside list-disc space-y-1 text-slate-300">
                            {execNotes.signals_to_watch.map((signal) => (
                              <li key={signal}>{signal}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
