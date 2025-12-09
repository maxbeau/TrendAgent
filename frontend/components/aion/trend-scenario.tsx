import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
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

export function TrendScenario({ scenarios }: { scenarios?: TrendScenarioType[] } = {}) {
  const storeScenarios = useAionStore((state) => state.analysis?.scenarios);
  const ordered = (scenarios ?? storeScenarios ?? [])
    .slice()
    .sort((a, b) => scenarioPriority[a.type] - scenarioPriority[b.type]);

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
        {ordered.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-300">
            暂无情景分析数据，等待后端接口接入。当前展示为占位信息。
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            {ordered.map((scenario) => {
              const probability = Math.min(100, Math.max(0, Math.round((scenario.probability ?? 0) * 100)));
              const direction = directionLabels[scenario.direction] ?? directionLabels.neutral;
              return (
                <div key={`${scenario.type}-${scenario.label}`} className="rounded-2xl border border-white/10 bg-gradient-to-br from-white/5 to-white/0 p-4 shadow-sm">
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
                  <p className="mt-3 text-sm text-slate-200">{scenario.description || '等待情景描述'}</p>
                  <div className="mt-4 grid gap-3 text-xs text-slate-400 sm:grid-cols-2">
                    <div>
                      <p className="uppercase tracking-[0.18em] text-slate-500">支撑</p>
                      <p className="mt-1 font-mono text-slate-100">{formatLevels(scenario.support ?? [])}</p>
                    </div>
                    <div>
                      <p className="uppercase tracking-[0.18em] text-slate-500">阻力</p>
                      <p className="mt-1 font-mono text-slate-100">{formatLevels(scenario.resistance ?? [])}</p>
                    </div>
                  </div>
                  <div className="mt-4 space-y-2 text-xs text-slate-400">
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
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
