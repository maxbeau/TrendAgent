import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useAionStore } from '@/store/aion-store';
import type { AionAnalysisResult, FactorKey } from '@/types/aion';
import { factorLabels } from '@/lib/factor-labels';

const clampPct = (score: number | null) => {
  if (score === null || !Number.isFinite(score)) return 0;
  return Math.min(100, Math.max(0, (Number(score) / 5) * 100));
};

const summaryOneLiner = (text?: string) => {
  if (!text) return '等待模型结果...';
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return '等待模型结果...';
  const parts = cleaned.split(/[,;；。.!？!]/).map((p) => p.trim()).filter(Boolean);
  return parts[0] || cleaned;
};

interface FactorGridProps {
  result?: AionAnalysisResult;
  onSelect?: (factor: FactorKey, data?: AionAnalysisResult['factors'][FactorKey]) => void;
  metaLoading?: boolean;
}

export function FactorGrid({ result, onSelect, metaLoading }: FactorGridProps) {
  const storeResult = useAionStore((state) => state.analysis);
  const factors = (result ?? storeResult)?.factors;

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {(Object.entries(factorLabels) as [FactorKey, string][]).map(([key, label]) => {
        const factor = factors?.[key];
        const score = typeof factor?.score === 'number' ? factor.score : null;
        const weight = typeof factor?.weight === 'number' ? factor.weight : null;
        const weightedScore = typeof factor?.weighted_score === 'number' ? factor.weighted_score : null;
        const summary = summaryOneLiner(factor?.summary);
        const strengthPct = clampPct(score);
        return (
          <button
            key={key}
            type="button"
            className="text-left"
            onClick={() => onSelect?.(key, factor)}
            disabled={!onSelect}
          >
            <Card className="glass-card transition hover:border-violet-400/40 hover:shadow-violet-500/10">
              <CardHeader className="space-y-3 pb-2">
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-base">{label}</CardTitle>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'rounded-md border px-2 py-1 text-xs font-mono',
                        score !== null
                          ? score >= 4
                            ? 'border-bullish/50 bg-bullish/10 text-bullish'
                            : score <= 2.5
                              ? 'border-bearish/50 bg-bearish/10 text-bearish'
                              : 'border-warning/40 bg-warning/10 text-amber-100'
                          : 'border-slate-400/40 bg-white/5 text-slate-200',
                      )}
                    >
                      {score !== null ? score.toFixed(2) : '—'}
                    </span>
                    {weight !== null ? (
                      <Badge variant="outline" className="text-[10px] uppercase tracking-[0.2em]">
                        权重 {(weight * 100).toFixed(0)}%
                      </Badge>
                    ) : null}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-slate-200 truncate leading-snug" title={summary}>
                  {summary}
                </p>
                <div className="relative h-2 overflow-hidden rounded-full bg-white/5">
                  <div className="absolute inset-0 bg-gradient-to-r from-violet-500/10 via-aion/10 to-cyan-300/10" />
                  <div
                    className="relative h-full rounded-full bg-gradient-to-r from-violet-400 via-aion to-cyan-300 shadow-[0_0_0_1px_rgba(255,255,255,0.08)]"
                    style={{ width: `${strengthPct}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  <span>{metaLoading ? '加载算法...' : '点击查看详情'}</span>
                  {weightedScore !== null ? <span>加权 {weightedScore.toFixed(2)}</span> : null}
                </div>
              </CardContent>
            </Card>
          </button>
        );
      })}
    </div>
  );
}
