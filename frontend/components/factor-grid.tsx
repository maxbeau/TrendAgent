import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useAionStore } from '@/store/aion-store';
import type { AionAnalysisResult, FactorKey } from '@/types/aion';
import { factorLabels } from '@/lib/factor-labels';

interface FactorGridProps {
  result?: AionAnalysisResult;
  onSelect?: (factor: FactorKey, data?: AionAnalysisResult['factors'][FactorKey]) => void;
  formulas?: Partial<Record<FactorKey, string>>;
  metaLoading?: boolean;
}

export function FactorGrid({ result, onSelect, formulas, metaLoading }: FactorGridProps) {
  const storeResult = useAionStore((state) => state.analysis);
  const factors = (result ?? storeResult)?.factors;

  const renderSummary = (text?: string) => {
    if (!text) return '等待模型结果...';
    const parts = text
      .split(/[;；]/)
      .map((p) => p.trim())
      .filter(Boolean);
    return parts.length ? (
      <div className="space-y-1">
        {parts.map((part) => (
          <p key={part} className="text-sm text-slate-200 leading-snug">
            {part}
          </p>
        ))}
      </div>
    ) : (
      text
    );
  };

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {(Object.entries(factorLabels) as [FactorKey, string][]).map(([key, label]) => {
        const factor = factors?.[key];
        const formula = formulas?.[key];
        const score = typeof factor?.score === 'number' ? factor.score : null;
        const weight = typeof factor?.weight === 'number' ? factor.weight : null;
        const weightedScore = typeof factor?.weighted_score === 'number' ? factor.weighted_score : null;
        return (
          <button
            key={key}
            type="button"
            className="text-left"
            onClick={() => onSelect?.(key, factor)}
            disabled={!onSelect}
          >
            <Card className="glass-card transition hover:border-violet-400/40">
              <CardHeader className="space-y-2 pb-2">
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-base">{label}</CardTitle>
                  <div className="flex items-center gap-2">
                    {weight !== null ? (
                      <Badge variant="outline" className="text-[10px] uppercase tracking-[0.2em]">
                        权重 · {(weight * 100).toFixed(0)}%
                      </Badge>
                    ) : null}
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
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {renderSummary(factor?.summary)}
                <p className="text-xs text-slate-500">
                  {metaLoading ? '加载算法...' : formula || (factor ? '算法待配置' : '点击运行 AION 获取最新数据')}
                </p>
                {weightedScore !== null ? (
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
                    加权分 · {weightedScore.toFixed(2)}
                  </p>
                ) : null}
              </CardContent>
            </Card>
          </button>
        );
      })}
    </div>
  );
}
