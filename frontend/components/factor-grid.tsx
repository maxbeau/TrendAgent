import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { AionAnalysisResult, FactorKey } from '@/types/aion';
import { factorLabels } from '@/lib/factor-labels';

interface FactorGridProps {
  result?: AionAnalysisResult;
  onSelect?: (factor: FactorKey, data?: AionAnalysisResult['factors'][FactorKey]) => void;
  formulas?: Partial<Record<FactorKey, string>>;
  metaLoading?: boolean;
}

export function FactorGrid({ result, onSelect, formulas, metaLoading }: FactorGridProps) {
  const factors = result?.factors;

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
        return (
          <button
            key={key}
            type="button"
            className="text-left"
            onClick={() => onSelect?.(key, factor)}
            disabled={!onSelect}
          >
            <Card className="glass-card transition hover:border-violet-400/40">
              <CardHeader className="flex items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base">{label}</CardTitle>
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
              </CardHeader>
              <CardContent className="space-y-2">
                {renderSummary(factor?.summary)}
                <p className="text-xs text-slate-500">
                  {metaLoading ? '加载算法...' : formula || (factor ? '算法待配置' : '点击运行 AION 获取最新数据')}
                </p>
              </CardContent>
            </Card>
          </button>
        );
      })}
    </div>
  );
}
