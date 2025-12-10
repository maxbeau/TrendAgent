import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useAionStore } from '@/store/aion-store';
import type { KeyVariable } from '@/types/aion';

const impactMap: Record<KeyVariable['impact'], { label: string; className: string; indicator: string }> = {
  bullish: { label: 'Bullish', className: 'text-bullish', indicator: '↑' },
  bearish: { label: 'Bearish', className: 'text-bearish', indicator: '↓' },
  neutral: { label: 'Neutral', className: 'text-slate-200', indicator: '→' },
};

export function KeyVariableTable({ variables, isLoading }: { variables?: KeyVariable[]; isLoading?: boolean } = {}) {
  const storeVariables = useAionStore((state) => state.analysis?.key_variables);
  const rows = isLoading ? [] : variables ?? storeVariables ?? [];

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>关键变量监控</CardTitle>
          <CardDescription>触发阈值 · 影响方向 · 操作建议</CardDescription>
        </div>
        <Badge variant="outline">Key Variables</Badge>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, idx) => (
              <div key={idx} className="rounded-xl border border-dashed border-white/20 bg-white/5 p-4">
                <Skeleton className="h-4 w-32" />
                <div className="mt-3 grid grid-cols-3 gap-2">
                  <Skeleton className="h-5 w-full" />
                  <Skeleton className="h-5 w-full" />
                  <Skeleton className="h-5 w-full" />
                </div>
              </div>
            ))}
          </div>
        ) : rows.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-300">
            暂无关键变量数据，可在后端就绪后自动填充。当前展示为默认占位。
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm text-slate-200">
              <thead>
                <tr className="text-xs uppercase tracking-[0.2em] text-slate-500">
                  <th className="py-2 pr-4 font-semibold">变量</th>
                  <th className="py-2 pr-4 font-semibold">触发阈值</th>
                  <th className="py-2 pr-4 font-semibold">影响方向</th>
                  <th className="py-2 pr-4 font-semibold">操作建议</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-sm">
                {rows.map((item) => {
                  const impact = impactMap[item.impact] ?? impactMap.neutral;
                  return (
                    <tr key={`${item.name}-${item.threshold}`}>
                      <td className="py-3 pr-4 font-medium text-slate-100">{item.name}</td>
                      <td className="py-3 pr-4">
                        <Badge variant="accent" className="text-[11px] uppercase tracking-wide">
                          {item.threshold}
                        </Badge>
                      </td>
                      <td className="py-3 pr-4">
                        <span className={`inline-flex items-center gap-1 font-semibold ${impact.className}`}>
                          {impact.indicator} {impact.label}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-slate-300">{item.suggestion}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
