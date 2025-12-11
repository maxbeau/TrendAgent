import { useMemo } from 'react';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { FactorKey } from '@/types/aion';

const prettyKey = (key: string) => key.replace(/^.+?\./, '').replace(/_/g, ' ').toUpperCase();

type FactorDialogData = {
  key: FactorKey;
  summary?: string;
  key_evidence?: string[];
  sources?: Array<{ title?: string; url?: string; source?: string }>;
  components?: Record<string, unknown>;
};

type NormalizedEventComponent = {
  label?: string;
  description?: string;
  date?: string;
  formattedDate?: string | null;
  source?: string;
};

type FactorDetailsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  factorKey?: FactorKey;
  factorLabel?: string;
  data: FactorDialogData | null;
  formulaText?: string;
};

export function FactorDetailsDialog({
  open,
  onOpenChange,
  factorKey,
  factorLabel,
  data,
  formulaText,
}: FactorDetailsDialogProps) {
  const renderSummary = (text?: string) => {
    if (!text) return '暂无摘要，可重新运行引擎获取最新结果。';
    const parts = text
      .split(/[;；]/)
      .map((p) => p.trim())
      .filter(Boolean);
    if (!parts.length) return text;
    return (
      <div className="space-y-1">
        {parts.map((part) => (
          <p key={part}>{part}</p>
        ))}
      </div>
    );
  };

  const factorComponents = data?.components as Record<string, any> | undefined;
  const isCatalystDialog = factorKey === 'catalyst';
  const toNum = (val: unknown) => {
    const num = Number(val);
    return Number.isFinite(num) ? num : null;
  };
  const formatEventDate = (value?: string) => {
    if (!value) return null;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(parsed);
  };
  const normalizeEventComponent = (raw: any): NormalizedEventComponent | null => {
    if (!raw || typeof raw !== 'object') return null;
    const label = typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : undefined;
    const description = typeof raw.description === 'string' && raw.description.trim() ? raw.description.trim() : undefined;
    const date = typeof raw.date === 'string' && raw.date.trim() ? raw.date.trim() : undefined;
    const source = typeof raw.source === 'string' && raw.source.trim() ? raw.source.trim() : undefined;
    const formattedDate = formatEventDate(date);
    if (!label && !description && !date) return null;
    return { label, description, date, formattedDate, source };
  };
  const weightContext = useMemo(() => {
    const weights = factorComponents?.weights_used;
    if (!weights || typeof weights !== 'object') return null;
    const map: Record<string, number> = {};
    Object.entries(weights).forEach(([k, v]) => {
      const num = toNum(v);
      if (num !== null) map[k] = num;
    });
    if (!Object.keys(map).length) return null;
    return { map, denom: toNum(factorComponents?.weight_denominator) };
  }, [factorComponents]);
  const weightSummary = useMemo(() => {
    if (!weightContext) return null;
    const denomNum = weightContext.denom;
    const text = Object.entries(weightContext.map)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k, base]) => {
        const pct = denomNum && denomNum > 0 ? (base / denomNum) * 100 : base * 100;
        return `${prettyKey(k)} ${pct.toFixed(0)}%`;
      })
      .join(' · ');
    const denomText = denomNum !== null ? `（权重基数 ${denomNum.toFixed(1)}）` : '';
    return `${text} ${denomText}`.trim();
  }, [weightContext]);
  const factorScoreSummary = useMemo(() => {
    const factorScores = factorComponents?.factor_scores;
    if (!factorScores || typeof factorScores !== 'object') return null;
    const entries = Object.entries(factorScores)
      .filter(([, v]) => typeof v === 'number')
      .map(([k, v]) => ({ key: k, score: Number(v) }));
    return entries.length ? entries : null;
  }, [factorComponents]);

  const factorScoreList = useMemo(() => {
    if (!factorScoreSummary) return [];
    const weights = weightContext?.map || {};
    const denomNum = weightContext?.denom ?? null;
    return factorScoreSummary.map((item) => {
      const weightNum = weights[item.key];
      const weight = Number.isFinite(weightNum)
        ? denomNum !== null && denomNum > 0
          ? (weightNum / denomNum) * 100
          : weightNum * 100
        : null;
      return {
        label: prettyKey(item.key),
        score: item.score,
        weight,
      };
    });
  }, [factorScoreSummary, weightContext]);
  const selectedEvent = useMemo(() => normalizeEventComponent(factorComponents?.selected_event), [factorComponents]);
  const eventCandidates = useMemo(() => {
    const raw = factorComponents?.event_candidates;
    if (!Array.isArray(raw)) return [];
    return raw
      .map((item) => normalizeEventComponent(item))
      .filter((item): item is NormalizedEventComponent => Boolean(item));
  }, [factorComponents]);
  const eventSourceCount = useMemo(() => {
    const val = toNum(factorComponents?.event_source_count);
    return val !== null ? Math.max(0, Math.round(val)) : null;
  }, [factorComponents]);
  const componentDetails = useMemo(() => {
    if (!factorComponents) return [];
    const exclude = new Set([
      'weights_used',
      'factor_scores',
      'weight_denominator',
      'reasons',
      'citations',
      'asof_date',
      'confidence',
      'selected_event',
      'event_candidates',
      'event_source_count',
    ]);
    const formatValue = (key: string, val: any) => {
      if (val === null || val === undefined) return null;
      if (typeof val === 'number') {
        const num = Number.isFinite(val) ? val : null;
        if (num === null) return null;
        const lower = key.toLowerCase();
        if (lower.includes('pct') || lower.includes('ratio')) return `${(num * 100).toFixed(2)}%`;
        if (lower.includes('vol') || lower.includes('iv') || lower.includes('hv')) return `${num.toFixed(4)}`;
        return num.toFixed(2);
      }
      if (typeof val === 'string') return val;
      if (typeof val === 'object' && val !== null) {
        if ('lower' in val && 'upper' in val) {
          const l = Number((val as any).lower);
          const u = Number((val as any).upper);
          if (Number.isFinite(l) && Number.isFinite(u)) return `${l.toFixed(2)} ~ ${u.toFixed(2)}`;
        }
        if ('put_call_ratio' in val) {
          const pcr = Number((val as any).put_call_ratio);
          return Number.isFinite(pcr) ? `PCR ${pcr.toFixed(2)}` : null;
        }
        if ('spot' in val && 'expiration' in val) {
          const spot = Number((val as any).spot);
          const exp = (val as any).expiration;
          const spotText = Number.isFinite(spot) ? `$${spot.toFixed(2)}` : '—';
          return `Spot ${spotText} · Exp ${exp ?? '—'}`;
        }
      }
      try {
        return JSON.stringify(val);
      } catch {
        return String(val);
      }
    };
    return Object.entries(factorComponents)
      .filter(([key]) => !exclude.has(key))
      .map(([key, val]) => ({ key: prettyKey(key), value: formatValue(key, val) }))
      .filter((item) => item.value !== null && item.value !== undefined && item.value !== '');
  }, [factorComponents]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="z-50 flex w-full max-w-3xl flex-col border border-white/10 bg-obsidian-950/95 backdrop-blur-xl duration-200 sm:max-w-4xl">
        <DialogHeader className="px-1 pt-4">
          <DialogTitle>因子详情 · {factorLabel ?? ''}</DialogTitle>
          <DialogDescription>AION 因子输出</DialogDescription>
        </DialogHeader>
        <div className="mt-4 flex-1 overflow-y-auto px-1 pb-8 min-h-0">
          <div className="space-y-3 text-sm text-slate-200">
            {renderSummary(data?.summary)}
            <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">关键证据</p>
              {data?.key_evidence?.length ? (
                <ul className="list-disc space-y-1 pl-4 text-xs text-slate-100">
                  {data.key_evidence.map((item) => (
                    <li key={item} className="leading-snug">
                      {item}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-400">暂无证据摘要，可重新运行引擎。</p>
              )}
            </div>
            <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">算法 & 权重</p>
              <p className="text-sm text-slate-100">{formulaText ?? '算法待配置'}</p>
              <p className="text-xs text-slate-300">当前权重：{weightSummary ?? '暂无权重数据'}</p>
              {factorScoreList.length ? (
                <div className="space-y-1 text-xs text-slate-300">
                  {factorScoreList.map((item) => (
                    <div key={item.label} className="flex items-center justify-between">
                      <span>{item.label}</span>
                      <span className="font-mono text-slate-100">
                        {item.score.toFixed(1)}
                        {item.weight !== null ? ` · ${item.weight.toFixed(0)}%` : ''}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-400">暂无子因子得分，可重新运行引擎。</p>
              )}
            </div>
            {isCatalystDialog ? (
              <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">近期催化事件</p>
                {selectedEvent ? (
                  <div className="rounded border border-white/10 bg-black/20 p-2 text-xs text-slate-200">
                    <p className="text-sm font-medium text-slate-100">{selectedEvent.label ?? '事件'}</p>
                    {selectedEvent.description ? (
                      <p className="text-xs text-slate-300">{selectedEvent.description}</p>
                    ) : null}
                    <p className="text-[11px] text-slate-500">
                      {selectedEvent.formattedDate ?? selectedEvent.date ?? '日期待定'}
                      {selectedEvent.source ? ` · 来源 ${selectedEvent.source}` : ''}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">暂无结构化事件，可刷新或等待更多新闻。</p>
                )}
                {eventCandidates.length ? (
                  <div className="space-y-1 text-xs text-slate-300">
                    {eventCandidates.map((candidate, idx) => (
                      <div key={`${candidate.label ?? 'event'}-${candidate.date ?? idx}`} className="flex flex-col rounded bg-black/10 p-2">
                        <span className="text-[11px] uppercase tracking-wide text-slate-500">
                          {candidate.label ?? 'EVENT'}
                        </span>
                        {candidate.description ? <span className="text-slate-200">{candidate.description}</span> : null}
                        <span className="text-[11px] text-slate-500">
                          {candidate.formattedDate ?? candidate.date ?? '日期待定'}
                          {candidate.source ? ` · ${candidate.source}` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : null}
                {eventSourceCount ? (
                  <p className="text-[11px] text-slate-500">事件来源映射 {eventSourceCount} 条</p>
                ) : null}
              </div>
            ) : null}
            <div className="rounded-lg border border-white/10 bg-white/5 p-3 space-y-2">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">原始数据</p>
              {componentDetails.length ? (
                <div className="space-y-1 text-xs text-slate-300">
                  {componentDetails.map((item) => (
                    <div key={item.key} className="flex items-start justify-between gap-3">
                      <span className="min-w-[120px] text-slate-400">{item.key}</span>
                      <span className="font-mono text-slate-100 break-all text-right">{item.value}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-400">暂无结构化数据。</p>
              )}
            </div>
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">来源</p>
              <div className="max-h-32 overflow-y-auto space-y-2 pr-1">
                {data?.sources?.length ? (
                  data.sources.map((src) => (
                    <div key={src.url ?? src.title} className="rounded-lg border border-white/10 bg-white/5 p-2">
                      <p className="text-slate-100">{src.title}</p>
                      <p className="text-xs text-slate-400">{src.source}</p>
                      {src.url ? (
                        <a className="text-xs text-violet-200" href={src.url} target="_blank" rel="noreferrer">
                          打开原文
                        </a>
                      ) : null}
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-400">暂无引用新闻源。</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
