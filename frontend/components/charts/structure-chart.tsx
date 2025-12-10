'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { IChartApi, Logical, LogicalRange, UTCTimestamp } from 'lightweight-charts';

interface CandlePoint {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface OverlayPoint {
  time: string | number;
  value: number;
}

interface RangeBandPoint {
  time: string | number;
  upper: number;
  lower: number;
}

interface StructureChartProps {
  candles: CandlePoint[];
  ma20: OverlayPoint[];
  ma50: OverlayPoint[];
  ma200: OverlayPoint[];
  bands: RangeBandPoint[];
}

export function StructureChart({ candles, ma20, ma50, ma200, bands }: StructureChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const applyingRangeRef = useRef(false);
  const [rangeLabel, setRangeLabel] = useState<string>('最新区间');

  // Normalize time to UTCTimestamp (seconds)
  const normalizeTime = (t: string | number | undefined): UTCTimestamp | null => {
    if (typeof t === 'number' && Number.isFinite(t)) return Math.floor(t) as UTCTimestamp;
    if (!t) return null;
    const dateOnly = typeof t === 'string' ? t.split('T')[0] : t;
    const ts = Date.parse(String(dateOnly));
    return Number.isFinite(ts) ? (Math.floor(ts / 1000) as UTCTimestamp) : null;
  };

  const normalized = useMemo(() => {
    const sortByTime = <T extends { time: UTCTimestamp }>(a: T, b: T) => Number(a.time) - Number(b.time);

    const normalizeCandles = (
      series: StructureChartProps['candles'],
    ): Array<{ time: UTCTimestamp; open: number; high: number; low: number; close: number }> =>
      series
        .map((c) => {
          const time = normalizeTime(c.time);
          if (time === null) return null;
          return { ...c, time };
        })
        .filter(
          (c): c is { time: UTCTimestamp; open: number; high: number; low: number; close: number } => c !== null,
        )
        .sort(sortByTime);

    const normalizeOverlay = (series: OverlayPoint[]): Array<{ time: UTCTimestamp; value: number }> =>
      series
        .map((p) => {
          const time = normalizeTime(p.time);
          if (time === null || !Number.isFinite(p.value)) return null;
          return { time, value: p.value };
        })
        .filter((p): p is { time: UTCTimestamp; value: number } => p !== null)
        .sort(sortByTime);

    const normalizeBands = (
      series: RangeBandPoint[],
    ): Array<{ time: UTCTimestamp; upper: number; lower: number }> =>
      series
        .map((b) => {
          const time = normalizeTime(b.time);
          if (time === null || !Number.isFinite(b.upper) || !Number.isFinite(b.lower)) return null;
          return { time, upper: b.upper, lower: b.lower };
        })
        .filter((b): b is { time: UTCTimestamp; upper: number; lower: number } => b !== null)
        .sort(sortByTime);

    return {
      candles: normalizeCandles(candles),
      ma20: normalizeOverlay(ma20),
      ma50: normalizeOverlay(ma50),
      ma200: normalizeOverlay(ma200),
      bands: normalizeBands(bands),
    };
  }, [bands, candles, ma20, ma50, ma200]);

  const { candles: normCandles, ma20: normMA20, ma50: normMA50, ma200: normMA200, bands: normBands } = normalized;
  const lastIndex = normCandles.length - 1;
  const minBars = Math.min(30, Math.max(10, lastIndex + 1)); // 至少 10 根，默认 30 根或总数
  const maxBars = Math.min(520, Math.max(minBars, lastIndex + 1)); // 上限 520 根（约 2 年交易日）

  const clampRange = useCallback(
    (range: LogicalRange | null | undefined) => {
      if (!range || !Number.isFinite(range.from) || !Number.isFinite(range.to) || lastIndex < 0) return null;
      const width = range.to - range.from;
      let from = range.from;
      let to = range.to;

      if (width + 1 < minBars) {
        to = Math.max(to, minBars - 1) as Logical;
        from = (to - (minBars - 1)) as Logical;
      } else if (width + 1 > maxBars) {
        to = Math.min(to, maxBars - 1) as Logical;
        from = (to - (maxBars - 1)) as Logical;
      }

      if (from < 0) {
        to = (to - from) as Logical;
        from = 0 as Logical;
      }
      if (to > lastIndex) {
        const delta = to - lastIndex;
        from = (from - delta) as Logical;
        to = lastIndex as Logical;
        if (from < 0) from = 0 as Logical;
      }

      return { from, to };
    },
    [lastIndex, maxBars, minBars],
  );

  const applyRange = useCallback(
    (nextRange: LogicalRange | null) => {
      if (!chartRef.current || !nextRange) return;
      applyingRangeRef.current = true;
      chartRef.current.timeScale().setVisibleLogicalRange(nextRange);
      requestAnimationFrame(() => {
        applyingRangeRef.current = false;
      });
    },
    [],
  );

  const setRangeByCountFromTail = useCallback(
    (count: number) => {
      if (lastIndex < 0) return;
      const span = Math.min(Math.max(count, minBars), maxBars);
      const to = lastIndex as Logical;
      const from = Math.max(0, to - (span - 1)) as Logical;
      applyRange({ from, to });
    },
    [applyRange, lastIndex, maxBars, minBars],
  );

  const setRangeByDays = useCallback(
    (days: number) => {
      if (normCandles.length === 0) return;
      const latestTs = normCandles[lastIndex].time;
      const targetTs = latestTs - days * 24 * 60 * 60;
      const fromIdx = Math.max(
        0,
        normCandles.findIndex((c) => c.time >= targetTs),
      );
      const to = lastIndex as Logical;
      const from = Math.max(0, Math.min(fromIdx, to)) as Logical;
      const span = to - from + 1;
      const clampedSpan = Math.min(Math.max(span, minBars), maxBars);
      const adjustedFrom = Math.max(0, to - (clampedSpan - 1)) as Logical;
      applyRange({ from: adjustedFrom, to });
    },
    [applyRange, lastIndex, maxBars, minBars, normCandles],
  );

  const setRangeYtd = useCallback(() => {
    if (normCandles.length === 0) return;
    const latest = normCandles[lastIndex].time * 1000;
    const d = new Date(latest);
    const start = Date.UTC(d.getUTCFullYear(), 0, 1) / 1000;
    const fromIdx = Math.max(
      0,
      normCandles.findIndex((c) => c.time >= start),
    );
    const to = lastIndex as Logical;
    const from = Math.max(0, fromIdx) as Logical;
    applyRange(clampRange({ from, to }));
  }, [applyRange, clampRange, lastIndex, normCandles]);

  useEffect(() => {
    if (!containerRef.current || normCandles.length === 0) return;

    let destroyed = false;
    let resizeObserver: ResizeObserver | null = null;
    let handleRangeChange: ((range: LogicalRange | null) => void) | null = null;
    let timeScale: ReturnType<IChartApi['timeScale']> | null = null;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    (async () => {
      try {
        const { createChart, CandlestickSeries, LineSeries, AreaSeries, LineStyle } = await import('lightweight-charts');

        if (destroyed || !containerRef.current) return;

        const chart = createChart(containerRef.current!, {
          layout: {
            background: { color: 'transparent' },
            textColor: '#cbd5e1',
          },
          grid: {
            vertLines: { color: 'rgba(255,255,255,0.04)' },
            horzLines: { color: 'rgba(255,255,255,0.04)' },
          },
          width: containerRef.current!.clientWidth,
          height: 340,
          timeScale: {
            borderColor: 'rgba(255,255,255,0.08)',
          },
          rightPriceScale: {
            borderColor: 'rgba(255,255,255,0.08)',
          },
          crosshair: {
            mode: 0,
          },
        });

        chartRef.current = chart;

        // Add candlestick series
        const candleSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#34d399',
          borderUpColor: '#34d399',
          wickUpColor: '#34d399',
          downColor: '#f43f5e',
          borderDownColor: '#f43f5e',
          wickDownColor: '#f43f5e',
        });
        candleSeries.setData(normCandles);

        // Add moving averages
        const ma20Series = chart.addSeries(LineSeries, { color: '#38bdf8', lineWidth: 2 });
        ma20Series.setData(normMA20);

        const ma50Series = chart.addSeries(LineSeries, { color: '#a78bfa', lineWidth: 2, lineStyle: LineStyle.Dotted });
        ma50Series.setData(normMA50);

        const ma200Series = chart.addSeries(LineSeries, { color: '#94a3b8', lineWidth: 2, lineStyle: LineStyle.Dashed });
        ma200Series.setData(normMA200);

        // Add Bollinger bands
        const upperBandSeries = chart.addSeries(AreaSeries, {
          topColor: 'rgba(147, 197, 253, 0.18)',
          bottomColor: 'rgba(147, 197, 253, 0.01)',
          lineColor: 'rgba(147, 197, 253, 0.5)',
          lineWidth: 1,
        });
        upperBandSeries.setData(normBands.map((b) => ({ time: b.time, value: b.upper })));

        const lowerBandSeries = chart.addSeries(AreaSeries, {
          topColor: 'rgba(244, 63, 94, 0.18)',
          bottomColor: 'rgba(244, 63, 94, 0.01)',
          lineColor: 'rgba(244, 63, 94, 0.5)',
          lineWidth: 1,
        });
        lowerBandSeries.setData(normBands.map((b) => ({ time: b.time, value: b.lower })));

        timeScale = chart.timeScale();

        const updateLabel = (range: LogicalRange | null | undefined) => {
          if (!range) return;
          const fromIdx = Math.max(0, Math.floor(range.from));
          const toIdx = Math.min(lastIndex, Math.ceil(range.to));
          const startTs = normCandles[fromIdx]?.time;
          const endTs = normCandles[toIdx]?.time;
          if (startTs && endTs) {
            const start = new Date(startTs * 1000);
            const end = new Date(endTs * 1000);
            const fmt = (d: Date) => `${d.getUTCFullYear()}/${String(d.getUTCMonth() + 1).padStart(2, '0')}/${String(d.getUTCDate()).padStart(2, '0')}`;
            setRangeLabel(`${fmt(start)} - ${fmt(end)}`);
          }
        };

        const initialSpan = Math.min(60, maxBars);
        const initialFrom = Math.max(0, lastIndex - (initialSpan - 1)) as Logical;
        const initialRange = clampRange({ from: initialFrom, to: lastIndex as Logical }) ?? undefined;
        if (initialRange) {
          applyRange(initialRange);
          updateLabel(initialRange);
        } else {
          timeScale.fitContent();
        }

        handleRangeChange = (visibleRange: LogicalRange | null) => {
          if (applyingRangeRef.current) return;
          const clamped = clampRange(visibleRange);
          if (!clamped) return;
          // 防止过缩/过移，必要时回调
          if (
            !visibleRange ||
            clamped.from !== visibleRange.from ||
            clamped.to !== visibleRange.to
          ) {
            applyRange(clamped);
            return;
          }
          updateLabel(clamped);
        };

        timeScale.subscribeVisibleLogicalRangeChange(handleRangeChange);

        // Handle resize
        resizeObserver = new ResizeObserver(() => {
          if (containerRef.current && chart) {
            chart.applyOptions({ width: containerRef.current.clientWidth });
          }
        });
        resizeObserver.observe(containerRef.current);

      } catch (err) {
        console.error('Failed to create chart:', err);
      }
    })();

    return () => {
      destroyed = true; // 防止异步导入返回后重复挂载导致出现多个图表
      resizeObserver?.disconnect();
      if (handleRangeChange) {
        timeScale?.unsubscribeVisibleLogicalRangeChange?.(handleRangeChange);
      }
      chartRef.current?.remove();
      applyingRangeRef.current = false;
      chartRef.current = null;
    };
  }, [
    applyRange,
    clampRange,
    lastIndex,
    maxBars,
    minBars,
    normBands,
    normCandles,
    normMA20,
    normMA50,
    normMA200,
  ]);

  const handleReset = () => setRangeByCountFromTail(60);
  const handle1M = () => setRangeByDays(22);
  const handle3M = () => setRangeByDays(66);
  const handle6M = () => setRangeByDays(132);
  const handleYtd = () => setRangeYtd();
  const handleMax = () => applyRange(clampRange({ from: 0 as Logical, to: lastIndex as Logical }));

  return (
    <div className="relative h-[380px] w-full space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 hover:border-violet-300/40 hover:text-slate-100"
          >
            回到最新
          </button>
          <button
            type="button"
            onClick={handle1M}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 hover:border-violet-300/40 hover:text-slate-100"
          >
            1M
          </button>
          <button
            type="button"
            onClick={handle3M}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 hover:border-violet-300/40 hover:text-slate-100"
          >
            3M
          </button>
          <button
            type="button"
            onClick={handle6M}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 hover:border-violet-300/40 hover:text-slate-100"
          >
            6M
          </button>
          <button
            type="button"
            onClick={handleYtd}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 hover:border-violet-300/40 hover:text-slate-100"
          >
            YTD
          </button>
          <button
            type="button"
            onClick={handleMax}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 hover:border-violet-300/40 hover:text-slate-100"
          >
            MAX
          </button>
        </div>
        <p className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 font-mono text-[11px] text-slate-300">
          {rangeLabel}
        </p>
      </div>
      {normCandles.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center rounded-lg border border-white/10 bg-white/5 text-sm text-slate-300">
          暂无可展示的行情数据
        </div>
      )}
      <div ref={containerRef} className="h-[340px] w-full" />
    </div>
  );
}
