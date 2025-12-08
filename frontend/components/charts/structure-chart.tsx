'use client';

import { useEffect, useMemo, useRef } from 'react';
import type { IChartApi, UTCTimestamp } from 'lightweight-charts';

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

  useEffect(() => {
    if (!containerRef.current || normCandles.length === 0) return;

    let destroyed = false;
    let resizeObserver: ResizeObserver | null = null;

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

        chart.timeScale().fitContent();

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
      chartRef.current?.remove();
      chartRef.current = null;
    };
  }, [normCandles, normMA20, normMA50, normMA200, normBands]);

  return (
    <div className="relative h-[340px] w-full">
      {normCandles.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center rounded-lg border border-white/10 bg-white/5 text-sm text-slate-300">
          暂无可展示的行情数据
        </div>
      )}
      <div ref={containerRef} className="h-full w-full" />
    </div>
  );
}
