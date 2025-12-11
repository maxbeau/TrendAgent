import { FormEvent, useEffect, useState } from 'react';

type DashboardNavProps = {
  modelLabel: string;
  initialTicker: string;
  onSubmit: (ticker: string) => void;
};

export function DashboardNav({ modelLabel, initialTicker, onSubmit }: DashboardNavProps) {
  const [tickerInput, setTickerInput] = useState(initialTicker);

  useEffect(() => {
    setTickerInput(initialTicker);
  }, [initialTicker]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit(tickerInput);
  };

  return (
    <nav className="sticky top-0 z-20 border-b border-white/5 bg-obsidian-950/80 backdrop-blur">
      <div className="page-shell flex flex-col gap-3 py-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-400">{modelLabel}</span>
        </div>
        <form
          onSubmit={handleSubmit}
          className="flex w-full flex-col gap-2 md:w-auto md:flex-row md:items-center md:gap-3"
        >
          <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 md:min-w-[320px]">
            <span className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Ticker</span>
            <input
              value={tickerInput}
              onChange={(event) => setTickerInput(event.target.value)}
              className="w-full bg-transparent text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none"
              placeholder="如：NVDA / AAPL / TSLA"
              aria-label="输入要搜索的 ticker"
            />
          </div>
          <button
            type="submit"
            className="inline-flex items-center justify-center rounded-xl bg-violet-500 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-violet-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-violet-300"
          >
            Analysis
          </button>
        </form>
      </div>
    </nav>
  );
}
