import axios from 'axios';
import { useEffect, useState } from 'react';

interface DashboardSummary {
  model: string;
  tickers: string[];
  highlights: string[];
}

const placeholder: DashboardSummary = {
  model: 'AION v2.3',
  tickers: ['NVDA', 'AAPL', 'MSFT'],
  highlights: [
    'Macro momentum improving',
    'Fundamental strength in AI leaders',
    'Volatility remains elevated around catalysts',
  ],
};

export default function Home() {
  const [summary, setSummary] = useState<DashboardSummary>(placeholder);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api';
    axios
      .get<DashboardSummary>(`${apiBase}/dashboard/summary`)
      .then((resp) => setSummary(resp.data))
      .catch(() => setError('Unable to reach backend dashboard summary.'));
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <section className="mx-auto max-w-4xl p-6">
        <header className="pb-4">
          <p className="text-sm uppercase tracking-[0.3em] text-slate-400">TrendAgent</p>
          <h1 className="text-4xl font-semibold tracking-tight">AION Dashboard</h1>
          <p className="mt-1 text-slate-400">Model {summary.model} · {summary.tickers.join(' / ')}</p>
        </header>

        {error ? (
          <p className="rounded-md border border-rose-500 bg-rose-900/70 p-3 text-sm text-rose-200">
            {error}
          </p>
        ) : null}

        <section className="mt-6 grid gap-6 md:grid-cols-2">
          <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg">
            <h2 className="text-lg font-semibold text-slate-200">Highlights</h2>
            <ul className="mt-3 space-y-2 text-sm text-slate-200">
              {summary.highlights.map((item) => (
                <li key={item} className="rounded-md bg-slate-800/50 p-3">
                  {item}
                </li>
              ))}
            </ul>
          </article>

          <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg">
            <h2 className="text-lg font-semibold text-slate-200">Watchlist</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {summary.tickers.map((ticker) => (
                <span
                  key={ticker}
                  className="rounded-full border border-slate-700 bg-slate-800 px-4 py-1 text-xs tracking-widest text-slate-300"
                >
                  {ticker}
                </span>
              ))}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
