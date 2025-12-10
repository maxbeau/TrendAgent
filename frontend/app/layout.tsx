import type { Metadata } from 'next';
import '@fontsource/inter/latin.css';
import '@fontsource/jetbrains-mono/latin.css';
import './globals.css';
import { Providers } from './providers';

export const metadata: Metadata = {
  title: 'TrendAgent Dashboard',
  description: 'AION 混合智能投资决策终端',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-obsidian-950 font-sans text-slate-100 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
