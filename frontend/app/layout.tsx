import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });
const jetBrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' });

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
    <html lang="en" className={`${inter.variable} ${jetBrainsMono.variable}`}>
      <body className="min-h-screen bg-obsidian-950 font-sans text-slate-100 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
