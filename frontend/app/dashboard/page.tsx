import { Suspense } from 'react';

import DashboardClientPage from './dashboard-client-page';

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-obsidian-950" />}>
      <DashboardClientPage />
    </Suspense>
  );
}
