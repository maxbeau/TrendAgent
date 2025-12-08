'use client';

import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

interface RadarPoint {
  factor: string;
  score: number;
}

interface AionRadarProps {
  data: RadarPoint[];
}

export function AionRadar({ data }: AionRadarProps) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <RadarChart data={data} outerRadius="75%">
        <PolarGrid gridType="circle" stroke="rgba(255,255,255,0.08)" radialLines={false} />
        <PolarAngleAxis dataKey="factor" tick={{ fill: '#cbd5e1', fontSize: 12 }} tickLine={false} axisLine={false} />
        <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 5]} />
        <Tooltip
          contentStyle={{ background: 'rgba(24,24,27,0.9)', border: '1px solid rgba(255,255,255,0.08)' }}
          cursor={{ stroke: '#a78bfa', strokeDasharray: '4 4' }}
          formatter={(value: number) => value.toFixed(2)}
        />
        <Radar
          name="AION"
          dataKey="score"
          stroke="#a78bfa"
          fill="#a78bfa"
          fillOpacity={0.25}
          strokeWidth={2}
          dot={{ r: 3, fill: '#a78bfa', stroke: '#a78bfa' }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
