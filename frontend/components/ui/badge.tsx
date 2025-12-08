import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-medium uppercase tracking-[0.08em] transition-colors',
  {
    variants: {
      variant: {
        default: 'border-white/20 bg-white/5 text-slate-100',
        accent: 'border-violet-400/50 bg-violet-500/15 text-violet-100',
        outline: 'border-white/20 text-slate-200',
        success: 'border-emerald-400/40 bg-emerald-500/10 text-emerald-200',
        warning: 'border-amber-400/40 bg-amber-500/10 text-amber-100',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant, className }))} {...props} />;
}

export { Badge, badgeVariants };
