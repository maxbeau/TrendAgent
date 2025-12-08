# Frontend (Next.js)

## 运行说明

1. 安装依赖

   ```bash
   npm install
   ```

2. 启动开发服务器

   ```bash
   npm run dev
   ```

3. 默认页面会尝试调用 `NEXT_PUBLIC_API_BASE_URL/dashboard/summary`，可在 `.env.local` 中设置对应值。

## 结构预览

* `app/layout.tsx`：全局布局与字体 (Inter/JetBrains Mono)。
* `app/globals.css`：Obsidian 主题、Tailwind base/utility。
* `app/dashboard/page.tsx`：仪表盘骨架页（Phase 1 Skeleton）。
* `components/ui/*`：shadcn 风格的 Card/Badge/Dialog/Skeleton 基建。
* `next.config.js`：保留默认配置，可根据需求扩展代理/优化。
