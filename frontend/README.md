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

* `pages/index.tsx`：展示仪表盘摘要。
* `styles/globals.css`：全局简易配色。
* `next.config.js`：保留默认配置，可根据需求扩展代理/优化。
