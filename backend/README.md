# Backend (FastAPI) Service

## 运行说明

1. 使用 Poetry 管理环境

   ```bash
   cd backend
   poetry install
   ```

   如果希望进入 Poetry 虚拟环境：

   ```bash
   poetry shell
   ```

2. 设置环境变量

   * 复制 `.env.example` 为 `.env`，设置 `DATABASE_URL` 为 Supabase/Postgres 的 `postgresql+asyncpg://` 链接。
   * 可选填充 `SUPABASE_URL` 与 `SUPABASE_KEY`，后续用于调用 supabase-py SDK。

3. 启动开发服务（Poetry + Uvicorn）

   ```bash
   poetry run uvicorn app.main:application --reload --reload-dir backend
   ```

   或者直接使用 Poetry 定义的脚本：

   ```bash
   poetry run trendagent-backend
   ```

4. 常用端点

   * `GET /health`
   * `GET /dashboard/summary`
   * `POST /engine/calculate`
   * `POST /narrative/generate`
   * `GET /narrative/status/{task_id}`
   * `POST /engine/policy-tailwind`（基于 Massive 新闻 + LLM 的政策顺风定性总结）
   * `POST /engine/event-intensity`（基于 Massive 新闻标题 + LLM 的事件强度评估）
   * `POST /engine/tam-expansion`（基于 Massive 新闻 + LLM 的 TAM 扩张定性判定）
   * `POST /engine/risk-reward`（基于近期高低点的 R/R 估算）

5. 数据访问

   * `app/db.py` 中定义 `AsyncSessionLocal` 与 `get_db` 依赖，可在路由或服务中注入用于访问 Supabase/Postgres。
   * 如果未来需要 supabase-py 直接访问，可在 `app/services` 添加 Supabase 客户端工厂。

## 结构速览

* `app/core/app.py`：创建 FastAPI 应用，与所有路由和中间件绑定。
* `app/api/`：划分业务路由（仪表盘/引擎/叙事/健康检查）。
* `app/services/`：承载核心引擎与数据处理逻辑，供接口层调用。
* `app/db.py`：异步 SQLAlchemy 引擎和 `AsyncSession` 依赖工厂。
