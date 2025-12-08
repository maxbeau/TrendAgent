#!/bin/bash
set -euo pipefail

# TrendAgent 开发环境一键启动脚本
# 请在项目根目录执行: ./start-dev.sh

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"

echo "🚀 正在启动 TrendAgent 后端服务..."
# 使用 subshell 在后台启动后端, 避免污染当前终端目录
(cd backend && uv run uvicorn app.main:application --reload --port 8000) &
BACKEND_PID=$!

cleanup() {
    echo "🛑 正在关闭后端服务 (PID: $BACKEND_PID)..."
    if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
        kill "$BACKEND_PID"
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
}

handle_exit() {
    cleanup
    exit "${1:-0}"
}

trap 'handle_exit 0' SIGINT SIGTERM

echo "⏱️  等待后端健康检查 (${HEALTH_URL})..."
for i in $(seq 1 "$HEALTH_TIMEOUT"); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
        echo "✅ 后端健康检查通过。"
        break
    fi
    if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
        echo "❌ 后端进程异常退出，终止脚本。"
        handle_exit 1
    fi
    sleep 1
    if [[ "$i" -eq "$HEALTH_TIMEOUT" ]]; then
        echo "❌ 后端健康检查超时 (${HEALTH_TIMEOUT}s)。"
        handle_exit 1
    fi
done

echo "🚀 正在启动 TrendAgent 前端服务..."
npm run dev --prefix frontend || {
    echo "❌ 前端启动失败。"
    handle_exit 1
}

handle_exit 0
