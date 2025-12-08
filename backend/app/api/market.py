from collections import deque
from datetime import date, datetime, timedelta
from statistics import pstdev
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db
from app.models import MarketDataDaily
from app.services.providers import YFinanceProvider

router = APIRouter(prefix="/market", tags=["market"])
settings = get_settings()
yfinance_provider = YFinanceProvider(proxy=settings.yfinance_proxy)


def _rolling_mean(values: List[float], window: int) -> List[float]:
    """Simple rolling average aligned to the latest point in the window."""
    if window <= 0:
        return []
    acc: deque[float] = deque(maxlen=window)
    means: List[float] = []
    for v in values:
        acc.append(v)
        if len(acc) == window:
            means.append(sum(acc) / window)
    return means


def _rolling_std(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return []
    acc: deque[float] = deque(maxlen=window)
    stds: List[float] = []
    for v in values:
        acc.append(v)
        if len(acc) == window:
            stds.append(pstdev(acc))
    return stds


@router.get("/ohlc")
async def get_ohlc(
    ticker: str = Query(..., min_length=1, max_length=10),
    limit: int = Query(200, ge=10, le=500),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    返回指定 ticker 的日线 OHLC 数据以及 MA20/50/200、1σ 波动带。
    """
    stmt = (
        select(MarketDataDaily)
        .where(MarketDataDaily.ticker == ticker)
        .order_by(desc(MarketDataDaily.date))
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = list(reversed(result.scalars().all()))
    source = "MarketDataDaily"

    # Fallback to live yfinance quotes if DB 中没有对应数据，便于本地开发。
    if not rows:
        start = date.today() - timedelta(days=max(limit, 200) * 2)  # 留冗余以覆盖交易日缺口
        try:
            fetched = await yfinance_provider.fetch_equity_daily(ticker, start=start, end=None)
        except Exception as exc:  # pragma: no cover - 网络/第三方异常
            raise HTTPException(status_code=502, detail=f"Failed to fetch OHLC for {ticker}: {exc}") from exc
        rows = fetched[-limit:]
        source = "yfinance"

    if not rows:
        raise HTTPException(status_code=404, detail=f"No OHLC data found for {ticker}")

    def _value(row: Any, key: str) -> Any:
        return row.get(key) if isinstance(row, dict) else getattr(row, key, None)

    candles = []
    for row in rows:
        raw_date = _value(row, "date")
        if not raw_date:
            continue
        time_iso = raw_date.date().isoformat() if isinstance(raw_date, datetime) else raw_date.isoformat()
        candles.append(
            {
                "time": time_iso,
                "open": _value(row, "open"),
                "high": _value(row, "high"),
                "low": _value(row, "low"),
                "close": _value(row, "close"),
            }
        )

    closes = [c["close"] for c in candles if c["close"] is not None]
    times = [c["time"] for c in candles if c["time"] is not None]

    def _align(series: List[float], win: int) -> List[Dict[str, float]]:
        offset = win - 1
        return [
            {"time": times[i + offset], "value": val}
            for i, val in enumerate(series)
            if i + offset < len(times)
        ]

    ma20 = _align(_rolling_mean(closes, 20), 20)
    ma50 = _align(_rolling_mean(closes, 50), 50)
    ma200 = _align(_rolling_mean(closes, 200), 200)

    std20 = _align(_rolling_std(closes, 20), 20)
    bands: List[Dict[str, float]] = []
    for std_entry in std20:
        time = std_entry["time"]
        close_idx = times.index(time)
        close_price = closes[close_idx]
        bands.append({"time": time, "upper": close_price + std_entry["value"], "lower": close_price - std_entry["value"]})

    return {
        "ticker": ticker,
        "candles": candles,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "bands": bands,
        "source": source,
    }
