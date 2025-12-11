from collections import deque
import logging
from collections import deque
from datetime import date, datetime, timedelta
from statistics import pstdev
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import MarketDataDaily
from app.services.providers import YFinanceProvider
from app.services.providers.base import ProviderError

router = APIRouter(prefix="/market", tags=["market"])
yfinance_provider = YFinanceProvider()
logger = logging.getLogger(__name__)


def _empty_payload(ticker: str, source: str, reason: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "candles": [],
        "ma20": [],
        "ma50": [],
        "ma200": [],
        "bands": [],
        "source": source,
        "status": "unavailable",
        "message": reason,
        "data_points": 0,
    }


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
        logger.info("ohlc_fallback_yfinance_start", extra={"ticker": ticker, "start": start.isoformat(), "limit": limit})
        try:
            fetched = await yfinance_provider.fetch_equity_daily(ticker, start=start, end=None)
        except ProviderError as exc:  # pragma: no cover - 网络/第三方异常
            message = str(exc)
            logger.warning("ohlc_fallback_yfinance_failed", extra={"ticker": ticker, "error": message})
            return _empty_payload(ticker, "yfinance_unavailable", message)
        except Exception as exc:  # pragma: no cover - 网络/第三方异常
            logger.exception("ohlc_fallback_yfinance_failed", extra={"ticker": ticker, "error": str(exc)})
            return _empty_payload(ticker, "yfinance_unavailable", str(exc))
        rows = fetched[-limit:]
        source = "yfinance"
        logger.info("ohlc_fallback_yfinance_success", extra={"ticker": ticker, "rows": len(rows)})

    if not rows:
        logger.warning("ohlc_empty_rows", extra={"ticker": ticker, "source": source})
        return _empty_payload(ticker, source, "No OHLC data found")

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

    payload = {
        "ticker": ticker,
        "candles": candles,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "bands": bands,
        "source": source,
        "status": "ok",
        "data_points": len(candles),
    }
    logger.info("ohlc_response", extra={"ticker": ticker, "source": source, "candles": len(candles)})
    return payload
