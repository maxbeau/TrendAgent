"""Lazy-load ingest worker used by Phase 2 data foundation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.providers import FMPProvider, MassiveProvider, YFinanceProvider


def _to_date(value: date | str | int) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, int):
        return datetime.utcfromtimestamp(value / 1000).date()
    return date.fromisoformat(value)


def _normalize_row(row: Dict) -> Dict:
    if "t" in row:  # Massive agg schema
        parsed_date = _to_date(row["t"])
        return {
            "ticker": row.get("T") or row.get("ticker"),
            "date": parsed_date,
            "open": row.get("o"),
            "high": row.get("h"),
            "low": row.get("l"),
            "close": row.get("c"),
            "volume": row.get("v"),
        }
    parsed_date = _to_date(row["date"])
    return {
        "ticker": row.get("symbol") or row.get("ticker"),
        "date": parsed_date,
        "open": row.get("open"),
        "high": row.get("high"),
        "low": row.get("low"),
        "close": row.get("close"),
        "volume": row.get("volume"),
    }


class IngestWorker:
    def __init__(
        self,
        primary: MassiveProvider,
        fallback: FMPProvider | None = None,
        yfinance: YFinanceProvider | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.yfinance = yfinance

    async def ingest_equity_ohlcv(
        self,
        session: AsyncSession,
        ticker: str,
        start: date,
        end: date,
    ) -> Dict[str, int]:
        records = await self.primary.fetch_equity_aggregates(ticker, start, end)
        if not records and self.yfinance:
            records = await self.yfinance.fetch_equity_daily(ticker, start=start, end=end)
        if not records and self.fallback:
            records = await self.fallback.fetch_equity_daily(ticker, start=start, end=end)

        normalized = [_normalize_row(row) for row in records if row]
        upsert_rows: List[Dict] = [
            {
                "ticker": row["ticker"] or ticker,
                "date": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for row in normalized
            if row.get("open") is not None
        ]

        if not upsert_rows:
            return {"upserted": 0}

        sql = text(
            """
            INSERT INTO market_data_daily (ticker, date, open, high, low, close, volume, updated_at)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume, NOW())
            ON CONFLICT (ticker, date) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                updated_at = NOW()
            """
        )
        await session.execute(sql, upsert_rows)
        await session.commit()
        return {"upserted": len(upsert_rows)}
