from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import ContextStore
from app.services.llm import build_small_llm
from app.services.providers import FMPProvider, YFinanceProvider
from app.services.providers.base import ProviderError

settings = get_settings()
CONTEXT_TTL_DAYS = 7


def _strip_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(cleaned.split())


def _normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    title = article.get("title") or ""
    summary = article.get("summary") or ""
    text = _strip_text(f"{title} {summary}")
    return {
        "title": title,
        "text": text,
        "url": article.get("url"),
        "source": article.get("source"),
        "published_at": article.get("published_at"),
    }


async def _summarize_news(snippets: List[str]) -> List[str]:
    if not snippets:
        return []

    try:
        llm = build_small_llm(temperature=0.2)
    except ValueError:
        return snippets[:3]

    system_prompt = (
        "你是一名简报助理，请用中文将提供的新闻片段压缩为 3-5 条摘要。只使用提供的文本，不要外推。"
    )
    news_block = "\n".join(f"- ({idx}) {snippet}" for idx, snippet in enumerate(snippets))
    human_prompt = (
        "输出要点数组，每条 <= 25 字，并尽量携带索引引用，例如：(0)(2)。\n"
        f"片段列表：\n{news_block}"
    )

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    lines = [line.strip(" -•") for line in response.content.splitlines() if line.strip()]
    return [line for line in lines if line][:5] or snippets[:3]


async def _get_cached_context(ticker: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    cutoff = date.today() - timedelta(days=CONTEXT_TTL_DAYS)
    stmt = (
        select(ContextStore)
        .where(
            and_(
                ContextStore.ticker == ticker,
                ContextStore.type == "news",
                ContextStore.asof_date >= cutoff,
            )
        )
        .order_by(ContextStore.asof_date.desc())
    )
    result = await db.execute(stmt)
    row = result.scalars().first()
    if row:
        return row.summary
    return None


async def build_context(
    ticker: str,
    *,
    db: AsyncSession,
    lookback_days: int = 30,
    max_news: int = 8,
) -> Dict[str, Any]:
    cached = await _get_cached_context(ticker, db)
    if cached:
        return cached

    start_date = date.today() - timedelta(days=lookback_days)
    provider = YFinanceProvider(proxy=settings.yfinance_proxy)
    articles: List[Dict[str, Any]] = await provider.fetch_news(ticker, limit=max_news * 2, start=start_date)

    if not articles:
        try:
            fmp = FMPProvider()
            articles = await fmp.fetch_news(ticker, limit=max_news * 2, start=start_date)
        except ProviderError:
            articles = []

    normalized = [_normalize_article(article) for article in articles][:max_news]
    snippets = [item["text"] for item in normalized if item.get("text")]
    summaries = await _summarize_news(snippets)

    payload = {
        "ticker": ticker,
        "asof_date": date.today().isoformat(),
        "news": {
            "summaries": summaries,
            "snippets": snippets,
        },
        "sources": normalized,
    }

    entry = ContextStore(
        ticker=ticker,
        asof_date=date.today(),
        type="news",
        summary=payload,
        source_refs=normalized,
    )
    db.add(entry)
    await db.commit()

    return payload
