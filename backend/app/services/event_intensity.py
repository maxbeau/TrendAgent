"""Event intensity scorer using Massive news + LLM."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm import build_llm
from app.services.providers import MassiveProvider
from app.services.providers.base import ProviderError


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    return text


def _parse_llm_content(content: str) -> Dict[str, Any]:
    text = _strip_code_fence(content)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "intensity": data.get("intensity", "unknown"),
                "score": data.get("score", 0),
                "confidence": data.get("confidence", 0),
                "summary": data.get("summary") or data.get("rationale") or text,
                "key_events": data.get("key_events") or data.get("key_points") or [],
            }
    except json.JSONDecodeError:
        pass

    return {
        "intensity": "unknown",
        "score": 0,
        "confidence": 0,
        "summary": content,
        "key_events": [],
    }


def _format_article(article: Dict[str, Any]) -> str:
    published = article.get("published_utc") or article.get("published_at") or ""
    title = article.get("title") or article.get("headline") or ""
    return f"- [{published}] {title}".strip()


def _build_news_block(articles: List[Dict[str, Any]]) -> str:
    lines = [_format_article(article) for article in articles if article]
    return "\n".join(lines)


def _news_sources(articles: List[Dict[str, Any]]) -> List[Dict[str, Optional[str]]]:
    sources: List[Dict[str, Optional[str]]] = []
    for article in articles:
        sources.append(
            {
                "title": article.get("title") or article.get("headline"),
                "published_utc": article.get("published_utc") or article.get("published_at"),
                "url": article.get("article_url") or article.get("url"),
            }
        )
    return sources


async def summarize_event_intensity(
    ticker: str,
    *,
    limit: int = 12,
    lookback_days: int = 30,
    massive: Optional[MassiveProvider] = None,
) -> Dict[str, Any]:
    """
    Fetch recent Massive news and classify event intensity for the ticker.
    """
    published_after = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()
    provider = massive or MassiveProvider()

    try:
        articles = await provider.fetch_news(ticker, limit=limit, published_utc=published_after)
    except ProviderError as exc:  # pragma: no cover - network
        raise ProviderError(f"Failed to fetch news from Massive: {exc}") from exc

    if not articles:
        return {
            "ticker": ticker,
            "intensity": "unknown",
            "score": 0,
            "confidence": 0,
            "summary": f"No Massive news found since {published_after}.",
            "key_events": [],
            "sources": [],
            "generated_at": datetime.utcnow(),
        }

    system_prompt = (
        "You are an event-driven analyst. Determine whether near-term news flow represents a HIGH, MEDIUM, "
        "or LOW event intensity for the ticker. Consider materiality (M&A, large contracts, regulation), "
        "time proximity, and relevance to the company."
    )
    human_prompt = (
        "Return a JSON object with keys: intensity (high|medium|low), score (1-5), confidence (1-5), "
        "summary (2 sentences), key_events (array of <=3 bullets). "
        f"Ticker: {ticker}\nRecent news headlines:\n{_build_news_block(articles)}"
    )

    llm = build_llm(temperature=0.2)
    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    parsed = _parse_llm_content(response.content)

    return {
        "ticker": ticker,
        "intensity": parsed["intensity"],
        "score": parsed["score"],
        "confidence": parsed["confidence"],
        "summary": parsed["summary"],
        "key_events": parsed["key_events"],
        "sources": _news_sources(articles),
        "generated_at": datetime.utcnow(),
    }
