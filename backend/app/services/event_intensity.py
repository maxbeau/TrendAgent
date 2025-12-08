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
            raw_events = data.get("key_events") or []
            parsed_events: List[str] = []
            event_metadata: List[Dict[str, Any]] = []
            if isinstance(raw_events, list):
                for entry in raw_events:
                    if isinstance(entry, dict):
                        event_type = str(entry.get("type") or entry.get("category") or "event").lower()
                        description = entry.get("description") or entry.get("summary") or ""
                        date_value = entry.get("event_date") or entry.get("date")
                        if date_value:
                            event_metadata.append(
                                {
                                    "type": event_type,
                                    "description": description,
                                    "date": date_value,
                                }
                            )
                        label = event_type.upper()
                        parts = [f"[{label}]"]
                        if date_value:
                            parts.append(str(date_value))
                        if description:
                            parts.append(str(description))
                        parsed_events.append(" ".join(part for part in parts if part).strip())
                    else:
                        parsed_events.append(str(entry))
            elif raw_events:
                parsed_events.append(str(raw_events))
            return {
                "intensity": data.get("intensity", "unknown"),
                "score": data.get("score", 0),
                "confidence": data.get("confidence", 0),
                "summary": data.get("summary") or data.get("rationale") or text,
                "key_events": parsed_events or data.get("key_points") or [],
                "event_metadata": event_metadata,
            }
    except json.JSONDecodeError:
        pass

    return {
        "intensity": "unknown",
        "score": 0,
        "confidence": 0,
        "summary": content,
        "key_events": [],
        "event_metadata": [],
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
                "source": article.get("source"),
            }
        )
    return sources


async def summarize_event_intensity(
    ticker: str,
    *,
    limit: int = 12,
    lookback_days: int = 30,
    massive: Optional[MassiveProvider] = None,
    articles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Fetch recent Massive news and classify event intensity for the ticker.
    """
    published_after = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()
    provider = massive

    if articles is None:
        provider = provider or MassiveProvider()
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
        "or LOW event intensity for the ticker. Prioritize AION catalyst classes: earnings (beat/miss/guidance), "
        "product launches, partnerships/contracts, and policy/regulation shifts. When summarizing, call out which "
        "event type each key point belongs to. Consider materiality, time proximity, and relevance to the company."
    )
    human_prompt = (
        "Return a JSON object with keys: intensity (high|medium|low), score (1-5), confidence (1-5), "
        "summary (2 sentences), key_events (array of <=3 objects with fields type (earnings|product|partnership|policy|other), "
        "description, event_date in ISO format or 'N/A'). "
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
        "event_metadata": parsed.get("event_metadata", []),
        "sources": _news_sources(articles),
        "generated_at": datetime.utcnow(),
    }
