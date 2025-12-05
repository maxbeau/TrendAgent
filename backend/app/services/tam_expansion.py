"""TAM expansion qualitative scorer using Massive news + LLM."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.services.providers import MassiveProvider
from app.services.providers.base import ProviderError


settings = get_settings()


def _get_llm() -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is missing; set OPENAI_API_KEY in the environment.")
    return ChatOpenAI(
        model=settings.openai_model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.2,
    )


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
                "outlook": data.get("outlook", "unknown"),
                "score": data.get("score", 0),
                "confidence": data.get("confidence", 0),
                "summary": data.get("summary") or data.get("rationale") or text,
                "key_points": data.get("key_points") or data.get("bullets") or [],
            }
    except json.JSONDecodeError:
        pass

    return {
        "outlook": "unknown",
        "score": 0,
        "confidence": 0,
        "summary": content,
        "key_points": [],
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


async def summarize_tam_expansion(
    ticker: str,
    *,
    limit: int = 12,
    lookback_days: int = 90,
    massive: Optional[MassiveProvider] = None,
) -> Dict[str, Any]:
    """
    Use Massive news to judge TAM expansion sentiment (expanding/stable/contracting).
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
            "outlook": "unknown",
            "score": 0,
            "confidence": 0,
            "summary": f"No Massive news found since {published_after}.",
            "key_points": [],
            "sources": [],
            "generated_at": datetime.utcnow(),
        }

    system_prompt = (
        "You are a market sizing analyst. Determine whether the company's TAM is expanding, stable, or contracting "
        "based on recent news about product launches, regulatory changes, partnerships, and market adoption."
    )
    human_prompt = (
        "Return a JSON object with keys: outlook (expanding|stable|contracting), score (1-5), confidence (1-5), "
        "summary (2 sentences), key_points (array of <=3 bullets). "
        f"Ticker: {ticker}\nRecent news headlines:\n{_build_news_block(articles)}"
    )

    llm = _get_llm()
    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    parsed = _parse_llm_content(response.content)

    return {
        "ticker": ticker,
        "outlook": parsed["outlook"],
        "score": parsed["score"],
        "confidence": parsed["confidence"],
        "summary": parsed["summary"],
        "key_points": parsed["key_points"],
        "sources": _news_sources(articles),
        "generated_at": datetime.utcnow(),
    }
