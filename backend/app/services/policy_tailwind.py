"""Policy tailwind qualitative analyzer using Massive news + LLM."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm import build_llm
from app.services.providers import MassiveProvider
from app.services.providers.base import ProviderError


def _format_article(article: Dict[str, Any]) -> str:
    published = article.get("published_utc") or article.get("published_at") or ""
    title = article.get("title") or article.get("headline") or ""
    desc = article.get("description") or article.get("summary") or ""
    tickers = article.get("tickers") or article.get("symbols") or []
    tickers_str = ", ".join(tickers) if tickers else ""
    return f"- [{published}] {title} ({tickers_str}) {desc}".strip()


def _build_news_block(articles: List[Dict[str, Any]]) -> str:
    lines = [_format_article(article) for article in articles if article]
    return "\n".join(lines)


def _parse_llm_content(content: str) -> Dict[str, Any]:
    # Allow models to wrap JSON in code fences.
    if content.strip().startswith("```"):
        content = content.strip().strip("`")
        if content.startswith("json"):
            content = content[4:].lstrip()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return {
                "verdict": data.get("verdict", "unknown"),
                "confidence": data.get("confidence", 0),
                "summary": data.get("summary") or data.get("rationale") or content,
                "key_points": data.get("key_points") or data.get("bullets") or [],
            }
    except json.JSONDecodeError:
        pass

    return {
        "verdict": "unknown",
        "confidence": 0,
        "summary": content,
        "key_points": [],
    }


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


async def summarize_policy_tailwind(
    ticker: str,
    *,
    limit: int = 12,
    lookback_days: int = 60,
    massive: Optional[MassiveProvider] = None,
) -> Dict[str, Any]:
    """
    Fetch recent Massive news for the ticker and ask the LLM to classify policy tailwind/headwind.
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
            "verdict": "unknown",
            "confidence": 0,
            "summary": f"No Massive news found since {published_after}.",
            "key_points": [],
            "sources": [],
            "generated_at": datetime.utcnow(),
        }

    system_prompt = (
        "You are a buy-side policy analyst. Determine whether the recent policy/regulatory backdrop "
        "creates a tailwind, headwind, or is neutral for the given ticker. Focus on government support, "
        "subsidies, regulation changes, export controls, and permits."
    )
    human_prompt = (
        "Return a short JSON object with keys: verdict (tailwind|headwind|neutral), "
        "confidence (1-5), summary (2 sentences), key_points (array of <=3 bullets). "
        f"Ticker: {ticker}\nRecent news:\n{_build_news_block(articles)}"
    )

    llm = build_llm(temperature=0.2)
    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    parsed = _parse_llm_content(response.content)

    return {
        "ticker": ticker,
        "verdict": parsed["verdict"],
        "confidence": parsed["confidence"],
        "summary": parsed["summary"],
        "key_points": parsed["key_points"],
        "sources": _news_sources(articles),
        "generated_at": datetime.utcnow(),
    }
