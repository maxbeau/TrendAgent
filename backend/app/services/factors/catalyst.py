from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.factors.core import _factor_scores_from, _sanitize_weights, _soft_result, FactorResult
from app.services.providers import MassiveProvider, YFinanceProvider
from app.services.providers.base import ProviderError
from app.services.soft_factors import fetch_soft_factor_score, score_soft_factor


async def compute_soft_factor(
    ticker: str,
    factor_code: str,
    db: Optional[AsyncSession],
    massive: Optional[MassiveProvider],
    yfinance: Optional[YFinanceProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    """
    Load latest soft factor score (e.g., F2 Industry, F7 Catalyst) from DB cache.
    If not available, attempt to generate using news context; otherwise return placeholder.
    """
    applied_weights, weight_denominator = _sanitize_weights(factor_weights)

    if db is None:
        return _soft_result(
            None,
            status="unavailable",
            summary="DB session required to load soft factor scores",
            key_evidence=[],
            sources=[],
            asof_date=None,
            confidence=None,
            errors=["DB session required to load soft factor scores"],
            weights=applied_weights,
            weight_denominator=weight_denominator,
            factor_scores=_factor_scores_from(None, applied_weights),
        )

    entry = await fetch_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        db=db,
        latest=True,
    )

    if not entry:
        texts: List[str] = []
        sources: List[Dict[str, Any]] = []
        news_articles: List[Dict[str, Any]] = []
        calendar_context: Dict[str, Any] = {}
        if massive:
            try:
                since = (date.today() - timedelta(days=30)).isoformat()
                news = await massive.fetch_news(ticker, limit=20, published_utc=since)
                news_articles = news or []
                for item in news_articles:
                    title = item.get("title") or ""
                    summary = item.get("description") or item.get("summary") or ""
                    combined = f"{title}. {summary}".strip(". ")
                    if combined:
                        texts.append(combined)
                    sources.append(
                        {
                            "title": title or summary or "",
                            "url": item.get("article_url") or item.get("url"),
                            "source": item.get("source"),
                            "published_utc": item.get("published_utc") or item.get("published_at"),
                        }
                    )
            except Exception as exc:  # pragma: no cover - network/third-party
                # soft factor is best-effort; log and continue
                pass

        if factor_code == "F7" and yfinance is not None:
            try:
                calendar_context = await yfinance.fetch_earnings_calendar(ticker)
            except ProviderError:
                calendar_context = {}
            except Exception:
                calendar_context = {}

        scoring_context: Optional[Dict[str, Any]] = None
        if factor_code == "F7":
            scoring_context = {
                "news_articles": news_articles,
                "sources": sources,
                "calendar": calendar_context,
            }

        should_attempt = bool(texts)
        if factor_code == "F7" and (calendar_context or news_articles):
            should_attempt = True

        if should_attempt:
            try:
                scored = await score_soft_factor(
                    ticker,
                    factor_code,
                    texts,
                    db=db,
                    context=scoring_context,
                )
                factor_scores_payload = scored.get("factor_scores")
                return _soft_result(
                    scored.get("score"),
                    status="ok" if scored.get("score") is not None else "static",
                    summary=scored.get("summary"),
                    key_evidence=scored.get("key_evidence", []),
                    sources=scored.get("sources", []) or sources,
                    asof_date=scored.get("asof_date"),
                    confidence=scored.get("confidence"),
                    errors=[],
                    weights=applied_weights,
                    weight_denominator=weight_denominator,
                    factor_scores=factor_scores_payload or _factor_scores_from(scored.get("score"), applied_weights),
                    extra_components=scored.get("components"),
                )
            except Exception:  # pragma: no cover - LLM/DB errors
                pass

        return _soft_result(
            None,
            status="static",
            summary="No soft factor score found",
            key_evidence=[],
            sources=sources,
            asof_date=None,
            confidence=None,
            errors=["No soft factor score found; run soft factor scoring pipeline"],
            weights=applied_weights,
            weight_denominator=weight_denominator,
            factor_scores=_factor_scores_from(None, applied_weights),
        )

    stored_factor_scores = None
    stored_components = None
    if isinstance(entry.reasons, dict):
        stored_factor_scores = entry.reasons.get("factor_scores")
        stored_components = entry.reasons.get("components")

    return _soft_result(
        entry.score,
        status="ok" if entry.score is not None else "static",
        summary=entry.reasons,
        key_evidence=entry.citations,
        sources=[],
        asof_date=entry.asof_date.isoformat(),
        confidence=entry.confidence,
        errors=["Soft factor score is null; check ingest/LLM pipeline"] if entry.score is None else [],
        weights=applied_weights,
        weight_denominator=weight_denominator,
        factor_scores=stored_factor_scores or _factor_scores_from(entry.score, applied_weights),
        extra_components=stored_components,
    )
