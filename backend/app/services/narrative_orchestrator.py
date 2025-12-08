from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AnalysisScore
from app.services.context_builder import build_context
from app.services.event_intensity import summarize_event_intensity
from app.services.llm import build_llm
from app.services.soft_factors import score_soft_factor
from app.services.tam_expansion import summarize_tam_expansion

logger = logging.getLogger(__name__)


class Scenario(BaseModel):
    title: str
    thesis: str
    drivers: List[str]
    risks: List[str]


class ActionCard(BaseModel):
    direction: str
    logic: str
    risks: List[str]
    execution_window: str


class NarrativeSchema(BaseModel):
    ticker: str
    base: Scenario
    bear: Scenario
    bull: Scenario
    action_card: ActionCard
    citations: List[int] = Field(default_factory=list)


async def _load_latest_analysis_score(
    ticker: str, db: AsyncSession, model_version: Optional[str] = None
) -> AnalysisScore:
    stmt = (
        select(AnalysisScore)
        .where(AnalysisScore.ticker == ticker)
        .order_by(AnalysisScore.created_at.desc())
    )
    if model_version:
        stmt = stmt.where(AnalysisScore.model_version == model_version)

    result = await db.execute(stmt)
    score = result.scalars().first()
    if score is None:
        raise ValueError(f"No analysis score found for {ticker}. Run /engine/calculate first.")
    return score


def _extract_hard_scores(factors: Dict[str, Any]) -> Dict[str, Optional[float]]:
    hard_scores: Dict[str, Optional[float]] = {}
    for code, payload in (factors or {}).items():
        hard_scores[code] = payload.get("score") if isinstance(payload, dict) else None
    return hard_scores


def _format_sources(label: str, sources: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for idx, src in enumerate(sources or []):
        title = src.get("title") or "N/A"
        published = src.get("published_utc") or src.get("published_at") or ""
        url = src.get("url") or ""
        prefix = f"{label}[{idx}]"
        parts = [part for part in (prefix, published, title, url) if part]
        lines.append(" | ".join(parts))
    return lines


def _normalize_confidence(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(val / 5.0, 1.0))


async def _compute_industry_score(
    ticker: str,
    *,
    news_snippets: List[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    try:
        tam = await summarize_tam_expansion(ticker)
        return {
            "factor_code": "F2",
            "score": tam.get("score"),
            "confidence": _normalize_confidence(tam.get("confidence")),
            "summary": tam.get("summary"),
            "key_evidence": tam.get("key_points", []),
            "outlook": tam.get("outlook"),
            "sources": tam.get("sources", []),
            "generated_at": tam.get("generated_at"),
        }
    except Exception as exc:  # pragma: no cover - network/LLM issues
        logger.warning("narrative_industry_soft_factor_fallback", extra={"ticker": ticker, "error": str(exc)})
        fallback = await score_soft_factor(ticker, "industry", news_snippets, db=db)
        fallback["sources"] = []
        return fallback


async def _compute_catalyst_score(
    ticker: str,
    *,
    news_snippets: List[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    try:
        events = await summarize_event_intensity(ticker)
        return {
            "factor_code": "F7",
            "score": events.get("score"),
            "confidence": _normalize_confidence(events.get("confidence")),
            "summary": events.get("summary"),
            "key_evidence": events.get("key_events", []),
            "intensity": events.get("intensity"),
            "sources": events.get("sources", []),
            "generated_at": events.get("generated_at"),
        }
    except Exception as exc:  # pragma: no cover - network/LLM issues
        logger.warning("narrative_catalyst_soft_factor_fallback", extra={"ticker": ticker, "error": str(exc)})
        fallback = await score_soft_factor(ticker, "catalyst", news_snippets, db=db)
        fallback["sources"] = []
        return fallback


async def _invoke_narrative_llm(payload: Dict[str, Any]) -> Dict[str, Any]:
    parser = JsonOutputParser(pydantic_object=NarrativeSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a sell-side strategist. Explain relationships across factors, produce base/bear/bull scenarios, "
                "and an action card. Only use provided data; if something is missing, set fields to 'N/A'. "
                "Avoid fabricating numbers or events. Respond in Chinese with concise bullets. "
                "Example of missing data handling: if macro score is null or N/A, explicitly note '宏观数据不足，暂不评估宏观影响' "
                "and proceed with available factors only. Apply the same pattern to any missing soft/hard factor.",
            ),
            (
                "human",
                "Ticker: {ticker}\nAION总分: {total_score}\n硬因子: {hard_scores}\n"
                "软因子: {soft_scores}\n上下文摘要: {context_summaries}\n新闻片段（索引:text）:\n{news_snippets}\n"
                "新闻来源（含时间/链接）：\n{news_sources}\n"
                "{format_instructions}",
            ),
        ]
    )
    news_lines = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(payload.get("news_snippets", [])))
    source_lines = "\n".join(payload.get("news_sources") or [])
    context_summaries = payload.get("context_summaries") or []
    chain = prompt | build_llm(temperature=0.3) | parser
    result = await chain.ainvoke(
        {
            "ticker": payload["ticker"],
            "total_score": payload.get("total_score"),
            "hard_scores": payload.get("hard_scores"),
            "soft_scores": payload.get("soft_scores"),
            "context_summaries": context_summaries,
            "news_snippets": news_lines,
            "news_sources": source_lines,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    if isinstance(result, dict):
        result = NarrativeSchema(**result)
    return result.model_dump()


async def generate_narrative(
    ticker: str,
    *,
    db: AsyncSession,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    score = await _load_latest_analysis_score(ticker, db, model_version=model_version)
    context = await build_context(ticker, db=db)

    news_snippets = context.get("news", {}).get("snippets", [])
    industry_score = await _compute_industry_score(ticker, news_snippets=news_snippets, db=db)
    catalyst_score = await _compute_catalyst_score(ticker, news_snippets=news_snippets, db=db)

    hard_scores = _extract_hard_scores(score.factors)
    soft_scores = {"industry": industry_score, "catalyst": catalyst_score}
    news_sources = _format_sources("Industry", industry_score.get("sources", [])) + _format_sources(
        "Catalyst", catalyst_score.get("sources", [])
    )
    if not news_sources:
        news_sources = _format_sources("Context", context.get("sources", []))

    payload = {
        "ticker": ticker,
        "total_score": score.total_score,
        "hard_scores": hard_scores,
        "soft_scores": soft_scores,
        "context_summaries": context.get("news", {}).get("summaries", []),
        "news_snippets": news_snippets,
        "news_sources": news_sources,
    }

    start = datetime.utcnow()
    narrative = await _invoke_narrative_llm(payload)
    latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

    return {
        "ticker": ticker,
        "analysis_task_id": score.task_id,
        "model_version": score.model_version,
        "report": narrative,
        "context": context,
        "soft_scores": soft_scores,
        "hard_scores": hard_scores,
        "latency_ms": latency_ms,
        "news_sources": news_sources,
    }
