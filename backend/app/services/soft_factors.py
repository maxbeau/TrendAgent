from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
import math
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SoftFactorScore
from app.services.event_intensity import summarize_event_intensity
from app.services.llm import build_small_llm

logger = logging.getLogger(__name__)

FACTOR_CODE_MAP = {
    "industry": "F2",
    "catalyst": "F7",
}

CATALYST_SUB_WEIGHTS = {
    "event_intensity": 0.5,
    "time_proximity": 0.3,
    "occurrence_probability": 0.2,
}


class SoftFactorSchema(BaseModel):
    factor_code: str = Field(description="System factor code, e.g., F2 or F7")
    score: Optional[float] = Field(description="Score between 1 and 5 or null if insufficient context")
    confidence: Optional[float] = Field(description="Confidence between 0 and 1")
    summary: str = Field(default="", description="Brief summary grounded in provided texts")
    key_evidence: List[str] = Field(default_factory=list, description="Short evidence snippets grounded in texts")


def normalize_summary(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        summary = value.get("summary")
        if summary is not None:
            return str(summary)
        return "; ".join(f"{key}={val}" for key, val in value.items() if val is not None)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    return str(value)


def normalize_key_evidence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _normalize_component_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score <= 0:
        return None
    if score <= 1:
        score *= 5.0
    return max(1.0, min(5.0, score))


def _parse_calendar_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        try:
            return date.fromisoformat(trimmed[:10])
        except ValueError:
            try:
                return datetime.fromisoformat(trimmed).date()
            except ValueError:
                return None
    return None


def _time_proximity_score(
    target_date: Optional[date],
    *,
    asof: date,
    event_label: Optional[str] = None,
) -> tuple[Optional[float], Optional[str]]:
    if target_date is None:
        return None, None
    label = event_label or "事件"
    delta_days = (target_date - asof).days
    if delta_days < 0:
        score = 1.0
        note = f"{label}已于 {target_date.isoformat()} 发生（{abs(delta_days)} 天前）"
        return score, note
    if delta_days <= 7:
        score = 5.0
    elif delta_days <= 14:
        score = 4.0
    elif delta_days <= 30:
        score = 3.0
    elif delta_days <= 60:
        score = 2.0
    else:
        score = 1.0
    note = f"距下个{label} {delta_days} 天（{target_date.isoformat()}）"
    return score, note


def _weighted_subfactor_score(components: Dict[str, Optional[float]]) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for key, score in components.items():
        weight = CATALYST_SUB_WEIGHTS.get(key)
        if weight is None or score is None:
            continue
        weighted_sum += weight * score
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


def _occurrence_probability_score(
    confidence_score: Optional[float],
    *,
    has_scheduled_event: bool,
) -> Optional[float]:
    base = confidence_score
    if base is None:
        base = 2.5 if has_scheduled_event else None
    if base is None:
        return None
    bonus = 1.5 if has_scheduled_event else 0.0
    score = base + bonus
    return max(1.0, min(5.0, score))


def _articles_from_texts(texts: List[str]) -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        snippet = (text or "").strip()
        if not snippet:
            continue
        articles.append(
            {
                "title": snippet[:120] or f"Snippet #{idx}",
                "description": snippet,
                "published_utc": None,
            }
        )
    return articles


def _news_evidence_from_articles(articles: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    evidence: List[str] = []
    for article in articles[:limit]:
        title = (article.get("title") or article.get("headline") or "").strip()
        summary = (article.get("description") or article.get("summary") or "").strip()
        if not title and not summary:
            continue
        published = article.get("published_utc") or article.get("published_at")
        source = article.get("source")
        context_bits = [str(bit) for bit in (source, published) if bit]
        context = f" ({' / '.join(context_bits)})" if context_bits else ""
        text_parts = [title + context if title else context.strip()]
        if summary and summary != title:
            text_parts.append(summary)
        text = " - ".join(part for part in text_parts if part)
        if text:
            evidence.append(text)
    return evidence


@dataclass
class _CatalystContext:
    ticker: str
    asof: date
    texts: List[str]
    raw_context: Dict[str, Any]
    articles: List[Dict[str, Any]] = field(init=False)
    sources: List[Dict[str, Any]] = field(init=False)
    calendar_date: Optional[date] = field(init=False)

    def __post_init__(self) -> None:
        ctx = self.raw_context or {}
        self.articles = ctx.get("news_articles") or ctx.get("articles") or []
        if not self.articles:
            self.articles = _articles_from_texts(self.texts)
        self.sources = ctx.get("sources") or []
        calendar_payload = ctx.get("calendar") or {}
        calendar_value = (
            calendar_payload.get("next_earnings_date")
            or calendar_payload.get("earnings_date")
            or calendar_payload.get("nextEarningsDate")
        )
        self.calendar_date = _parse_calendar_date(calendar_value)

    def candidate_pool(self, event_metadata: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        metadata_candidates = _event_candidates_from_metadata(event_metadata)
        pool: List[Dict[str, Any]] = list(metadata_candidates)
        if self.calendar_date:
            pool.append(
                {
                    "date": self.calendar_date,
                    "label": "earnings",
                    "description": "下次财报",
                    "source": "calendar",
                }
            )
        pool.extend(_event_candidates_from_articles(self.articles))
        return pool, metadata_candidates

    def compute_time_components(self, selected_event: Optional[Dict[str, Any]]) -> tuple[Optional[float], Optional[str]]:
        if selected_event and selected_event.get("date"):
            label_text = (selected_event.get("label") or "催化事件").strip()
            score, proximity_note = _time_proximity_score(
                selected_event["date"],
                asof=self.asof,
                event_label=label_text or "催化事件",
            )
            event_desc = _describe_event_candidate(selected_event)
            if proximity_note and event_desc:
                note = f"{proximity_note} · {event_desc}"
            else:
                note = proximity_note or event_desc
            return score, note
        if self.calendar_date:
            return _time_proximity_score(self.calendar_date, asof=self.asof, event_label="财报事件")
        return None, None

    def build_evidence(
        self,
        key_events: Any,
        *,
        summary: Optional[str],
        calendar_note: Optional[str],
    ) -> List[str]:
        evidence = [item for item in normalize_key_evidence(key_events) if len(item) >= 8]
        if len(evidence) < 3:
            for snippet in _news_evidence_from_articles(self.articles):
                if snippet not in evidence:
                    evidence.append(snippet)
                if len(evidence) >= 5:
                    break
        if not evidence and summary:
            evidence = [summary]
        if calendar_note:
            evidence.append(calendar_note)
        return evidence


def _event_candidates_from_metadata(metadata: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not isinstance(metadata, list):
        return candidates
    for item in metadata:
        if not isinstance(item, dict):
            continue
        date_value = item.get("date") or item.get("event_date")
        parsed_date = _parse_calendar_date(date_value) if date_value else None
        if not parsed_date:
            continue
        candidates.append(
            {
                "date": parsed_date,
                "label": item.get("type") or item.get("label") or "event",
                "description": item.get("description") or item.get("summary"),
                "source": "metadata",
            }
        )
    return candidates


def _event_candidates_from_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for article in articles:
        date_value = (
            article.get("event_date")
            or article.get("published_utc")
            or article.get("published_at")
        )
        parsed_date = _parse_calendar_date(date_value)
        if not parsed_date:
            continue
        candidates.append(
            {
                "date": parsed_date,
                "label": article.get("source") or "news",
                "description": article.get("title") or article.get("headline"),
                "source": "news",
            }
        )
    return candidates


def _select_best_event_candidate(candidates: List[Dict[str, Any]], *, asof: date) -> Optional[Dict[str, Any]]:
    dated = [c for c in candidates if c.get("date") is not None]
    if not dated:
        return None
    future = [c for c in dated if c["date"] >= asof]
    if future:
        return min(future, key=lambda item: (item["date"] - asof).days)
    return max(dated, key=lambda item: item["date"])


def _describe_event_candidate(candidate: Optional[Dict[str, Any]]) -> Optional[str]:
    if not candidate:
        return None
    parts: List[str] = []
    label = candidate.get("label")
    description = candidate.get("description")
    if label:
        parts.append(str(label))
    if description:
        parts.append(str(description))
    date_value = candidate.get("date")
    if isinstance(date_value, date):
        parts.append(date_value.isoformat())
    return " · ".join(parts) if parts else None


def _serialize_event_component(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidate:
        return None
    payload: Dict[str, Any] = {}
    if candidate.get("label"):
        payload["label"] = str(candidate["label"])
    if candidate.get("description"):
        payload["description"] = str(candidate["description"])
    date_value = candidate.get("date")
    if isinstance(date_value, date):
        payload["date"] = date_value.isoformat()
    elif isinstance(date_value, str):
        payload["date"] = date_value
    if candidate.get("source"):
        payload["source"] = str(candidate["source"])
    return payload or None


async def fetch_soft_factor_score(
    *,
    ticker: str,
    factor_code: str,
    db: AsyncSession,
    asof_date: Optional[date] = None,
    latest: bool = False,
) -> Optional[SoftFactorScore]:
    stmt = select(SoftFactorScore).where(
        SoftFactorScore.ticker == ticker,
        SoftFactorScore.factor_code == factor_code,
    )
    if asof_date is not None:
        stmt = stmt.where(SoftFactorScore.asof_date == asof_date)
    if latest or asof_date is None:
        stmt = stmt.order_by(desc(SoftFactorScore.asof_date))
    stmt = stmt.limit(1)
    result = await db.execute(stmt)
    return result.scalars().first()


async def upsert_soft_factor_score(
    *,
    ticker: str,
    factor_code: str,
    asof_date: date,
    score: Optional[float],
    confidence: Optional[float],
    reasons: Any,
    citations: Optional[List[Any]],
    db: AsyncSession,
) -> SoftFactorScore:
    entry = await fetch_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof_date,
        db=db,
    )
    if entry:
        entry.score = score
        entry.confidence = confidence
        entry.reasons = reasons
        entry.citations = citations or []
    else:
        entry = SoftFactorScore(
            ticker=ticker,
            factor_code=factor_code,
            asof_date=asof_date,
            score=score,
            confidence=confidence,
            reasons=reasons,
            citations=citations or [],
        )
        db.add(entry)
    await db.commit()
    return entry


async def score_soft_factor(
    ticker: str,
    factor: str,
    texts: List[str],
    *,
    db: AsyncSession,
    asof_date: Optional[date] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    asof = asof_date or date.today()
    factor_code = FACTOR_CODE_MAP.get(factor, factor)

    cached = await fetch_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof,
        db=db,
    )
    cached_factor_scores = None
    cached_components = None
    if cached and isinstance(cached.reasons, dict):
        cached_factor_scores = cached.reasons.get("factor_scores")
        cached_components = cached.reasons.get("components")
    if cached:
        return {
            "factor_code": factor_code,
            "score": cached.score,
            "confidence": cached.confidence,
            "summary": normalize_summary(cached.reasons),
            "key_evidence": normalize_key_evidence(cached.citations),
            "asof_date": asof.isoformat(),
            "factor_scores": cached_factor_scores,
            "components": cached_components,
        }

    cleaned = [text.strip() for text in texts if text and text.strip()]
    context_payload = context or {}

    if factor_code == "F7":
        return await _score_catalyst_factor(
            ticker,
            factor_code,
            cleaned,
            db=db,
            asof=asof,
            context=context_payload,
        )

    if not cleaned:
        fallback = SoftFactorSchema(
            factor_code=factor_code,
            score=None,
            confidence=None,
            summary="No context available",
            key_evidence=[],
        )
        await upsert_soft_factor_score(
            ticker=ticker,
            factor_code=factor_code,
            asof_date=asof,
            score=fallback.score,
            confidence=fallback.confidence,
            reasons=normalize_summary(fallback.summary),
            citations=normalize_key_evidence(fallback.key_evidence),
            db=db,
        )
        return fallback.model_dump()

    try:
        llm = build_small_llm(temperature=0.2)
    except ValueError:
        fallback = SoftFactorSchema(
            factor_code=factor_code,
            score=None,
            confidence=None,
            summary="OpenAI API key missing; skipping soft factor scoring",
            key_evidence=[],
        )
        await upsert_soft_factor_score(
            ticker=ticker,
            factor_code=factor_code,
            asof_date=asof,
            score=fallback.score,
            confidence=fallback.confidence,
            reasons=normalize_summary(fallback.summary),
            citations=normalize_key_evidence(fallback.key_evidence),
            db=db,
        )
        return fallback.model_dump()

    parser = JsonOutputParser(pydantic_object=SoftFactorSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是行业分析的软因子打分 Agent。仅依据提供的文本，输出 1-5 分数、0-1 置信度，"
                "并生成简短摘要（summary）以及关键证据（key_evidence，引用原文的简短句子）。"
                "不得臆造信息，缺数据时 score 设为 null。",
            ),
            (
                "human",
                "Ticker: {ticker}\nFactor: {factor_code}\n文本列表（带索引）：\n{numbered_texts}\n"
                "{format_instructions}",
            ),
        ]
    )

    numbered_texts = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(cleaned))
    chain = prompt | llm | parser
    parsed = await chain.ainvoke(
        {
            "ticker": ticker,
            "factor_code": factor_code,
            "numbered_texts": numbered_texts,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    if isinstance(parsed, dict):
        parsed = SoftFactorSchema(**parsed)

    parsed_dict = parsed.model_dump()
    # 保持 sources 接口对齐，即便 LLM 不返回，调用方也可补充来源。
    parsed_dict["sources"] = []

    summary_text = normalize_summary(parsed.summary)
    reasons_payload: Any = summary_text
    factor_scores_payload = parsed_dict.get("factor_scores")
    if isinstance(factor_scores_payload, dict):
        reasons_payload = {
            "summary": summary_text,
            "factor_scores": factor_scores_payload,
        }

    await upsert_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof,
        score=parsed.score,
        confidence=parsed.confidence,
        reasons=reasons_payload,
        citations=normalize_key_evidence(parsed.key_evidence),
        db=db,
    )
    return parsed_dict


async def _score_catalyst_factor(
    ticker: str,
    factor_code: str,
    texts: List[str],
    *,
    db: AsyncSession,
    asof: date,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    catalyst_context = _CatalystContext(ticker=ticker, asof=asof, texts=texts, raw_context=context)

    try:
        events = await summarize_event_intensity(ticker, articles=catalyst_context.articles)
    except Exception as exc:  # pragma: no cover - LLM/provider issues
        logger.warning("catalyst_event_intensity_failed", extra={"ticker": ticker, "error": str(exc)})
        events = {
            "score": None,
            "confidence": None,
            "summary": f"事件强度分析失败: {exc}",
            "key_events": [],
            "event_metadata": [],
            "sources": catalyst_context.sources,
        }

    event_intensity_score = _normalize_component_score(events.get("score"))
    confidence_component = _normalize_component_score(events.get("confidence"))
    normalized_confidence = (
        confidence_component / 5.0 if confidence_component is not None else None
    )

    candidate_pool, metadata_candidates = catalyst_context.candidate_pool(events.get("event_metadata"))
    selected_event = _select_best_event_candidate(candidate_pool, asof=asof)
    time_score, calendar_note = catalyst_context.compute_time_components(selected_event)

    occurrence_score = _occurrence_probability_score(
        confidence_component,
        has_scheduled_event=bool(metadata_candidates or catalyst_context.calendar_date),
    )

    factor_scores = {
        "event_intensity": event_intensity_score,
        "time_proximity": time_score,
        "occurrence_probability": occurrence_score,
    }
    final_score = _weighted_subfactor_score(factor_scores)

    selected_event_component = _serialize_event_component(selected_event)
    candidate_preview = [
        comp
        for comp in (_serialize_event_component(item) for item in candidate_pool[:5])
        if comp is not None
    ]
    components_payload = {
        "selected_event": selected_event_component,
        "event_candidates": candidate_preview,
        "event_source_count": len(candidate_pool),
    }

    summary_parts = [events.get("summary")]
    if calendar_note:
        summary_parts.append(calendar_note)
    summary = " | ".join(part for part in summary_parts if part)
    if not summary:
        summary = "未获取到催化事件摘要"

    evidence = catalyst_context.build_evidence(
        events.get("key_events"),
        summary=summary,
        calendar_note=calendar_note,
    )

    sources = catalyst_context.sources or events.get("sources") or []
    result = {
        "factor_code": factor_code,
        "score": final_score,
        "confidence": normalized_confidence,
        "summary": summary,
        "key_evidence": evidence,
        "sources": sources,
        "asof_date": asof.isoformat(),
        "factor_scores": factor_scores,
        "components": components_payload,
    }

    reasons_payload = {"summary": summary, "factor_scores": factor_scores, "components": components_payload}
    await upsert_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof,
        score=final_score,
        confidence=normalized_confidence,
        reasons=reasons_payload,
        citations=evidence,
        db=db,
    )
    return result
