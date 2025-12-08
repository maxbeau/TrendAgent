from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SoftFactorScore
from app.services.llm import build_small_llm

FACTOR_CODE_MAP = {
    "industry": "F2",
    "catalyst": "F7",
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
) -> Dict[str, Any]:
    asof = asof_date or date.today()
    factor_code = FACTOR_CODE_MAP.get(factor, factor)

    cached = await fetch_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof,
        db=db,
    )
    if cached:
        return {
            "factor_code": factor_code,
            "score": cached.score,
            "confidence": cached.confidence,
            "summary": normalize_summary(cached.reasons),
            "key_evidence": normalize_key_evidence(cached.citations),
            "asof_date": asof.isoformat(),
        }

    cleaned = [text.strip() for text in texts if text and text.strip()]
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

    await upsert_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof,
        score=parsed.score,
        confidence=parsed.confidence,
        reasons=normalize_summary(parsed.summary),
        citations=normalize_key_evidence(parsed.key_evidence),
        db=db,
    )
    return parsed_dict
