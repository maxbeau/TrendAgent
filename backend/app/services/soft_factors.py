from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import and_, select
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
    reasons: List[str] = Field(default_factory=list, description="Short reasons grounded in provided texts")
    citations: List[int] = Field(default_factory=list, description="Indexes of snippets used")


async def _get_cached_score(ticker: str, factor_code: str, asof_date: date, db: AsyncSession) -> Optional[SoftFactorScore]:
    stmt = (
        select(SoftFactorScore)
        .where(
            and_(
                SoftFactorScore.ticker == ticker,
                SoftFactorScore.factor_code == factor_code,
                SoftFactorScore.asof_date == asof_date,
            )
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalars().first()


async def _persist_score(
    ticker: str,
    factor_code: str,
    asof_date: date,
    payload: SoftFactorSchema,
    db: AsyncSession,
) -> SoftFactorScore:
    entry = SoftFactorScore(
        ticker=ticker,
        factor_code=factor_code,
        asof_date=asof_date,
        score=payload.score,
        confidence=payload.confidence,
        reasons=payload.reasons,
        citations=payload.citations,
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

    cached = await _get_cached_score(ticker, factor_code, asof, db)
    if cached:
        return {
            "factor_code": factor_code,
            "score": cached.score,
            "confidence": cached.confidence,
            "reasons": cached.reasons,
            "citations": cached.citations,
            "asof_date": asof.isoformat(),
        }

    cleaned = [text.strip() for text in texts if text and text.strip()]
    if not cleaned:
        fallback = SoftFactorSchema(
            factor_code=factor_code,
            score=None,
            confidence=None,
            reasons=["No context available"],
            citations=[],
        )
        await _persist_score(ticker, factor_code, asof, fallback, db)
        return fallback.model_dump()

    try:
        llm = build_small_llm(temperature=0.2)
    except ValueError:
        fallback = SoftFactorSchema(
            factor_code=factor_code,
            score=None,
            confidence=None,
            reasons=["OpenAI API key missing; skipping soft factor scoring"],
            citations=[],
        )
        await _persist_score(ticker, factor_code, asof, fallback, db)
        return fallback.model_dump()

    parser = JsonOutputParser(pydantic_object=SoftFactorSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是行业分析的软因子打分 Agent。仅依据提供的文本，输出 1-5 分数、0-1 置信度，"
                "并给出引用索引的要点。不得臆造信息，缺数据时 score 设为 null。",
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

    await _persist_score(ticker, factor_code, asof, parsed, db)
    return parsed.model_dump()
