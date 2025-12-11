from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.market import get_ohlc
from app.api.meta import list_factor_meta
from app.db import get_db
from app.models import AnalysisScore
from app.schemas import AnalysisScoreSchema, ReportV2Response


router = APIRouter(prefix="/report", tags=["report"])


@router.get("/v2/{ticker}", response_model=ReportV2Response)
async def get_full_report(
    ticker: str,
    limit: int = Query(200, ge=50, le=500),
    db: AsyncSession = Depends(get_db),
) -> ReportV2Response:
    normalized_ticker = ticker.upper()
    stmt = (
        select(AnalysisScore)
        .where(AnalysisScore.ticker == normalized_ticker)
        .order_by(desc(AnalysisScore.calculated_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    score = result.scalars().first()
    if not score:
        raise HTTPException(status_code=404, detail=f"No analysis score found for {normalized_ticker}")

    analysis = AnalysisScoreSchema.model_validate(score)
    ohlc = await get_ohlc(ticker=normalized_ticker, limit=limit, db=db)
    factor_meta = await list_factor_meta(model_version=analysis.model_version, db=db)

    return ReportV2Response(
        analysis=analysis,
        ohlc=ohlc,
        factor_meta=factor_meta,
    )
