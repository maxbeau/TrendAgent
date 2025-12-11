from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy import desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db import get_db
from app.models import AnalysisScore
from app.schemas import AnalysisScoreSchema, DashboardSummaryResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)) -> DashboardSummaryResponse:
    """
    获取每个 Ticker 最新的 AION 评分结果。
    """
    stmt = (
        select(AnalysisScore)
        .order_by(AnalysisScore.ticker, desc(AnalysisScore.calculated_at))
        .distinct(AnalysisScore.ticker)
    )
    result = await db.execute(stmt)
    latest_scores = result.scalars().all()

    return DashboardSummaryResponse(
        latest_scores=[AnalysisScoreSchema.model_validate(score) for score in latest_scores]
    )
