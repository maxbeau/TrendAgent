from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy import desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.serializers import serialize_analysis_score
from app.db import get_db
from app.models import AnalysisScore

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)) -> Dict[str, List[Dict[str, Any]]]:
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

    return {"latest_scores": [serialize_analysis_score(score) for score in latest_scores]}
