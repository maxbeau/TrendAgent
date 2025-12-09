import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.serializers import serialize_analysis_score
from app.db import AsyncSessionLocal, get_db
from app.models import AnalysisScore, AionModel
from app.services import aion_engine
from app.services.event_intensity import summarize_event_intensity
from app.services.policy_tailwind import summarize_policy_tailwind
from app.services.providers.base import ProviderError
from app.services.risk_reward import summarize_risk_reward
from app.services.tam_expansion import summarize_tam_expansion


router = APIRouter(prefix="/engine", tags=["engine"])
logger = logging.getLogger(__name__)


class CalculationRequest(BaseModel):
    ticker: str
    model_version: Optional[str] = None


class PolicyTailwindRequest(BaseModel):
    ticker: str
    limit: int = 12
    lookback_days: int = 60


class PolicyTailwindResponse(BaseModel):
    ticker: str
    verdict: str
    confidence: int
    summary: str
    key_points: List[str]
    sources: List[Dict[str, Optional[str]]]
    generated_at: datetime


class EventIntensityRequest(BaseModel):
    ticker: str
    limit: int = 12
    lookback_days: int = 30


class EventIntensityResponse(BaseModel):
    ticker: str
    intensity: str
    score: int
    confidence: int
    summary: str
    key_events: List[str]
    sources: List[Dict[str, Optional[str]]]
    generated_at: datetime


class TAMExpansionRequest(BaseModel):
    ticker: str
    limit: int = 12
    lookback_days: int = 90


class TAMExpansionResponse(BaseModel):
    ticker: str
    outlook: str
    score: int
    confidence: int
    summary: str
    key_points: List[str]
    sources: List[Dict[str, Optional[str]]]
    generated_at: datetime


class RiskRewardRequest(BaseModel):
    ticker: str
    lookback_days: int = 120


class RiskRewardResponse(BaseModel):
    ticker: str
    status: str
    close: Optional[float] = None
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None
    reward: Optional[float] = None
    risk: Optional[float] = None
    ratio: Optional[float] = None
    lookback_days: int
    message: Optional[str] = None
    generated_at: datetime


class CalculationTaskResponse(BaseModel):
    message: str
    task_id: str


async def run_calculation_in_background(
    task_id: str, ticker: str, model_version: str
) -> None:
    """
    后台执行 AION 计算并写入数据库。
    """
    try:
        async with AsyncSessionLocal() as db:
            category_weights, factor_weights = await aion_engine.load_model_weights(model_version, db)
            result_data = await aion_engine.calculate(
                ticker,
                model_version=model_version,
                factor_weights=factor_weights,
                category_weights=category_weights,
                db=db,
            )
            factors_payload = jsonable_encoder(result_data["factors"])

            score_entry = AnalysisScore(
                task_id=task_id,
                ticker=result_data["ticker"],
                total_score=result_data["total_score"],
                model_version=result_data["model_version"],
                action_card=result_data["action_card"],
                factors=factors_payload,
                weight_denominator=result_data["weight_denominator"],
                calculated_at=result_data["calculated_at"],
                scenarios=result_data.get("scenarios"),
                key_variables=result_data.get("key_variables"),
                stock_strategy=result_data.get("stock_strategy"),
                option_strategies=result_data.get("option_strategies"),
                risk_management=result_data.get("risk_management"),
                execution_notes=result_data.get("execution_notes"),
            )

            db.add(score_entry)
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Task %s failed", task_id)


@router.post("/calculate", status_code=202, response_model=CalculationTaskResponse)
async def calculate_aion_score(
    background_tasks: BackgroundTasks,
    payload: CalculationRequest,
    db: AsyncSession = Depends(get_db),
) -> CalculationTaskResponse:
    model_version = payload.model_version
    if not model_version:
        # fetch the latest model version from the database
        stmt = select(AionModel).order_by(AionModel.created_at.desc())
        result = await db.execute(stmt)
        latest_model = result.scalars().first()
        if not latest_model:
            raise HTTPException(status_code=404, detail="No AION model found in the database.")
        model_version = latest_model.model_name

    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_calculation_in_background, task_id, payload.ticker, model_version)
    return CalculationTaskResponse(message="Calculation started", task_id=task_id)


@router.get("/status/{task_id}")
async def get_task_status(task_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    根据 Task ID 从数据库查询计算结果。
    """
    stmt = select(AnalysisScore).where(AnalysisScore.task_id == task_id)
    result = await db.execute(stmt)
    score_entry = result.scalar_one_or_none()

    if score_entry:
        return {"status": "completed", "data": serialize_analysis_score(score_entry)}

    return {"status": "pending"}


@router.post("/policy-tailwind", response_model=PolicyTailwindResponse)
async def policy_tailwind(payload: PolicyTailwindRequest) -> PolicyTailwindResponse:
    try:
        result = await summarize_policy_tailwind(
            payload.ticker, limit=payload.limit, lookback_days=payload.lookback_days
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PolicyTailwindResponse(**result)


@router.post("/event-intensity", response_model=EventIntensityResponse)
async def event_intensity(payload: EventIntensityRequest) -> EventIntensityResponse:
    try:
        result = await summarize_event_intensity(
            payload.ticker, limit=payload.limit, lookback_days=payload.lookback_days
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return EventIntensityResponse(**result)


@router.post("/tam-expansion", response_model=TAMExpansionResponse)
async def tam_expansion(payload: TAMExpansionRequest) -> TAMExpansionResponse:
    try:
        result = await summarize_tam_expansion(
            payload.ticker, limit=payload.limit, lookback_days=payload.lookback_days
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TAMExpansionResponse(**result)


@router.post("/risk-reward", response_model=RiskRewardResponse)
async def risk_reward(payload: RiskRewardRequest) -> RiskRewardResponse:
    try:
        result = await summarize_risk_reward(
            payload.ticker, lookback_days=payload.lookback_days
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Normalize optional fields for Pydantic
    return RiskRewardResponse(
        ticker=result.get("ticker"),
        status=result.get("status"),
        close=result.get("close"),
        recent_high=result.get("recent_high"),
        recent_low=result.get("recent_low"),
        reward=result.get("reward"),
        risk=result.get("risk"),
        ratio=result.get("ratio"),
        lookback_days=result.get("lookback_days", payload.lookback_days),
        message=result.get("message"),
        generated_at=result.get("generated_at"),
    )
