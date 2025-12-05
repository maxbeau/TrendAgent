from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import aion_engine
from app.services.event_intensity import summarize_event_intensity
from app.services.policy_tailwind import summarize_policy_tailwind
from app.services.tam_expansion import summarize_tam_expansion
from app.services.risk_reward import summarize_risk_reward
from app.services.providers.base import ProviderError


router = APIRouter(prefix="/engine", tags=["engine"])


class CalculationRequest(BaseModel):
    ticker: str
    model_version: str


class CalculationResult(BaseModel):
    ticker: str
    total_score: float
    action_card: str
    calculated_at: datetime


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


@router.post("/calculate", response_model=CalculationResult)
def calculate(payload: CalculationRequest) -> CalculationResult:
    base = aion_engine.calculate(payload.ticker, payload.model_version)
    return CalculationResult(
        ticker=payload.ticker,
        total_score=base["total_score"],
        action_card=base["action_card"],
        calculated_at=base["calculated_at"],
    )


@router.post("/narrative/generate")
def generate_narrative(ticker: str) -> dict:
    prompt = aion_engine.build_narrative_context(ticker)
    # TODO: schedule async LangChain run
    return {"status": "scheduled", "ticket": f"narrative-{ticker}-{int(datetime.utcnow().timestamp())}", "prompt": prompt}


@router.get("/narrative/status/{ticket}")
def narrative_status(ticket: str) -> dict:
    return {"ticket": ticket, "status": "queued", "updated_at": datetime.utcnow()}


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
