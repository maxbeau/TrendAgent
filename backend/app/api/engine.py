from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

from app.services import aion_engine


router = APIRouter(prefix="/engine", tags=["engine"])


class CalculationRequest(BaseModel):
    ticker: str
    model_version: str


class CalculationResult(BaseModel):
    ticker: str
    total_score: float
    action_card: str
    calculated_at: datetime


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
