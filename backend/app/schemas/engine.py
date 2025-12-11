from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CalculationRequest(BaseModel):
    ticker: str
    model_version: Optional[str] = None


class CalculationTaskResponse(BaseModel):
    message: str
    task_id: str


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


class AnalysisScoreSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    task_id: str
    ticker: str
    total_score: float
    model_version: str
    action_card: Optional[str] = None
    factors: Dict[str, Any]
    weight_denominator: Optional[float] = None
    scenarios: Optional[Union[Dict[str, Any], List[Any]]] = None
    key_variables: Optional[Union[Dict[str, Any], List[Any]]] = None
    stock_strategy: Optional[Union[Dict[str, Any], List[Any]]] = None
    option_strategies: Optional[Union[Dict[str, Any], List[Any]]] = None
    risk_management: Optional[Union[Dict[str, Any], List[Any]]] = None
    execution_notes: Optional[Union[Dict[str, Any], List[Any]]] = None
    calculated_at: datetime
    created_at: Optional[datetime] = None


class TaskStatusResponse(BaseModel):
    status: str
    data: Optional[AnalysisScoreSchema] = None


class DashboardSummaryResponse(BaseModel):
    latest_scores: List[AnalysisScoreSchema]


class ReportV2Response(BaseModel):
    analysis: AnalysisScoreSchema
    ohlc: Any
    factor_meta: Any
