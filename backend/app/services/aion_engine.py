from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models import (
    AionModel,
    FactorCategory,
    ModelCategoryWeight,
    ModelFactorWeight,
    SystemFactor,
)
from app.services.factors import compute_all_factors
from app.services.insights_builder import build_advanced_insights


def _action_card(score: Optional[float]) -> str:
    if score is None:
        return "Data unavailable"
    if score >= 4.5:
        return "Strong Buy"
    if score >= 4.0:
        return "Buy / Accumulate"
    if score >= 3.0:
        return "Hold / Wait"
    if score >= 2.0:
        return "Reduce Risk"
    return "Avoid"


async def _resolve_model_id(model_version: str, db: AsyncSession) -> uuid.UUID:
    name_candidates = {model_version}
    if model_version.startswith("AION "):
        name_candidates.add(model_version.replace("AION ", "", 1))
    else:
        name_candidates.add(f"AION {model_version}")

    stmt = select(AionModel.id).where(AionModel.model_name.in_(name_candidates))
    result = await db.execute(stmt)
    row = result.first()
    if row is None:
        raise ValueError(f"No model found for version '{model_version}'")
    return row[0]


async def load_category_weights(model_version: str, db: AsyncSession) -> Dict[str, float]:
    """
    Load category-level weights (Level 1) for a given model_version from the database.
    """
    model_id = await _resolve_model_id(model_version, db)
    stmt = (
        select(FactorCategory.category_code, ModelCategoryWeight.weight)
        .join(ModelCategoryWeight, FactorCategory.id == ModelCategoryWeight.category_id)
        .where(ModelCategoryWeight.model_id == model_id)
    )
    result = await db.execute(stmt)
    rows = result.all()
    if not rows:
        raise ValueError(f"No category weights found for model_version '{model_version}'")
    return {code: float(weight) for code, weight in rows}


async def load_factor_weights(model_version: str, db: AsyncSession) -> Dict[str, float]:
    """
    Load factor weights (Level 2) for a given model_version from the database.
    """
    model_id = await _resolve_model_id(model_version, db)
    stmt = (
        select(SystemFactor.factor_code, ModelFactorWeight.weight)
        .join(ModelFactorWeight, SystemFactor.id == ModelFactorWeight.factor_id)
        .where(ModelFactorWeight.model_id == model_id)
    )
    result = await db.execute(stmt)
    rows = result.all()
    if not rows:
        raise ValueError(f"No factor weights found for model_version '{model_version}'")
    return {code: float(weight) for code, weight in rows}


async def load_model_weights(model_version: str, db: AsyncSession) -> tuple[Dict[str, float], Dict[str, float]]:
    category_weights = await load_category_weights(model_version, db)
    factor_weights = await load_factor_weights(model_version, db)
    return category_weights, factor_weights


def _aggregate_scores(
    factors: Dict[str, Any], category_weights: Dict[str, float]
) -> Dict[str, Any]:
    weighted_score = 0.0
    weight_sum = 0.0
    factor_output: Dict[str, Any] = {}

    for fid, result in factors.items():
        weight = float(category_weights.get(fid, 0.0) or 0.0)
        factor_output[fid] = {
            "score": result.score,
            "status": result.status,
            "weight": weight,
            "summary": getattr(result, "summary", None),
            "key_evidence": getattr(result, "key_evidence", []),
            "sources": getattr(result, "sources", []),
            "components": result.components,
            "errors": result.errors,
            "weight_denominator": result.weight_denominator,
        }
        if result.score is None or weight <= 0:
            continue
        weighted_score += result.score * weight
        weight_sum += weight

    total_score = weighted_score / weight_sum if weight_sum > 0 else None
    return {
        "factor_output": factor_output,
        "total_score": total_score,
        "weight_denominator": weight_sum,
    }


async def calculate(
    ticker: str,
    model_version: str,
    *,
    factor_weights: Optional[Dict[str, float]] = None,
    category_weights: Optional[Dict[str, float]] = None,
    db: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """
    Run all implemented factors, aggregate to a total AION score, and return structured output.
    Missing or unavailable factors are excluded from the weight denominator.
    """
    if factor_weights is None or category_weights is None:
        if db is None:
            raise ValueError("Either weights or db must be provided to calculate AION score.")
        category_weights, factor_weights = await load_model_weights(model_version, db)

    factors = await compute_all_factors(ticker, factor_weights=factor_weights, db=db)

    aggregated = _aggregate_scores(factors, category_weights)
    card = _action_card(aggregated["total_score"])

    insights = await build_advanced_insights(
        ticker,
        total_score=aggregated["total_score"],
        action_card=card,
        factors=aggregated["factor_output"],
    )

    return {
        "ticker": ticker,
        "model_version": model_version,
        "total_score": aggregated["total_score"] or 0.0,
        "action_card": card,
        "calculated_at": datetime.utcnow(),
        "factors": aggregated["factor_output"],
        "weight_denominator": aggregated["weight_denominator"],
        **insights,
    }
