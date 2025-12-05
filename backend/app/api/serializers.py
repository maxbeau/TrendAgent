from typing import Any, Dict

from app.models import AnalysisScore


def serialize_analysis_score(score: AnalysisScore) -> Dict[str, Any]:
    """将 AnalysisScore ORM 对象转换为可序列化字典。"""
    return {
        "id": str(score.id),
        "task_id": score.task_id,
        "ticker": score.ticker,
        "total_score": score.total_score,
        "model_version": score.model_version,
        "action_card": score.action_card,
        "factors": score.factors,
        "weight_denominator": score.weight_denominator,
        "calculated_at": score.calculated_at.isoformat() if score.calculated_at else None,
        "created_at": score.created_at.isoformat() if score.created_at else None,
    }
