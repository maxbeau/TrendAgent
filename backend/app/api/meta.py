from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import FactorCategory, ModelFactorWeight, SystemFactor
from app.services.aion_engine import _resolve_model_id


router = APIRouter(prefix="/meta", tags=["meta"])

CATEGORY_MAP = {
    "F1": "macro",
    "F2": "industry",
    "F3": "fundamental",
    "F4": "technical",
    "F5": "flow",
    "F6": "sentiment",
    "F7": "catalyst",
    "F8": "volatility",
}


def _derive_formula(entries: List[Dict[str, Any]], *, fallback: Optional[str]) -> str:
    positives = [e for e in entries if e["weight"] > 0]
    total = sum(e["weight"] for e in positives)
    if total <= 0 and fallback:
        return fallback
    if total <= 0:
        return "公式待配置"
    terms = []
    for entry in sorted(positives, key=lambda x: x["weight"], reverse=True):
        pct = entry["weight"] / total
        name = entry["name"] or entry["code"].split(".")[-1]
        terms.append(f"{pct:.2f}×{name}")
    return "Score = " + " + ".join(terms)


@router.get("/factors")
async def list_factor_meta(
    model_version: str = "AION v8.0",
    db: AsyncSession = Depends(get_db),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    返回因子元数据：公式说明（基于数据库权重动态生成）、描述、子因子权重。
    """
    # 取类别定义
    cat_result = await db.execute(select(FactorCategory))
    cat_rows = cat_result.scalars().all()
    code_to_key = {row.category_code: CATEGORY_MAP.get(row.category_code) for row in cat_rows}

    # 取模型权重并拼接可读公式
    model_id = await _resolve_model_id(model_version, db)
    stmt = (
        select(
            FactorCategory.category_code,
            FactorCategory.description,
            SystemFactor.factor_code,
            SystemFactor.name,
            ModelFactorWeight.weight,
        )
        .join(SystemFactor, SystemFactor.category_id == FactorCategory.id)
        .join(ModelFactorWeight, ModelFactorWeight.factor_id == SystemFactor.id)
        .where(ModelFactorWeight.model_id == model_id)
    )
    result = await db.execute(stmt)
    rows = result.fetchall()

    factors: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        category_code = row.category_code
        factor_key = code_to_key.get(category_code)
        if not factor_key:
            continue
        bucket = factors.setdefault(
            factor_key,
            {
                "factor_key": factor_key,
                "description": row.description or "",
                "items": [],
            },
        )
        bucket["items"].append(
            {
                "code": row.factor_code,
                "name": row.name,
                "weight": float(row.weight or 0),
            }
        )

    response: List[Dict[str, Any]] = []
    for factor_key, payload in factors.items():
        items = payload["items"]
        formula_text = _derive_formula(items, fallback=None)
        total = sum(item["weight"] for item in items if item["weight"] > 0)
        response.append(
            {
                "factor_key": factor_key,
                "formula_text": formula_text,
                "description": payload.get("description") or "",
                "source": "weights",
                "updated_at": None,
                "weights": [
                    {
                        "factor_code": it["code"],
                        "name": it["name"],
                        "weight": it["weight"],
                        "weight_pct": (it["weight"] / total * 100) if total > 0 else None,
                    }
                    for it in sorted(items, key=lambda x: x["weight"], reverse=True)
                ],
                "weight_denominator": total,
            }
        )

    # 若某些分类无权重仍返回占位（使用 fallback 或默认公式）
    for code, factor_key in CATEGORY_MAP.items():
        if factor_key in {item["factor_key"] for item in response}:
            continue
        cat = next((c for c in cat_rows if c.category_code == code), None)
        response.append(
            {
                "factor_key": factor_key,
                "formula_text": (cat.formula_text if cat else None) or "公式待配置",
                "description": cat.description if cat else "",
                "source": "default",
                "updated_at": None,
                "weights": [],
                "weight_denominator": 0,
            }
        )

    return {"factors": response}
