from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from app.services.factors.core import (
    _bucketize,
    _fetch_with_default,
    _mean,
    _safe_float,
    _summarize_institutional_holders,
    _weighted_average,
    FactorResult,
)
from app.services.providers import FMPProvider, YFinanceProvider
from app.services.providers.base import ProviderError


def _summarize_flow(
    holder_summary: Dict[str, Any],
    put_call_payload: Dict[str, Any],
    gex_payload: Dict[str, Any],
) -> str:
    parts: List[str] = []
    trend_metric = _safe_float(holder_summary.get("trend_metric"))
    if trend_metric is not None:
        direction = "回升" if trend_metric >= 0 else "下降"
        parts.append(f"机构持仓{direction} {abs(trend_metric)*100:.1f}%")
    else:
        count = holder_summary.get("latest_holder_count")
        if isinstance(count, (int, float)) and count > 0:
            parts.append(f"机构覆盖 {int(count)} 家")

    ratio = _safe_float((put_call_payload or {}).get("put_call_ratio"))
    if ratio is not None:
        if ratio > 1.2:
            parts.append(f"Put/Call {ratio:.2f} · 防守升温")
        elif ratio < 0.8:
            parts.append(f"Put/Call {ratio:.2f} · 看涨占优")
        else:
            parts.append(f"Put/Call {ratio:.2f} · 中性")

    total_gamma = _safe_float((gex_payload or {}).get("total_gamma_exposure"))
    if total_gamma is not None:
        if total_gamma >= 0:
            parts.append("正Gamma 提供缓冲")
        else:
            parts.append("负Gamma 放大波动")

    return "；".join(part for part in parts if part) or "资金流向数据等待更新"


async def _load_institutional_summary(
    ticker: str,
    yfinance: YFinanceProvider,
    fmp: FMPProvider,
) -> tuple[Dict[str, Any], List[str]]:
    """
    Fetch institutional holders from both providers in parallel, then summarize with fallback.
    """
    y_task = asyncio.create_task(_fetch_with_default(yfinance.fetch_holders(ticker), "holders", []))
    fmp_task = asyncio.create_task(
        _fetch_with_default(fmp.fetch_institutional_holders(ticker), "fmp_holders", [])
    )
    holders, holders_error = await y_task
    fmp_holders, fmp_error = await fmp_task

    preferred_records = fmp_holders or holders
    summary = _summarize_institutional_holders(
        preferred_records,
        source="fmp" if fmp_holders else "yfinance",
    )
    summary["available_sources"] = {"fmp": bool(fmp_holders), "yfinance": bool(holders)}
    summary["source_preference"] = summary["source"]

    errors = [err for err in (holders_error, fmp_error) if err]
    return summary, errors


async def compute_flow(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    massive: Optional[MassiveProvider] = None,
    fmp: Optional[FMPProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    y_provider = yfinance or YFinanceProvider()
    fmp_provider = fmp or FMPProvider()
    weights = factor_weights or {}
    errors: List[str] = []

    holder_summary, holder_errors = await _load_institutional_summary(ticker, y_provider, fmp_provider)
    errors.extend(holder_errors)

    put_call_task = asyncio.create_task(
        _fetch_with_default(y_provider.fetch_put_call_ratio(ticker), "put_call", {})
    )
    gex_task = asyncio.create_task(
        _fetch_with_default(y_provider.fetch_gamma_exposure(ticker), "gex", {})
    )
    put_call, put_call_error = await put_call_task
    gex, gex_error = await gex_task
    errors.extend(err for err in (put_call_error, gex_error) if err)

    inst_presence_score = _bucketize(holder_summary.get("latest_holder_count"), [1, 3, 5, 8])
    trend_metric = holder_summary.get("trend_metric")
    trend_score = None
    if trend_metric is not None:
        trend_score = _bucketize(trend_metric, [-0.2, -0.05, 0.05, 0.15])
    inst_score = _mean([inst_presence_score, trend_score]) or inst_presence_score

    pcr_score = _bucketize(put_call.get("put_call_ratio"), [0.6, 0.8, 1.0, 1.2], higher_is_better=False)
    gex_score = None
    if gex.get("total_gamma_exposure") is not None:
        gex_score = _bucketize(gex["total_gamma_exposure"], [-1e8, -1e7, 0, 1e7])

    factor_scores = {
        "flow.institutional": inst_score,
        "flow.options_activity": pcr_score,
        "flow.gamma_exposure": gex_score,
        "flow.etf_behavior": None,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "degraded" if errors else "unavailable"

    summary_text = _summarize_flow(holder_summary, put_call, gex)

    return FactorResult(
        score=score,
        status=status,
        summary=summary_text,
        components={
            "institutional_count": holder_summary.get("latest_holder_count"),
            "institutional_trend": holder_summary,
            "institutional_sources": holder_summary.get("available_sources"),
            "put_call": put_call,
            "gamma_exposure": gex,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )
