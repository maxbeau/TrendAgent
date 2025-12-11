from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SoftFactorScore
from app.services.factors.core import (
    _bucketize,
    _latest_values,
    _percentile_thresholds_from_values,
    _relative_change,
    _sanitize_weights,
    _safe_float,
    _records_to_frame,
    _weighted_average,
    FactorResult,
    _summarize_institutional_holders,
)
from app.services.policy_tailwind import summarize_policy_tailwind
from app.services.providers import FMPProvider, MassiveProvider, YFinanceProvider
from app.services.providers.base import ProviderError
from app.services.soft_factors import (
    fetch_soft_factor_score,
    normalize_summary,
    score_soft_factor,
    upsert_soft_factor_score,
)


@dataclass
class IndustryMetrics:
    revenue_growth: Optional[float] = None
    operating_margin: Optional[float] = None
    previous_margin: Optional[float] = None
    margin_trend: Optional[float] = None
    capex_latest: Optional[float] = None
    capex_previous: Optional[float] = None
    capex_trend: Optional[float] = None


def _latest_capex_pair(cash_flow: pd.DataFrame, quarterly_cash_flow: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """
    Prefer quarterly CapEx (fresher), fallback to annual cash flow.
    """
    best_values: List[float] = []
    for frame in (quarterly_cash_flow, cash_flow):
        values = _latest_values(frame, ["Capital Expenditures", "Capital Expenditure"])
        if len(values) >= 2:
            best_values = values
            break
        if not best_values:
            best_values = values
    latest = best_values[0] if best_values else None
    previous = best_values[1] if len(best_values) >= 2 else None
    return latest, previous


def _industry_metrics_from_financials(financials: Optional[Dict[str, Any]]) -> IndustryMetrics:
    if not financials:
        return IndustryMetrics()

    income = _records_to_frame(financials.get("income_statement", []))
    cash_flow = _records_to_frame(financials.get("cash_flow", []))
    quarterly_cash_flow = _records_to_frame(financials.get("quarterly_cash_flow", []))

    rev = _latest_values(income, ["Total Revenue", "TotalRevenue", "Total Revenue USD"])
    op_inc = _latest_values(income, ["Operating Income", "OperatingIncome"])

    growth = None
    if len(rev) >= 2 and rev[1] not in (None, 0):
        growth = (rev[0] - rev[1]) / rev[1]

    op_margin = None
    prev_margin = None
    if op_inc and rev and rev[0]:
        op_margin = op_inc[0] / rev[0]
    if len(op_inc) >= 2 and len(rev) >= 2 and rev[1]:
        prev_margin = op_inc[1] / rev[1]

    margin_trend = _relative_change(op_margin, prev_margin)

    capex_latest, capex_previous = _latest_capex_pair(cash_flow, quarterly_cash_flow)
    capex_trend = None
    if capex_latest is not None and capex_previous is not None:
        capex_trend = _relative_change(abs(capex_latest), abs(capex_previous))

    return IndustryMetrics(
        revenue_growth=growth,
        operating_margin=op_margin,
        previous_margin=prev_margin,
        margin_trend=margin_trend,
        capex_latest=capex_latest,
        capex_previous=capex_previous,
        capex_trend=capex_trend,
    )


def _score_policy_tailwind_result(result: Dict[str, Any]) -> tuple[Optional[float], Optional[str], Optional[float]]:
    verdict = (result.get("verdict") or "").lower()
    confidence = _safe_float(result.get("confidence"))
    if confidence is not None:
        confidence = max(1.0, min(confidence, 5.0))

    base_score = {"tailwind": 5.0, "neutral": 3.0, "headwind": 1.0}.get(verdict)
    if base_score is None:
        return None, verdict, confidence
    if confidence is None:
        return base_score, verdict, None

    adjustment = (base_score - 3.0) * (confidence / 5.0)
    score = max(1.0, min(5.0, 3.0 + adjustment))
    return score, verdict, confidence


async def _load_peer_industry_metrics(
    ticker: str,
    provider: YFinanceProvider,
    *,
    limit: int = 6,
) -> tuple[List[IndustryMetrics], List[str], List[str]]:
    """
    Fetch basic industry metrics for peer tickers to build relative thresholds.
    """
    try:
        peer_symbols = await provider.fetch_peer_tickers(ticker, limit=limit)
    except ProviderError as exc:
        return [], [], [f"peers: {exc}"]

    filtered = [sym for sym in peer_symbols if isinstance(sym, str) and sym.upper() != ticker.upper()]
    if not filtered:
        return [], [], []

    tasks = [asyncio.create_task(provider.fetch_financials(symbol)) for symbol in filtered[:limit]]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    metrics: List[IndustryMetrics] = []
    errors: List[str] = []
    used_symbols: List[str] = []

    for symbol, payload in zip(filtered[:limit], results):
        if isinstance(payload, Exception):
            errors.append(f"{symbol}: {payload}")
            continue
        metrics.append(_industry_metrics_from_financials(payload))
        used_symbols.append(symbol)

    return metrics, used_symbols, errors


_POLICY_TAILWIND_FACTOR_CODE = "industry.policy_tailwind"


def _policy_payload(entry: SoftFactorScore) -> Dict[str, Any]:
    if isinstance(entry.reasons, dict):
        payload = dict(entry.reasons)
    else:
        payload = {
            "summary": normalize_summary(entry.reasons),
        }
    payload.setdefault("verdict", None)
    payload.setdefault("key_points", [])
    payload.setdefault("sources", entry.citations or [])
    return payload


async def _load_cached_policy_tailwind(ticker: str, db: Optional[AsyncSession]) -> Optional[Dict[str, Any]]:
    if db is None:
        return None
    entry = await fetch_soft_factor_score(
        ticker=ticker,
        factor_code=_POLICY_TAILWIND_FACTOR_CODE,
        db=db,
        latest=True,
    )
    if not entry:
        return None
    payload = _policy_payload(entry)
    return {
        "ticker": ticker,
        "verdict": payload.get("verdict"),
        "confidence": entry.confidence or payload.get("confidence"),
        "summary": payload.get("summary"),
        "key_points": payload.get("key_points") or [],
        "sources": payload.get("sources") or entry.citations or [],
        "generated_at": entry.asof_date.isoformat(),
        "score": entry.score,
    }


async def _persist_policy_tailwind_score(
    ticker: str,
    *,
    summary: Optional[str],
    verdict: Optional[str],
    key_points: List[str],
    sources: List[Dict[str, Any]],
    score: Optional[float],
    confidence: Optional[float],
    db: Optional[AsyncSession],
) -> None:
    if db is None:
        return
    asof = date.today()
    payload = {"summary": summary, "verdict": verdict, "key_points": key_points, "sources": sources}
    await upsert_soft_factor_score(
        ticker=ticker,
        factor_code=_POLICY_TAILWIND_FACTOR_CODE,
        asof_date=asof,
        score=score,
        confidence=confidence,
        reasons=payload,
        citations=sources,
        db=db,
    )


async def _resolve_policy_tailwind(
    ticker: str,
    massive: MassiveProvider,
    db: Optional[AsyncSession],
    errors: List[str],
) -> tuple[Dict[str, Any], str]:
    cached = None
    if db is not None:
        try:
            cached = await _load_cached_policy_tailwind(ticker, db)
        except Exception as exc:  # pragma: no cover - db issues
            errors.append(f"policy_tailwind_cache: {exc}")
    if cached:
        cached["source"] = "cache"
        return cached, "cache"

    live: Dict[str, Any] = {}
    try:
        live = await summarize_policy_tailwind(ticker, massive=massive)
    except ProviderError as exc:  # pragma: no cover - network
        errors.append(f"policy_tailwind: {exc}")
    except Exception as exc:  # pragma: no cover - LLM issues
        errors.append(f"policy_tailwind: {exc}")
    policy_score, _, policy_conf = _score_policy_tailwind_result(live)
    if live:
        try:
            await _persist_policy_tailwind_score(
                ticker,
                summary=live.get("summary"),
                verdict=live.get("verdict"),
                key_points=live.get("key_points") or [],
                sources=live.get("sources") or [],
                score=policy_score,
                confidence=policy_conf or _safe_float(live.get("confidence")),
                db=db,
            )
        except Exception as exc:  # pragma: no cover - db issues
            errors.append(f"policy_tailwind_cache_store: {exc}")
    live["score"] = policy_score
    live["confidence"] = policy_conf or live.get("confidence")
    live["source"] = "live"
    return live, "live"


async def compute_industry(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    massive: Optional[MassiveProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
    financials: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None,
) -> FactorResult:
    """
    Industry factor: combine hard financial proxies (revenue growth, margin/capex trends)
    with a qualitative policy-tailwind check. Thresholds adapt to peer percentiles.
    """
    yf_provider = yfinance or YFinanceProvider()
    massive_provider = massive or MassiveProvider()
    raw_weights: Dict[str, float] = factor_weights or {}
    sanitized_weights, _ = _sanitize_weights(raw_weights)

    errors: List[str] = []
    financial_payload = financials
    if financial_payload is None:
        try:
            financial_payload = await yf_provider.fetch_financials(ticker)
        except ProviderError as exc:  # pragma: no cover - network
            errors.append(f"financials: {exc}")
            financial_payload = {}

    metrics = _industry_metrics_from_financials(financial_payload)

    peer_metrics, peer_symbols, peer_errors = await _load_peer_industry_metrics(ticker, yf_provider)
    if peer_errors:
        errors.extend([f"peer_metrics: {err}" for err in peer_errors])

    growth_values = [m.revenue_growth for m in peer_metrics if m.revenue_growth is not None]
    margin_values = [m.margin_trend for m in peer_metrics if m.margin_trend is not None]
    capex_values = [m.capex_trend for m in peer_metrics if m.capex_trend is not None]

    if metrics.revenue_growth is not None:
        growth_values.append(metrics.revenue_growth)
    if metrics.margin_trend is not None:
        margin_values.append(metrics.margin_trend)
    if metrics.capex_trend is not None:
        capex_values.append(metrics.capex_trend)

    growth_thresholds = _percentile_thresholds_from_values(
        growth_values, [0.0, 0.05, 0.10, 0.20]
    )
    margin_thresholds = _percentile_thresholds_from_values(
        margin_values, [-0.15, -0.05, 0.05, 0.15]
    )
    capex_thresholds = _percentile_thresholds_from_values(
        capex_values, [-0.20, -0.05, 0.05, 0.20]
    )

    policy_result, policy_source = await _resolve_policy_tailwind(ticker, massive_provider, db, errors)
    policy_score_val, policy_verdict, policy_conf_val = _score_policy_tailwind_result(policy_result)

    factor_scores = {
        "industry.growth": _bucketize(metrics.revenue_growth, growth_thresholds),
        "industry.margin_trend": _bucketize(metrics.margin_trend, margin_thresholds),
        "industry.capex_cycle": _bucketize(metrics.capex_trend, capex_thresholds),
        "industry.policy_tailwind": policy_score_val,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, sanitized_weights)

    status = "ok" if score is not None else "error" if errors else "unavailable"
    if weight_denominator == 0.0 and not applied_weights:
        status = "unavailable"
        errors.append("No industry sub-weights configured in DB")

    summary_parts = []
    if metrics.revenue_growth is not None:
        summary_parts.append(f"营收增速 {metrics.revenue_growth*100:.1f}%")
    if metrics.margin_trend is not None:
        summary_parts.append(f"利润率趋势 {metrics.margin_trend*100:.1f}%")
    if metrics.capex_trend is not None:
        summary_parts.append(f"CapEx 变化 {metrics.capex_trend*100:.1f}%")
    if policy_result.get("summary"):
        summary_parts.append(f"政策: {policy_result.get('summary')}")
    summary_text = "; ".join(summary_parts) if summary_parts else policy_result.get("summary") or "No industry data"

    components: Dict[str, Any] = {
        "revenue_growth": metrics.revenue_growth,
        "operating_margin": metrics.operating_margin,
        "operating_margin_prev": metrics.previous_margin,
        "operating_margin_trend": metrics.margin_trend,
        "capex_cycle_trend": metrics.capex_trend,
        "capex_latest": metrics.capex_latest,
        "capex_previous": metrics.capex_previous,
        "policy_tailwind_summary": policy_result.get("summary"),
        "policy_tailwind_verdict": policy_verdict,
        "policy_tailwind_score": policy_score_val,
        "policy_tailwind_confidence_raw": policy_conf_val,
        "policy_tailwind_key_points": policy_result.get("key_points") or [],
        "policy_tailwind_source": policy_source,
        "peer_symbols_used": peer_symbols,
        "dynamic_thresholds": {
            "growth": growth_thresholds,
            "margin_trend": margin_thresholds,
            "capex_cycle": capex_thresholds,
        },
        "factor_scores": factor_scores,
        "weights_used": applied_weights,
        "weight_denominator": weight_denominator,
    }
    confidence_component = policy_conf_val / 5.0 if policy_conf_val is not None else None
    if confidence_component is not None:
        components["confidence"] = confidence_component

    return FactorResult(
        score=score,
        status=status,
        summary=summary_text,
        key_evidence=policy_result.get("key_points") or [],
        sources=policy_result.get("sources") or [],
        components=components,
        errors=errors,
        weight_denominator=weight_denominator,
    )
