from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.factors.core import (
    _bucketize,
    _relative_change,
    _safe_float,
    _latest_values,
    _records_to_frame,
    _weighted_average,
    FactorResult,
)
from app.services.providers import MassiveProvider, YFinanceProvider
from app.services.providers.base import ProviderError
from app.services.tam_expansion import summarize_tam_expansion


_FUNDAMENTAL_DOC_WEIGHTS: Dict[str, float] = {
    # Reflect AION v8.0 formula weights so compute_fundamental stays aligned with docs.
    "fundamental.growth": 25.0,
    "fundamental.profitability": 25.0,
    "fundamental.cash_flow": 20.0,
    "fundamental.leverage": 15.0,
    "fundamental.tam_expansion": 15.0,
}


@dataclass
class FundamentalMetrics:
    revenue_growth: Optional[float]
    operating_margin: Optional[float]
    previous_margin: Optional[float]
    margin_trend: Optional[float]
    cash_flow_quality: Optional[float]
    leverage: Optional[float]
    roic: Optional[float]
    raw: Dict[str, List[float]]


def _fundamental_metrics_from_statements(statements: Optional[Dict[str, Any]]) -> FundamentalMetrics:
    income = _records_to_frame((statements or {}).get("income_statement", []))
    balance = _records_to_frame((statements or {}).get("balance_sheet", []))
    cash_flow = _records_to_frame((statements or {}).get("cash_flow", []))

    rev = _latest_values(income, ["Total Revenue", "TotalRevenue", "Total Revenue USD"])
    op_inc = _latest_values(income, ["Operating Income", "OperatingIncome"])
    net_inc = _latest_values(income, ["Net Income", "NetIncome"])
    ebitda = _latest_values(income, ["EBITDA", "Ebitda"])
    cash = _latest_values(balance, ["Cash And Cash Equivalents", "Cash and cash equivalents", "Cash"], count=1)
    debt_lt = _latest_values(balance, ["Long Term Debt", "LongTermDebt"], count=1)
    debt_st = _latest_values(balance, ["Short Long Term Debt", "ShortLongTermDebt"], count=1)
    fcf = _latest_values(cash_flow, ["Free Cash Flow", "FreeCashFlow", "Free Cash Flow USD"])
    op_cf = _latest_values(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _latest_values(cash_flow, ["Capital Expenditures", "Capital Expenditure"])
    invested_capital = _latest_values(balance, ["Total Assets", "TotalAssets"])
    current_liab = _latest_values(balance, ["Total Current Liabilities"], count=1)

    fcf_final = fcf
    if not fcf_final and op_cf and capex:
        fcf_final = [op_cf[0] + capex[0]]  # capex is negative in statements

    growth = None
    if len(rev) >= 2 and rev[1]:
        growth = (rev[0] - rev[1]) / rev[1]

    op_margin = None
    op_margin_prev = None
    if op_inc and rev and rev[0]:
        op_margin = op_inc[0] / rev[0]
    if len(op_inc) >= 2 and len(rev) >= 2 and rev[1]:
        op_margin_prev = op_inc[1] / rev[1]

    margin_trend = None
    if op_margin is not None and op_margin_prev is not None:
        margin_trend = _relative_change(op_margin, op_margin_prev) or (op_margin - op_margin_prev)

    cf_quality = None
    if fcf_final and net_inc and net_inc[0]:
        cf_quality = fcf_final[0] / net_inc[0]

    leverage = None
    debt_total = (debt_lt[0] if debt_lt else 0.0) + (debt_st[0] if debt_st else 0.0)
    cash_val = cash[0] if cash else 0.0
    if ebitda and ebitda[0]:
        leverage = (debt_total - cash_val) / ebitda[0]

    roic = None
    invested = None
    if op_inc:
        if invested_capital:
            invested = invested_capital[0]
            if current_liab:
                invested -= current_liab[0]
    if invested and invested != 0:
        roic = op_inc[0] / invested

    return FundamentalMetrics(
        revenue_growth=growth,
        operating_margin=op_margin,
        previous_margin=op_margin_prev,
        margin_trend=margin_trend,
        cash_flow_quality=cf_quality,
        leverage=leverage,
        roic=roic,
        raw={
            "revenue": rev[:2],
            "operating_income": op_inc[:2],
            "net_income": net_inc[:2],
            "ebitda": ebitda[:2],
            "fcf": fcf_final[:2] if fcf_final else [],
        },
    )


@dataclass
class TamExpansionComponent:
    score: Optional[float] = None
    outlook: Optional[str] = None
    confidence: Optional[float] = None
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)


async def _tam_expansion_component(
    ticker: str, massive: Optional[MassiveProvider]
) -> tuple[TamExpansionComponent, Optional[str]]:
    provider = massive or MassiveProvider()
    try:
        result = await summarize_tam_expansion(ticker, massive=provider)
    except Exception as exc:  # pragma: no cover - network/LLM
        return TamExpansionComponent(), f"tam_expansion: {exc}"

    score = _safe_float(result.get("score"))
    if score is not None:
        score = max(1.0, min(score, 5.0))
    confidence = _safe_float(result.get("confidence"))
    component = TamExpansionComponent(
        score=score,
        outlook=result.get("outlook"),
        confidence=confidence,
        summary=result.get("summary"),
        key_points=result.get("key_points") or [],
        sources=result.get("sources") or [],
    )
    return component, None


async def compute_fundamental(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    massive: Optional[MassiveProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
    financials: Optional[Dict[str, Any]] = None,
) -> FactorResult:
    provider = yfinance or YFinanceProvider()
    weights = {**_FUNDAMENTAL_DOC_WEIGHTS, **(factor_weights or {})}
    weights.pop("fundamental.roic", None)
    statements = financials
    if statements is None:
        try:
            statements = await provider.fetch_financials(ticker)
        except ProviderError as exc:  # pragma: no cover - network
            return FactorResult(score=None, status="error", errors=[str(exc)])

    metrics = _fundamental_metrics_from_statements(statements)
    tam_component, tam_error = await _tam_expansion_component(ticker, massive)
    errors: List[str] = [tam_error] if tam_error else []

    factor_scores = {
        "fundamental.growth": _bucketize(metrics.revenue_growth, [-0.05, 0.0, 0.1, 0.2]),
        "fundamental.profitability": _bucketize(metrics.margin_trend, [-0.15, -0.05, 0.02, 0.08]),
        "fundamental.cash_flow": _bucketize(metrics.cash_flow_quality, [0.5, 0.8, 1.0, 1.5]),
        "fundamental.leverage": _bucketize(metrics.leverage, [0.5, 1.5, 2.5, 3.5], higher_is_better=False),
        "fundamental.roic": _bucketize(metrics.roic, [0.05, 0.1, 0.15, 0.2]),
        "fundamental.tam_expansion": tam_component.score,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else ("degraded" if errors else "unavailable")

    return FactorResult(
        score=score,
        status=status,
        summary=tam_component.summary,
        key_evidence=tam_component.key_points,
        sources=tam_component.sources,
        components={
            "revenue_growth": metrics.revenue_growth,
            "operating_margin": metrics.operating_margin,
            "operating_margin_previous": metrics.previous_margin,
            "operating_margin_trend": metrics.margin_trend,
            "cash_flow_quality": metrics.cash_flow_quality,
            "leverage": metrics.leverage,
            "roic": metrics.roic,
            "tam_expansion": {
                "score": tam_component.score,
                "outlook": tam_component.outlook,
                "confidence": tam_component.confidence,
                "summary": tam_component.summary,
                "key_points": tam_component.key_points,
            },
            "raw": metrics.raw,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )
