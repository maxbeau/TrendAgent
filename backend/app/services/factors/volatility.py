from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, List

from app.services.factors.core import (
    _bucketize,
    _build_expected_move_payload,
    _fetch_with_default,
    _weighted_average,
    FactorResult,
)
from app.services.providers import MassiveProvider, YFinanceProvider
from app.services.providers.base import ProviderError
from app.services.risk_reward import summarize_risk_reward


async def compute_volatility(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    massive: Optional[MassiveProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    y_provider = yfinance or YFinanceProvider()
    massive_provider = massive or MassiveProvider()
    weights = factor_weights or {}
    errors: List[str] = []

    iv_task = asyncio.create_task(_fetch_with_default(y_provider.fetch_iv_hv(ticker), "iv_hv", {}))
    skew_task = asyncio.create_task(_fetch_with_default(y_provider.fetch_vol_skew(ticker), "skew", {}))
    rr_task = asyncio.create_task(
        _fetch_with_default(
            summarize_risk_reward(ticker, massive=massive_provider),
            "risk_reward",
            {"status": "unavailable", "ratio": None},
        )
    )
    (iv_hv, iv_error), (skew, skew_error), (rr, rr_error) = await asyncio.gather(
        iv_task,
        skew_task,
        rr_task,
    )
    errors.extend(err for err in (iv_error, skew_error, rr_error) if err)

    spot = iv_hv.get("spot")
    atm_iv = iv_hv.get("atm_iv")
    hv = iv_hv.get("hv")
    iv_vs_hv = iv_hv.get("iv_vs_hv")
    skew_val = skew.get("skew_25d")
    rr_ratio = rr.get("ratio")

    iv_score = _bucketize(iv_vs_hv, [-0.05, 0.0, 0.05, 0.1], higher_is_better=False) if iv_vs_hv is not None else None
    skew_score = _bucketize(skew_val, [0.0, 0.05, 0.1, 0.2]) if skew_val is not None else None
    rr_score = _bucketize(rr_ratio, [0.8, 1.0, 1.5, 2.0]) if rr_ratio is not None else None

    factor_scores = {
        "volatility.iv_vs_hv": iv_score,
        "volatility.skew": skew_score,
        "volatility.risk_reward": rr_score,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "degraded" if errors else "unavailable"

    expected_move = _build_expected_move_payload(spot, atm_iv, hv)

    components: Dict[str, Any] = {
        "iv_vs_hv": iv_vs_hv,
        "skew": skew,
        "risk_reward": rr,
        "factor_scores": factor_scores,
        "weights_used": applied_weights,
        "weight_denominator": weight_denominator,
    }
    if expected_move:
        components["expected_move"] = expected_move
    return FactorResult(
        score=score,
        status=status,
        components=components,
        errors=errors,
        weight_denominator=weight_denominator,
    )
