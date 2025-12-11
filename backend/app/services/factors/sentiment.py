from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any, Dict, Optional

from app.services.factors.core import _bucketize, _safe_float, _weighted_average, FactorResult
from app.services.providers import FearGreedIndexProvider, YFinanceProvider
from app.services.providers.base import ProviderError


async def compute_sentiment(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    fear_greed: Optional[FearGreedIndexProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    y_provider = yfinance or YFinanceProvider()
    fear_greed_provider = fear_greed or FearGreedIndexProvider()
    weights = factor_weights or {}
    errors: list[str] = []

    async def _fetch_or_log(name: str, coro, default: Any) -> Any:
        try:
            return await coro
        except ProviderError as exc:  # pragma: no cover - network
            errors.append(f"{name}: {exc}")
            return default

    today = date.today()
    start = today - timedelta(days=60)
    vix_history, put_call, skew, fear_greed_index = await asyncio.gather(
        _fetch_or_log("vix", y_provider.fetch_equity_daily("^VIX", start=start, end=today), []),
        _fetch_or_log("put_call", y_provider.fetch_put_call_ratio(ticker), {}),
        _fetch_or_log("skew", y_provider.fetch_vol_skew(ticker), {}),
        _fetch_or_log("fear_greed", fear_greed_provider.fetch_index(), {}),
    )

    vix_close = vix_history[-1]["close"] if vix_history else None
    fear_greed_value = _safe_float(fear_greed_index.get("score")) if fear_greed_index else None

    metric_configs = [
        ("sentiment.vix", vix_close, [12, 18, 24, 32], True),
        ("sentiment.put_call", put_call.get("put_call_ratio") if put_call else None, [0.7, 0.9, 1.1, 1.3], True),
        ("sentiment.skew", skew.get("skew_25d") if skew else None, [0.0, 0.05, 0.1, 0.2], True),
        ("sentiment.fear_greed", fear_greed_value, [25, 40, 60, 75], False),
    ]
    factor_scores = {
        code: _bucketize(value, thresholds, higher_is_better=higher_is_better)
        if value is not None
        else None
        for code, value, thresholds, higher_is_better in metric_configs
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "degraded" if errors else "unavailable"

    return FactorResult(
        score=score,
        status=status,
        components={
            "vix": vix_close,
            "put_call": put_call,
            "skew": skew,
            "fear_greed": fear_greed_index,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )
