from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.factors.catalyst import compute_soft_factor
from app.services.factors.fundamental import compute_fundamental
from app.services.factors.industry import compute_industry
from app.services.factors.macro import compute_macro
from app.services.factors.sentiment import compute_sentiment
from app.services.factors.technical import compute_technical
from app.services.factors.volatility import compute_volatility
from app.services.factors.flow import compute_flow
from app.services.factors.core import FactorResult
from app.services.providers import (
    FMPProvider,
    FearGreedIndexProvider,
    MassiveProvider,
    YFinanceProvider,
)
from app.services.providers.fred import FREDProvider
from app.services.providers.base import ProviderError
import logging

logger = logging.getLogger(__name__)


def _weights_for(prefix: str, weights: Dict[str, float]) -> Dict[str, float]:
    """Filter factor weights by prefix to route them to the correct category calculator."""
    return {code: weight for code, weight in weights.items() if code.startswith(prefix)}


async def compute_all_factors(
    ticker: str,
    *,
    factor_weights: Optional[Dict[str, float]],
    yfinance: Optional[YFinanceProvider] = None,
    fear_greed: Optional[FearGreedIndexProvider] = None,
    massive: Optional[MassiveProvider] = None,
    fmp: Optional[FMPProvider] = None,
    db: Optional[AsyncSession] = None,
) -> Dict[str, FactorResult]:
    """
    Kick off all factor computations in parallel.
    """
    if not factor_weights:
        raise ValueError("factor_weights are required for computing AION factor scores.")

    y_provider = yfinance or YFinanceProvider()
    fear_greed_provider = fear_greed or FearGreedIndexProvider()
    massive_provider = massive or MassiveProvider()
    fmp_provider = fmp or FMPProvider()
    fred_provider = FREDProvider()

    macro_weights = _weights_for("macro.", factor_weights)
    fundamental_weights = _weights_for("fundamental.", factor_weights)
    technical_weights = _weights_for("technical.", factor_weights)
    flow_weights = _weights_for("flow.", factor_weights)
    sentiment_weights = _weights_for("sentiment.", factor_weights)
    volatility_weights = _weights_for("volatility.", factor_weights)
    industry_weights = _weights_for("industry.", factor_weights)
    catalyst_weights = _weights_for("catalyst.", factor_weights)

    preloaded_financials: Optional[Dict[str, Any]] = None
    try:
        preloaded_financials = await y_provider.fetch_financials(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        logger.warning(
            "financial_prefetch_failed",
            extra={"ticker": ticker, "error": str(exc)},
        )

    tasks = {
        "F1": compute_macro(fred=fred_provider, factor_weights=macro_weights),
        "F2": compute_industry(
            ticker,
            yfinance=y_provider,
            massive=massive_provider,
            factor_weights=industry_weights,
            financials=preloaded_financials,
            db=db,
        ),
        "F3": compute_fundamental(
            ticker,
            yfinance=y_provider,
            massive=massive_provider,
            factor_weights=fundamental_weights,
            financials=preloaded_financials,
        ),
        "F4": compute_technical(ticker, yfinance=y_provider, factor_weights=technical_weights),
        "F5": compute_flow(
            ticker,
            yfinance=y_provider,
            massive=massive_provider,
            fmp=fmp_provider,
            factor_weights=flow_weights,
        ),
        "F6": compute_sentiment(
            ticker,
            yfinance=y_provider,
            fear_greed=fear_greed_provider,
            factor_weights=sentiment_weights,
        ),
        "F7": compute_soft_factor(
            ticker,
            "F7",
            db=db,
            massive=massive_provider,
            yfinance=y_provider,
            factor_weights=catalyst_weights,
        ),
        "F8": compute_volatility(
            ticker, yfinance=y_provider, massive=massive_provider, factor_weights=volatility_weights
        ),
    }

    results: Dict[str, FactorResult] = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, res in zip(tasks.keys(), gathered):
        if isinstance(res, Exception):
            logger.warning("factor_compute_failed", extra={"factor": key, "ticker": ticker, "error": str(res)})
            results[key] = FactorResult(score=None, status="error", errors=[str(res)])
        else:
            results[key] = res
    return results
