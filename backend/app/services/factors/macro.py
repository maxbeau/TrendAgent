from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from app.config import MacroThresholds, get_settings
from app.services.factors.core import (
    _bucketize,
    _latest_series_value,
    _mean,
    _net_liquidity_series,
    _percentile_thresholds_from_values,
    _quantile_thresholds,
    _series_from_massive,
    _series_from_observations,
    _weighted_average,
    FactorResult,
)
from app.services.providers import FearGreedIndexProvider, MassiveProvider
from app.services.providers.fred import FREDProvider

settings = get_settings()

_MACRO_SERIES_IDS: Dict[str, str] = {
    "yield_curve": "T10Y2Y",
    "high_yield_spread": "BAMLH0A0HYM2",
    "fed_funds": "EFFR",
    "fed_total_assets": "WALCL",
    "treasury_general_account": "WTREGEN",
    "reverse_repo": "RRPONTSYD",
    "industrial_production": "INDPRO",
}
_RATE_EXPECTATION_TASK = "rate_expectations"


async def _collect_macro_time_series(
    fred_provider: FREDProvider,
    massive_provider: MassiveProvider,
    *,
    start: date,
    end: date,
    window_years: int,
) -> tuple[pd.DataFrame, List[str]]:
    errors: List[str] = []
    raw_series: Dict[str, List[Dict[str, Any]]] = {}

    fred_tasks = [
        (name, asyncio.create_task(fred_provider.fetch_series(series_id, start=start, end=end)))
        for name, series_id in _MACRO_SERIES_IDS.items()
    ]
    expectation_task = asyncio.create_task(
        massive_provider.fetch_inflation_expectations(horizon="1y", limit=max(window_years * 24, 24))
    )
    task_entries = fred_tasks + [(_RATE_EXPECTATION_TASK, expectation_task)]
    task_results = await asyncio.gather(*(task for _, task in task_entries), return_exceptions=True)

    expectation_records: List[Dict[str, Any]] = []
    for (name, _task), result in zip(task_entries, task_results):
        if isinstance(result, Exception):
            errors.append(f"{name}: {result}")
            continue
        if name == _RATE_EXPECTATION_TASK:
            expectation_records = list(result)
        else:
            raw_series[name] = list(result)

    series_map: Dict[str, pd.Series] = {}
    for name, observations in raw_series.items():
        series = _series_from_observations(observations)
        if not series.empty:
            series_map[name] = series
    expectation_series = _series_from_massive(expectation_records)
    if not expectation_series.empty:
        series_map["fed_expectations"] = expectation_series

    frame = pd.DataFrame(series_map).sort_index() if series_map else pd.DataFrame()
    if not frame.empty:
        frame = frame.ffill()
    return frame, errors


@dataclass
class MacroSnapshot:
    latest: Dict[str, Optional[float]]
    threshold_series: Dict[str, Optional[pd.Series]]
    rate_expectation_source: str


@dataclass
class MacroScoreResult:
    factor_scores: Dict[str, Optional[int]]
    net_liquidity_score: Optional[int]
    spread_score: Optional[int]
    hy_score: Optional[int]
    fed_score: Optional[int]
    rate_trend_score: Optional[int]
    global_demand_score: Optional[int]


def _build_macro_snapshot(frame: pd.DataFrame) -> MacroSnapshot:
    yield_curve_series = frame.get("yield_curve")
    hy_spread_series = frame.get("high_yield_spread")
    fed_funds_series = frame.get("fed_funds")
    assets_series = frame.get("fed_total_assets")
    tga_series = frame.get("treasury_general_account")
    rrp_series = frame.get("reverse_repo")
    indpro_series = frame.get("industrial_production")

    net_liquidity_series = _net_liquidity_series(assets_series, tga_series, rrp_series)
    net_liquidity_change_series = net_liquidity_series.diff() if net_liquidity_series is not None else None

    expectation_series_aligned = frame.get("fed_expectations")
    rate_expectation_series = expectation_series_aligned if expectation_series_aligned is not None else fed_funds_series
    rate_expectation_source = "massive_inflation_expectations" if expectation_series_aligned is not None else "fred_fed_funds"

    rate_expectation_change_series = rate_expectation_series.diff() if rate_expectation_series is not None else None
    rate_easing_series = -rate_expectation_change_series if rate_expectation_change_series is not None else None
    indpro_yoy_series = indpro_series.pct_change(periods=12) if indpro_series is not None else None

    latest_values = {
        "yield_curve": _latest_series_value(yield_curve_series),
        "high_yield_spread": _latest_series_value(hy_spread_series),
        "fed_funds_latest": _latest_series_value(fed_funds_series),
        "fed_funds_change": _latest_series_value(fed_funds_series.diff() if fed_funds_series is not None else None),
        "fed_funds_expectation_latest": _latest_series_value(rate_expectation_series),
        "fed_funds_expectation_change": _latest_series_value(rate_expectation_change_series),
        "rate_easing": _latest_series_value(rate_easing_series),
        "fed_total_assets": _latest_series_value(assets_series),
        "treasury_general_account": _latest_series_value(tga_series),
        "reverse_repo": _latest_series_value(rrp_series),
        "net_liquidity_latest": _latest_series_value(net_liquidity_series),
        "net_liquidity_change": _latest_series_value(net_liquidity_change_series),
        "industrial_production_yoy": _latest_series_value(indpro_yoy_series),
    }

    threshold_series = {
        "net_liquidity_change": net_liquidity_change_series,
        "yield_curve": yield_curve_series,
        "credit_spread": hy_spread_series,
        "rate_expectations": rate_easing_series,
        "global_demand": indpro_yoy_series,
    }

    return MacroSnapshot(
        latest=latest_values,
        threshold_series=threshold_series,
        rate_expectation_source=rate_expectation_source,
    )


def _resolve_macro_thresholds(
    series_map: Mapping[str, Optional[pd.Series]],
    defaults: MacroThresholds,
) -> Dict[str, List[float]]:
    thresholds: Dict[str, List[float]] = {}
    for key, series in series_map.items():
        fallback = list(getattr(defaults, key))
        thresholds[key] = _quantile_thresholds(series) or fallback
    return thresholds


def _calculate_macro_scores(
    snapshot: MacroSnapshot,
    thresholds: Mapping[str, Sequence[float]],
) -> MacroScoreResult:
    net_liquidity_score = _bucketize(
        snapshot.latest.get("net_liquidity_change"),
        thresholds.get("net_liquidity_change", ()),
        higher_is_better=True,
    )
    spread_score = _bucketize(
        snapshot.latest.get("yield_curve"),
        thresholds.get("yield_curve", ()),
        higher_is_better=True,
    )
    hy_score = _bucketize(
        snapshot.latest.get("high_yield_spread"),
        thresholds.get("credit_spread", ()),
        higher_is_better=False,
    )
    fed_score = _bucketize(
        snapshot.latest.get("rate_easing"),
        thresholds.get("rate_expectations", ()),
        higher_is_better=True,
    )
    rate_trend_score = _mean([spread_score, fed_score])
    global_demand_score = _bucketize(
        snapshot.latest.get("industrial_production_yoy"),
        thresholds.get("global_demand", ()),
    )

    factor_scores = {
        "macro.liquidity_direction": net_liquidity_score,
        "macro.credit_spread": hy_score,
        "macro.rate_trend": rate_trend_score,
        "macro.global_demand": global_demand_score,
    }

    return MacroScoreResult(
        factor_scores=factor_scores,
        net_liquidity_score=net_liquidity_score,
        spread_score=spread_score,
        hy_score=hy_score,
        fed_score=fed_score,
        rate_trend_score=rate_trend_score,
        global_demand_score=global_demand_score,
    )


async def compute_macro(
    *,
    fred: Optional[FREDProvider] = None,
    massive: Optional[MassiveProvider] = None,
    window_years: int = 2,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    """
    Compute AION Macro factors from FRED time series:
    - Net liquidity: Fed total assets - (TGA + RRP) change
    - Credit spread: High yield OAS level
    - Rate trend: Blend of 10Y-2Y curve shape and fed funds drift
    - Global demand: INDPRO year-over-year change
    """
    fred_provider = fred or FREDProvider()
    massive_provider = massive or MassiveProvider()
    end = date.today()
    start = end - timedelta(days=window_years * 365)
    weights = factor_weights or {}

    frame, errors = await _collect_macro_time_series(
        fred_provider,
        massive_provider,
        start=start,
        end=end,
        window_years=window_years,
    )
    snapshot = _build_macro_snapshot(frame)
    thresholds = _resolve_macro_thresholds(snapshot.threshold_series, settings.macro_thresholds)
    score_result = _calculate_macro_scores(snapshot, thresholds)
    score, weight_denominator, applied_weights = _weighted_average(score_result.factor_scores, weights)
    status = "ok" if score is not None else "error" if errors else "unavailable"

    return FactorResult(
        score=score,
        status=status,
        components={
            "yield_curve": snapshot.latest.get("yield_curve"),
            "yield_curve_score": score_result.spread_score,
            "factor_scores": score_result.factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
            "high_yield_spread": snapshot.latest.get("high_yield_spread"),
            "fed_funds_latest": snapshot.latest.get("fed_funds_latest"),
            "fed_funds_change": snapshot.latest.get("fed_funds_change"),
            "fed_funds_expectation_latest": snapshot.latest.get("fed_funds_expectation_latest"),
            "fed_funds_expectation_change": snapshot.latest.get("fed_funds_expectation_change"),
            "rate_expectation_source": snapshot.rate_expectation_source,
            "fed_total_assets": snapshot.latest.get("fed_total_assets"),
            "treasury_general_account": snapshot.latest.get("treasury_general_account"),
            "reverse_repo": snapshot.latest.get("reverse_repo"),
            "net_liquidity_latest": snapshot.latest.get("net_liquidity_latest"),
            "net_liquidity_change": snapshot.latest.get("net_liquidity_change"),
            "industrial_production_yoy": snapshot.latest.get("industrial_production_yoy"),
            "thresholds": thresholds,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )
