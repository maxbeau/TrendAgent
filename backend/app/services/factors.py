"""AION factor calculators (F1/F3/F4/F5/F6/F8) using available providers."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from app.services.providers import MassiveProvider, YFinanceProvider
from app.services.providers.base import ProviderError
from app.services.providers.fred import FREDProvider
from app.services.risk_reward import summarize_risk_reward


# ---------- Common helpers ----------


def _bucketize(
    value: Optional[float],
    thresholds: Sequence[float],
    *,
    higher_is_better: bool = True,
) -> Optional[int]:
    """Map a numeric value into 1-5 buckets based on monotonically increasing thresholds."""
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None

    score = 1
    for threshold in thresholds:
        if (value >= threshold and higher_is_better) or (value <= threshold and not higher_is_better):
            score += 1
    return min(score, 5)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None and math.isfinite(v)]
    return sum(nums) / len(nums) if nums else None


def _weighted_average(
    scores: Dict[str, Optional[float]], weights: Dict[str, float]
) -> tuple[Optional[float], float, Dict[str, float]]:
    """
    Calculate weighted average of factor scores using configured weights.
    Missing weights or scores are ignored to avoid skewing the denominator.
    """
    weighted_sum = 0.0
    weight_denominator = 0.0
    applied_weights: Dict[str, float] = {}

    for code, score in scores.items():
        weight = weights.get(code)
        if weight is None:
            continue
        weight_value = float(weight)
        if weight_value <= 0:
            continue
        applied_weights[code] = weight_value
        if score is None or not math.isfinite(score):
            continue
        weighted_sum += score * weight_value
        weight_denominator += weight_value

    averaged = weighted_sum / weight_denominator if weight_denominator > 0 else None
    return averaged, weight_denominator, applied_weights


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _latest_numeric(observations: List[Dict[str, Any]], *, count: int = 1) -> List[float]:
    values: List[float] = []
    for obs in reversed(observations or []):
        v = _safe_float(obs.get("value"))
        if v is None:
            continue
        values.append(v)
        if len(values) >= count:
            break
    return values


def _records_to_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert provider financial records (list of dicts with line-item rows) into a DataFrame
    indexed by metric names, columns sorted by period desc.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "date" in df.columns:
        df = df.set_index("date")

    def _col_key(col: Any) -> Any:
        try:
            return pd.to_datetime(col)
        except Exception:
            return col

    sorted_cols = sorted(list(df.columns), key=_col_key, reverse=True)
    return df[sorted_cols]


def _latest_values(frame: pd.DataFrame, keys: Sequence[str], count: int = 2) -> List[float]:
    if frame.empty:
        return []

    key_map = {str(idx).lower(): idx for idx in frame.index}
    target_idx: Optional[Any] = None
    for key in keys:
        if key.lower() in key_map:
            target_idx = key_map[key.lower()]
            break
    if target_idx is None:
        return []

    row = frame.loc[target_idx]
    values: List[float] = []
    for col in frame.columns:
        val = _safe_float(row.get(col))
        if val is None:
            continue
        values.append(val)
        if len(values) >= count:
            break
    return values


def _calc_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if series is None or series.empty or len(series) <= period:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if math.isfinite(val) else None


def _zscore(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    mean = series.mean()
    std = series.std()
    if std == 0 or std is None or not math.isfinite(std):
        return None
    val = (series.iloc[-1] - mean) / std
    return float(val) if math.isfinite(val) else None


@dataclass
class FactorResult:
    score: Optional[float]
    status: str
    components: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    weight_denominator: float = 0.0


# ---------- Factor calculators ----------


async def compute_macro(
    *,
    fred: Optional[FREDProvider] = None,
    window_years: int = 2,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    fred_provider = fred or FREDProvider()
    end = date.today()
    start = end - timedelta(days=window_years * 365)
    weights = factor_weights or {}

    series_ids = {
        "yield_curve": "T10Y2Y",
        "high_yield_spread": "BAMLH0A0HYM2",
        "fed_funds": "FEDFUNDS",
    }
    values: Dict[str, Any] = {}
    errors: List[str] = []

    for name, sid in series_ids.items():
        try:
            observations = await fred_provider.fetch_series(sid, start=start, end=end)
            latest_two = _latest_numeric(observations, count=2)
            values[name] = latest_two
        except ProviderError as exc:  # pragma: no cover - network
            errors.append(f"{name}: {exc}")
            values[name] = []

    spread = values.get("yield_curve", [])
    hy_spread = values.get("high_yield_spread", [])
    fed = values.get("fed_funds", [])

    spread_score = _bucketize(spread[0], [-1.0, 0.0, 0.5, 1.0], higher_is_better=True) if spread else None
    hy_score = _bucketize(hy_spread[0], [2.0, 3.0, 4.0, 6.0], higher_is_better=False) if hy_spread else None
    fed_delta = fed[0] - fed[1] if len(fed) >= 2 else None
    fed_score = _bucketize(-fed_delta if fed_delta is not None else None, [0.0, 0.25, 0.5, 1.0])

    factor_scores = {
        "macro.liquidity_direction": spread_score,
        "macro.credit_spread": hy_score,
        "macro.rate_trend": fed_score,
        "macro.global_demand": None,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "error" if errors else "unavailable"

    return FactorResult(
        score=score,
        status=status,
        components={
            "yield_curve": spread[0] if spread else None,
            "high_yield_spread": hy_spread[0] if hy_spread else None,
            "fed_funds_latest": fed[0] if fed else None,
            "fed_funds_change": fed_delta,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )


async def compute_fundamental(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    provider = yfinance or YFinanceProvider()
    weights = factor_weights or {}
    try:
        financials = await provider.fetch_financials(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        return FactorResult(score=None, status="error", errors=[str(exc)])

    income = _records_to_frame(financials.get("income_statement", []))
    balance = _records_to_frame(financials.get("balance_sheet", []))
    cash_flow = _records_to_frame(financials.get("cash_flow", []))

    rev = _latest_values(income, ["Total Revenue", "TotalRevenue", "Total Revenue USD"])
    op_inc = _latest_values(income, ["Operating Income", "OperatingIncome"])
    net_inc = _latest_values(income, ["Net Income", "NetIncome"])
    ebitda = _latest_values(income, ["EBITDA", "Ebitda"])
    cash = _latest_values(balance, ["Cash And Cash Equivalents", "Cash"], count=1)
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
    if len(rev) >= 2 and rev[1] != 0:
        growth = (rev[0] - rev[1]) / rev[1]

    op_margin = None
    if op_inc and rev and rev[0] != 0:
        op_margin = op_inc[0] / rev[0]

    cf_quality = None
    if fcf_final and net_inc and net_inc[0]:
        cf_quality = fcf_final[0] / net_inc[0]

    leverage = None
    debt_total = (debt_lt[0] if debt_lt else 0.0) + (debt_st[0] if debt_st else 0.0)
    cash_val = cash[0] if cash else 0.0
    if ebitda and ebitda[0]:
        leverage = (debt_total - cash_val) / ebitda[0]

    roic = None
    if op_inc:
        invested = None
        if invested_capital:
            invested = invested_capital[0]
            if current_liab:
                invested -= current_liab[0]
    if invested and invested != 0:
        roic = op_inc[0] / invested

    factor_scores = {
        "fundamental.growth": _bucketize(growth, [-0.05, 0.0, 0.1, 0.2]),
        "fundamental.profitability": _bucketize(op_margin, [0.05, 0.1, 0.2, 0.3]),
        "fundamental.cash_flow": _bucketize(cf_quality, [0.5, 0.8, 1.0, 1.5]),
        "fundamental.leverage": _bucketize(leverage, [0.5, 1.5, 2.5, 3.5], higher_is_better=False),
        "fundamental.roic": _bucketize(roic, [0.05, 0.1, 0.15, 0.2]),
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "unavailable"

    return FactorResult(
        score=score,
        status=status,
        components={
            "revenue_growth": growth,
            "operating_margin": op_margin,
            "cash_flow_quality": cf_quality,
            "leverage": leverage,
            "roic": roic,
            "raw": {
                "revenue": rev[:2],
                "operating_income": op_inc[:2],
                "net_income": net_inc[:2],
                "ebitda": ebitda[:2],
                "fcf": fcf_final[:2] if fcf_final else [],
            },
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=[],
        weight_denominator=weight_denominator,
    )


async def compute_technical(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    lookback_days: int = 400,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    provider = yfinance or YFinanceProvider()
    end = date.today()
    start = end - timedelta(days=lookback_days)
    weights = factor_weights or {}
    errors: List[str] = []

    try:
        history = await provider.fetch_equity_daily(ticker, start=start, end=end)
    except ProviderError as exc:  # pragma: no cover - network
        return FactorResult(score=None, status="error", errors=[str(exc)])

    if not history:
        return FactorResult(score=None, status="unavailable", components={"message": "no price history"})

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    closes = df["close"]
    volumes = df["volume"]
    df["ma20"] = closes.rolling(window=20).mean()
    df["ma50"] = closes.rolling(window=50).mean()
    df["ma200"] = closes.rolling(window=200).mean()

    latest = df.iloc[-1]
    ma20 = _safe_float(latest.get("ma20"))
    ma50 = _safe_float(latest.get("ma50"))
    ma200 = _safe_float(latest.get("ma200"))

    trend_score = None
    if ma20 and ma50 and ma200:
        if ma20 > ma50 > ma200:
            trend_score = 5
        elif ma20 > ma50 and ma50 > 0.98 * ma200:
            trend_score = 4
        elif ma20 > ma50 or ma50 > ma200:
            trend_score = 3
        else:
            trend_score = 2

    rsi_val = _calc_rsi(closes, period=14)
    rsi_score = _bucketize(rsi_val, [30, 40, 60, 70]) if rsi_val is not None else None

    vol_z = _zscore(volumes.tail(90))
    volume_score = _bucketize(vol_z, [-1.0, 0.0, 1.0, 2.0]) if vol_z is not None else None

    factor_scores = {
        "technical.trend_structure": trend_score,
        "technical.relative_strength": rsi_score,
        "technical.volume_profile": volume_score,
        "technical.volatility_structure": None,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "degraded" if errors else "unavailable"

    return FactorResult(
        score=score,
        status=status,
        components={
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "rsi": rsi_val,
            "volume_z": vol_z,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )


async def compute_flow(
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

    try:
        holders = await y_provider.fetch_holders(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        holders = []
        errors.append(f"holders: {exc}")

    try:
        put_call = await y_provider.fetch_put_call_ratio(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        put_call = {}
        errors.append(f"put_call: {exc}")

    try:
        gex = await y_provider.fetch_gamma_exposure(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        gex = {}
        errors.append(f"gex: {exc}")

    inst_score = _bucketize(len(holders), [1, 3, 5, 8])
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

    return FactorResult(
        score=score,
        status=status,
        components={
            "institutional_count": len(holders),
            "put_call": put_call,
            "gamma_exposure": gex,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )


async def compute_sentiment(
    ticker: str,
    *,
    yfinance: Optional[YFinanceProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    provider = yfinance or YFinanceProvider()
    weights = factor_weights or {}
    errors: List[str] = []

    try:
        vix_history = await provider.fetch_equity_daily("^VIX", start=date.today() - timedelta(days=60), end=date.today())
        vix_close = vix_history[-1]["close"] if vix_history else None
    except ProviderError as exc:  # pragma: no cover - network
        vix_close = None
        errors.append(f"vix: {exc}")

    try:
        put_call = await provider.fetch_put_call_ratio(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        put_call = {}
        errors.append(f"put_call: {exc}")

    try:
        skew = await provider.fetch_vol_skew(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        skew = {}
        errors.append(f"skew: {exc}")

    vix_score = _bucketize(vix_close, [12, 18, 24, 32]) if vix_close is not None else None
    pcr_score = _bucketize(put_call.get("put_call_ratio"), [0.7, 0.9, 1.1, 1.3]) if put_call else None
    skew_score = _bucketize(skew.get("skew_25d"), [0.0, 0.05, 0.1, 0.2]) if skew else None

    factor_scores = {
        "sentiment.vix": vix_score,
        "sentiment.put_call": pcr_score,
        "sentiment.skew": skew_score,
        "sentiment.fear_greed": None,
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
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )


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

    try:
        iv_hv = await y_provider.fetch_iv_hv(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        iv_hv = {}
        errors.append(f"iv_hv: {exc}")

    try:
        skew = await y_provider.fetch_vol_skew(ticker)
    except ProviderError as exc:  # pragma: no cover - network
        skew = {}
        errors.append(f"skew: {exc}")

    try:
        rr = await summarize_risk_reward(ticker, massive=massive_provider)
    except ProviderError as exc:  # pragma: no cover - network
        rr = {"status": "unavailable", "message": str(exc)}
        errors.append(f"risk_reward: {exc}")

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

    return FactorResult(
        score=score,
        status=status,
        components={
            "iv_vs_hv": iv_vs_hv,
            "skew": skew,
            "risk_reward": rr,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )


# ---------- Orchestration helper ----------


def _weights_for(prefix: str, weights: Dict[str, float]) -> Dict[str, float]:
    """Filter factor weights by prefix to route them to the correct category calculator."""
    return {code: weight for code, weight in weights.items() if code.startswith(prefix)}


async def compute_all_factors(
    ticker: str,
    *,
    factor_weights: Optional[Dict[str, float]],
    yfinance: Optional[YFinanceProvider] = None,
    massive: Optional[MassiveProvider] = None,
    fred: Optional[FREDProvider] = None,
) -> Dict[str, FactorResult]:
    """
    Kick off all factor computations in parallel.
    """
    if not factor_weights:
        raise ValueError("factor_weights are required for computing AION factor scores.")

    y_provider = yfinance or YFinanceProvider()
    massive_provider = massive or MassiveProvider()
    fred_provider = fred or FREDProvider()

    macro_weights = _weights_for("macro.", factor_weights)
    fundamental_weights = _weights_for("fundamental.", factor_weights)
    technical_weights = _weights_for("technical.", factor_weights)
    flow_weights = _weights_for("flow.", factor_weights)
    sentiment_weights = _weights_for("sentiment.", factor_weights)
    volatility_weights = _weights_for("volatility.", factor_weights)

    tasks = {
        "F1": compute_macro(fred=fred_provider, factor_weights=macro_weights),
        "F3": compute_fundamental(ticker, yfinance=y_provider, factor_weights=fundamental_weights),
        "F4": compute_technical(ticker, yfinance=y_provider, factor_weights=technical_weights),
        "F5": compute_flow(ticker, yfinance=y_provider, massive=massive_provider, factor_weights=flow_weights),
        "F6": compute_sentiment(ticker, yfinance=y_provider, factor_weights=sentiment_weights),
        "F8": compute_volatility(
            ticker, yfinance=y_provider, massive=massive_provider, factor_weights=volatility_weights
        ),
    }

    results: Dict[str, FactorResult] = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, res in zip(tasks.keys(), gathered):
        if isinstance(res, Exception):
            results[key] = FactorResult(score=None, status="error", errors=[str(res)])
        else:
            results[key] = res
    return results
