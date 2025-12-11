from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from app.services.factors.core import (
    _bucketize,
    _ohlcv_records_to_df,
    _rs_rating,
    _safe_float,
    _volume_profile,
    _weighted_average,
    FactorResult,
)
from app.services.providers import YFinanceProvider
from app.services.providers.base import ProviderError


def _summarize_technical(
    latest_close: Optional[float],
    ma20: Optional[float],
    ma50: Optional[float],
    ma200: Optional[float],
    rs_metrics: Optional[Dict[str, float]],
    price_vs_poc: Optional[float],
) -> str:
    parts: List[str] = []
    if ma20 and ma50 and ma200:
        if ma20 > ma50 > ma200:
            parts.append("均线多头排列")
        elif ma20 < ma50 < ma200:
            parts.append("均线空头排列")
        else:
            parts.append("均线结构混沌")

    if rs_metrics and rs_metrics.get("rs") is not None:
        rs_value = rs_metrics["rs"]
        if rs_value >= 0.05:
            parts.append(f"跑赢基准 {rs_value*100:.1f}%")
        elif rs_value <= -0.05:
            parts.append(f"落后基准 {abs(rs_value)*100:.1f}%")
        else:
            parts.append("相对强度中性")

    if price_vs_poc is not None:
        diff_pct = price_vs_poc * 100
        if price_vs_poc >= 0.05:
            parts.append(f"价格高于筹码 {diff_pct:.1f}%")
        elif price_vs_poc <= -0.05:
            parts.append(f"价格低于筹码 {abs(diff_pct):.1f}%")
        else:
            parts.append("价格接近成本区")

    if not parts and latest_close is not None:
        parts.append(f"最新收盘 ${latest_close:.2f}")
    return "；".join(parts)


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
    benchmark_symbol = "SPY"
    weights = factor_weights or {}
    errors: List[str] = []

    try:
        history = await provider.fetch_equity_daily(ticker, start=start, end=end)
    except ProviderError as exc:  # pragma: no cover - network
        return FactorResult(score=None, status="error", errors=[str(exc)])

    try:
        benchmark_history = await provider.fetch_equity_daily(benchmark_symbol, start=start, end=end)
    except ProviderError as exc:  # pragma: no cover - network
        benchmark_history = []
        errors.append(f"benchmark: {exc}")

    df = _ohlcv_records_to_df(history)
    if df.empty:
        return FactorResult(score=None, status="unavailable", components={"message": "no price history"})

    bench_df = _ohlcv_records_to_df(benchmark_history)
    if bench_df.empty:
        errors.append("benchmark: no price history")

    closes = df["close"].astype(float)
    df["ma20"] = closes.rolling(window=20).mean()
    df["ma50"] = closes.rolling(window=50).mean()
    df["ma200"] = closes.rolling(window=200).mean()

    latest = df.iloc[-1]
    latest_close = _safe_float(latest.get("close"))
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

    rs_metrics = _rs_rating(df, bench_df) if not bench_df.empty else None
    rs_value = rs_metrics.get("rs") if rs_metrics else None
    rs_score = _bucketize(rs_value, [-0.05, -0.01, 0.01, 0.05]) if rs_value is not None else None

    vp_metrics = _volume_profile(df)
    poc_price = _safe_float(vp_metrics.get("top_price")) if vp_metrics else None
    price_vs_poc = None
    if latest_close is not None and poc_price is not None and poc_price != 0:
        price_vs_poc = (latest_close - poc_price) / abs(poc_price)
    volume_score = _bucketize(price_vs_poc, [-0.05, -0.02, 0.02, 0.05]) if price_vs_poc is not None else None

    factor_scores = {
        "technical.trend_structure": trend_score,
        "technical.relative_strength": rs_score,
        "technical.volume_profile": volume_score,
        "technical.volatility_structure": None,
    }
    score, weight_denominator, applied_weights = _weighted_average(factor_scores, weights)
    status = "ok" if score is not None else "degraded" if errors else "unavailable"

    summary_text = _summarize_technical(latest_close, ma20, ma50, ma200, rs_metrics, price_vs_poc)

    return FactorResult(
        score=score,
        status=status,
        summary=summary_text,
        components={
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "benchmark_symbol": benchmark_symbol,
            "relative_strength": rs_metrics,
            "volume_profile": vp_metrics,
            "price_vs_poc": price_vs_poc,
            "factor_scores": factor_scores,
            "weights_used": applied_weights,
            "weight_denominator": weight_denominator,
        },
        errors=errors,
        weight_denominator=weight_denominator,
    )
