"""AION factor calculators (F1-F8) using available providers/soft factors."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Awaitable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from app.config import MacroThresholds, get_settings
from app.services.providers import (
    FMPProvider,
    FearGreedIndexProvider,
    MassiveProvider,
    YFinanceProvider,
)
from app.services.providers.base import ProviderError
from app.services.providers.fred import FREDProvider
from app.services.risk_reward import summarize_risk_reward
from app.services.policy_tailwind import summarize_policy_tailwind
from app.services.soft_factors import (
    score_soft_factor,
    normalize_summary,
    normalize_key_evidence,
    fetch_soft_factor_score,
    upsert_soft_factor_score,
)
from app.services.tam_expansion import summarize_tam_expansion
from app.models import SoftFactorScore
from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)
settings = get_settings()


def _soft_result(
    score: Optional[float],
    *,
    status: str,
    summary: Optional[str],
    key_evidence: Optional[List[str]],
    sources: Optional[List[Dict[str, Any]]] = None,
    asof_date: Optional[str],
    confidence: Optional[float],
    errors: List[str],
    weights: Optional[Dict[str, float]] = None,
    weight_denominator: float = 0.0,
    factor_scores: Optional[Dict[str, Optional[float]]] = None,
    extra_components: Optional[Dict[str, Any]] = None,
) -> "FactorResult":
    sanitized_summary = normalize_summary(summary)
    if isinstance(key_evidence, list):
        sanitized_key_evidence = [str(item) for item in key_evidence if item is not None]
    elif key_evidence is None:
        sanitized_key_evidence = []
    else:
        sanitized_key_evidence = [str(key_evidence)]
    sanitized_sources = sources or []
    components_payload = {
        "asof_date": asof_date,
        "confidence": confidence,
        "weights_used": weights or {},
        "weight_denominator": weight_denominator,
        "factor_scores": factor_scores or {},
    }
    if extra_components:
        components_payload.update(extra_components)
    return FactorResult(
        score=score,
        status=status,
        summary=sanitized_summary,
        key_evidence=sanitized_key_evidence,
        sources=sanitized_sources,
        components=components_payload,
        errors=errors,
        weight_denominator=weight_denominator,
    )


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
    if not thresholds:
        return None
    score = 1
    for threshold in thresholds:
        if (value >= threshold and higher_is_better) or (value <= threshold and not higher_is_better):
            score += 1
    return min(score, 5)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None and math.isfinite(v)]
    return sum(nums) / len(nums) if nums else None


def _relative_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """
    Relative change helper used for trend-style metrics (YoY/QoQ growth).
    """
    if current is None or previous is None:
        return None
    denom = abs(previous)
    if denom == 0:
        return None
    change = (current - previous) / denom
    return float(change) if math.isfinite(change) else None


def _percentile_thresholds_from_values(
    values: Iterable[Optional[float]],
    fallback: Sequence[float],
    *,
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    min_samples: int = 5,
) -> List[float]:
    """
    Build monotonic thresholds from peer values; fall back to static defaults when data is sparse.
    """
    clean = [
        float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)
    ]
    if len(clean) < max(min_samples, len(quantiles) + 1):
        return list(fallback)
    try:
        series = pd.Series(clean)
        quantile_values = series.quantile(list(quantiles))
        thresholds = [float(v) for v in quantile_values.tolist()]
    except Exception:
        return list(fallback)
    if any(math.isnan(val) for val in thresholds):
        return list(fallback)
    return thresholds


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


def _sanitize_weights(raw_weights: Optional[Dict[str, float]]) -> tuple[Dict[str, float], float]:
    """
    Keep only positive numeric weights and return their sum for denominator calculations.
    """
    applied: Dict[str, float] = {}
    for code, weight in (raw_weights or {}).items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(w) or w <= 0:
            continue
        applied[code] = w
    return applied, sum(applied.values())


def _factor_scores_from(score: Optional[float], weights: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Mirror the main score to all weighted sub-keys for soft factors."""
    return {code: score for code in weights}


def _expected_move_range(
    spot: Optional[float],
    vol: Optional[float],
    *,
    days: int = 30,
) -> Optional[Dict[str, Any]]:
    """Estimate a 1-sigma price range over the provided horizon."""
    if spot is None or vol is None:
        return None
    if not isinstance(spot, (int, float)) or not isinstance(vol, (int, float)):
        return None
    if not math.isfinite(spot) or not math.isfinite(vol) or spot <= 0 or vol <= 0:
        return None
    horizon = max(days, 1)
    move_pct = vol * math.sqrt(horizon / 252)
    move_abs = spot * move_pct
    return {
        "spot": spot,
        "days": horizon,
        "vol_used": vol,
        "move_pct": move_pct,
        "move_abs": move_abs,
        "lower": spot - move_abs,
        "upper": spot + move_abs,
    }


def _build_expected_move_payload(
    spot: Optional[float],
    atm_iv: Optional[float],
    hv: Optional[float],
) -> Dict[str, Any]:
    """Return IV/HV 1-sigma ranges when they can be computed."""
    payload: Dict[str, Any] = {}
    iv_move = _expected_move_range(spot, atm_iv)
    hv_move = _expected_move_range(spot, hv)
    if iv_move:
        payload["iv"] = iv_move
    if hv_move:
        payload["hv"] = hv_move
    return payload


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _ohlcv_records_to_df(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize historical OHLCV records to a sorted DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    else:
        df = df.sort_index()
    return df


def _volume_profile(df: pd.DataFrame, bins: int = 20) -> Optional[Dict[str, float]]:
    if df.empty or "close" not in df.columns or "volume" not in df.columns:
        return None
    clean = df[["close", "volume"]].astype(float).dropna()
    if clean.empty:
        return None
    price_min, price_max = clean["close"].min(), clean["close"].max()
    if price_max == price_min:
        return {
            "top_volume": float(clean["volume"].sum()),
            "top_price": float(price_min),
            "bins": 1,
        }
    try:
        clean["bucket"] = pd.cut(clean["close"], bins=bins, include_lowest=True)
        grouped = clean.groupby("bucket", observed=False)["volume"].sum()
        if grouped.empty:
            return None
        top_bucket = grouped.idxmax()
        top_price = float(top_bucket.mid) if top_bucket is not None else None
        return {
            "top_volume": float(grouped.max()),
            "top_price": top_price,
            "bins": bins,
        }
    except Exception:
        return None


def _rs_rating(df_symbol: pd.DataFrame, df_bench: pd.DataFrame) -> Optional[Dict[str, float]]:
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "close" not in df.columns:
            return pd.DataFrame()
        cleaned = df.dropna(subset=["close"])
        if "date" in cleaned.columns:
            cleaned = cleaned.sort_values("date")
        else:
            cleaned = cleaned.sort_index()
        return cleaned

    s = _prepare(df_symbol)
    b = _prepare(df_bench)
    if s.empty or b.empty:
        return None
    try:
        ret_sym = float(s["close"].iloc[-1]) / float(s["close"].iloc[0]) - 1.0
        ret_bench = float(b["close"].iloc[-1]) / float(b["close"].iloc[0]) - 1.0
        rs = ret_sym - ret_bench
        denom = 1.0 + ret_bench
        ratio = (1.0 + ret_sym) / denom - 1.0 if denom != 0 else None
        return {
            "return": ret_sym,
            "bench_return": ret_bench,
            "rs": rs,
            "rs_ratio": ratio,
        }
    except Exception:
        return None


_MASSIVE_VALUE_KEYS: Sequence[str] = (
    "value",
    "rate",
    "yield",
    "expectation",
    "median",
    "projection",
    "estimate",
    "fed_funds",
)
_DATE_KEYS: Sequence[str] = ("date", "as_of", "timestamp", "time")


def _extract_numeric_from_record(
    record: Dict[str, Any],
    *,
    value_keys: Sequence[str],
) -> Optional[float]:
    for key in value_keys:
        if key not in record:
            continue
        val = _safe_float(record.get(key))
        if val is not None:
            return val
    return None


def _format_percent(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value * 100:.1f}%"


def _summarize_flow(
    holder_summary: Dict[str, Any],
    put_call_payload: Dict[str, Any],
    gex_payload: Dict[str, Any],
) -> str:
    parts: List[str] = []
    trend_metric = _safe_float(holder_summary.get("trend_metric"))
    if trend_metric is not None:
        direction = "回升" if trend_metric >= 0 else "下降"
        parts.append(f"机构持仓{direction} {_format_percent(abs(trend_metric))}")
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
            parts.append(f"跑赢基准 {_format_percent(rs_value)}")
        elif rs_value <= -0.05:
            parts.append(f"落后基准 {_format_percent(abs(rs_value))}")
        else:
            parts.append("相对强度中性")

    if price_vs_poc is not None:
        diff_text = _format_percent(price_vs_poc)
        if price_vs_poc >= 0.05:
            parts.append(f"价格高于筹码 {diff_text}")
        elif price_vs_poc <= -0.05:
            parts.append(f"价格低于筹码 {diff_text}")
        else:
            parts.append("价格接近成本区")

    if not parts and latest_close is not None:
        parts.append(f"最新收盘 ${latest_close:.2f}")
    return "；".join(parts)


def _records_to_series(
    records: Sequence[Dict[str, Any]],
    *,
    value_keys: Sequence[str],
    date_keys: Sequence[str],
) -> pd.Series:
    data: List[tuple[pd.Timestamp, float]] = []
    for row in records or []:
        if not isinstance(row, dict):
            continue
        val = _extract_numeric_from_record(row, value_keys=value_keys)
        if val is None:
            continue
        date_value: Optional[Any] = None
        for key in date_keys:
            if key in row:
                date_value = row.get(key)
                break
        if date_value is None:
            continue
        try:
            timestamp = pd.to_datetime(date_value)
        except Exception:
            continue
        data.append((timestamp, val))
    if not data:
        return pd.Series(dtype=float)
    data.sort(key=lambda item: item[0])
    index, values = zip(*data)
    return pd.Series(values, index=pd.Index(index, name="date"), dtype=float)


def _series_from_observations(observations: Sequence[Dict[str, Any]]) -> pd.Series:
    return _records_to_series(observations, value_keys=("value",), date_keys=("date",))


def _series_from_massive(records: Sequence[Dict[str, Any]]) -> pd.Series:
    return _records_to_series(records, value_keys=_MASSIVE_VALUE_KEYS, date_keys=_DATE_KEYS)


def _latest_series_value(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    val = clean.iloc[-1]
    return float(val) if math.isfinite(val) else None


def _net_liquidity_series(
    assets: Optional[pd.Series],
    tga: Optional[pd.Series],
    rrp: Optional[pd.Series],
) -> Optional[pd.Series]:
    if assets is None or tga is None or rrp is None:
        return None
    combined = pd.concat([assets, tga, rrp], axis=1)
    combined.columns = ["assets", "tga", "rrp"]
    return combined["assets"] - (combined["tga"] + combined["rrp"])


def _quantile_thresholds(
    series: Optional[pd.Series],
    *,
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
) -> List[float]:
    if series is None or series.empty:
        return []
    clean = series.dropna()
    if clean.empty:
        return []
    try:
        quantile_values = clean.quantile(list(quantiles))
    except Exception:
        return []
    return [float(v) for v in quantile_values.tolist()]


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


_HOLDER_DATE_FIELDS: Sequence[str] = (
    "date",
    "reportDate",
    "report_date",
    "filingDate",
    "filing_date",
    "filedDate",
    "filed_date",
    "period",
    "as_of",
    "Date Reported",
)
_HOLDER_SHARE_FIELDS: Sequence[str] = (
    "shares",
    "sharesNumber",
    "shares_number",
    "sharesAmount",
    "numberOfShares",
    "Number of Shares",
    "Shares",
)
_HOLDER_VALUE_FIELDS: Sequence[str] = (
    "value",
    "marketValue",
    "market_value",
    "valueOfShares",
    "Value",
)
_HOLDER_NAME_FIELDS: Sequence[str] = (
    "holder",
    "holderName",
    "holder_name",
    "organization",
    "organizationName",
    "organization_name",
    "investmentManager",
    "investment_manager",
    "investorName",
    "investorname",
    "name",
)


def _coerce_holder_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        try:
            return datetime.utcfromtimestamp(float(value)).date()
        except Exception:
            return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            parsed = pd.to_datetime(cleaned, errors="coerce")
        except Exception:
            return None
        if isinstance(parsed, pd.Timestamp):
            return parsed.date()
        if isinstance(parsed, datetime):
            return parsed.date()
        if isinstance(parsed, date):
            return parsed
    return None


def _sanitize_holder_numeric(value: Any) -> Optional[float]:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace(",", "").replace("$", "").replace("%", "")
        value = cleaned
    return _safe_float(value)


def _summarize_institutional_holders(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> Dict[str, Any]:
    """
    Aggregate raw 13F holder records into quarterly totals and compute QoQ change metrics.
    """
    summary: Dict[str, Any] = {
        "source": source,
        "record_count": len(records),
        "timeline": [],
        "latest_holder_count": len(records),
        "latest_period": None,
        "previous_period": None,
        "qoq_change_value": None,
        "qoq_change_shares": None,
        "trend_metric": None,
    }
    if not records:
        return summary

    aggregated: Dict[tuple[int, int], Dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        as_of: Optional[date] = None
        for field in _HOLDER_DATE_FIELDS:
            if field not in row:
                continue
            as_of = _coerce_holder_date(row.get(field))
            if as_of is not None:
                break
        if as_of is None:
            continue
        quarter = ((as_of.month - 1) // 3) + 1
        bucket_key = (as_of.year, quarter)
        bucket = aggregated.setdefault(
            bucket_key,
            {
                "value_total": 0.0,
                "value_samples": 0,
                "share_total": 0.0,
                "share_samples": 0,
                "holder_ids": set(),
                "row_count": 0,
            },
        )
        bucket["row_count"] += 1

        holder_id: Optional[str] = None
        for name_key in _HOLDER_NAME_FIELDS:
            candidate = row.get(name_key)
            if candidate:
                holder_id = str(candidate)
                break
        if holder_id:
            bucket["holder_ids"].add(holder_id)

        for value_key in _HOLDER_VALUE_FIELDS:
            if value_key not in row:
                continue
            value = _sanitize_holder_numeric(row.get(value_key))
            if value is None:
                continue
            bucket["value_total"] += value
            bucket["value_samples"] += 1
            break

        for share_key in _HOLDER_SHARE_FIELDS:
            if share_key not in row:
                continue
            shares = _sanitize_holder_numeric(row.get(share_key))
            if shares is None:
                continue
            bucket["share_total"] += shares
            bucket["share_samples"] += 1
            break

    if not aggregated:
        return summary

    timeline: List[Dict[str, Any]] = []
    for (year, quarter), bucket in sorted(aggregated.items(), key=lambda item: item[0], reverse=True):
        holder_count = len(bucket["holder_ids"]) or bucket["row_count"]
        entry = {
            "period": f"{year}-Q{quarter}",
            "holder_count": holder_count,
            "total_value": float(bucket["value_total"]) if bucket["value_samples"] > 0 else None,
            "total_shares": float(bucket["share_total"]) if bucket["share_samples"] > 0 else None,
        }
        timeline.append(entry)

    if not timeline:
        return summary

    summary["timeline"] = timeline
    summary["latest_holder_count"] = timeline[0]["holder_count"]
    summary["latest_period"] = timeline[0]["period"]
    summary["previous_period"] = timeline[1]["period"] if len(timeline) > 1 else None

    if len(timeline) > 1:
        summary["qoq_change_value"] = _relative_change(
            timeline[0]["total_value"], timeline[1]["total_value"]
        )
        summary["qoq_change_shares"] = _relative_change(
            timeline[0]["total_shares"], timeline[1]["total_shares"]
        )
    trend_metric = summary["qoq_change_value"]
    if trend_metric is None:
        trend_metric = summary["qoq_change_shares"]
    summary["trend_metric"] = trend_metric
    return summary


async def _fetch_with_default(
    coro: Awaitable[Any],
    label: str,
    default: Any,
) -> tuple[Any, Optional[str]]:
    """
    Execute provider coroutine and capture ProviderError into a log-friendly message.
    """
    try:
        result = await coro
        return result, None
    except ProviderError as exc:  # pragma: no cover - network
        return default, f"{label}: {exc}"


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


@dataclass
class FactorResult:
    score: Optional[float]
    status: str
    summary: Optional[str] = None
    key_evidence: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    components: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    weight_denominator: float = 0.0


_FUNDAMENTAL_DOC_WEIGHTS: Dict[str, float] = {
    # Reflect AION v8.0 formula weights so compute_fundamental stays aligned with docs.
    "fundamental.growth": 25.0,
    "fundamental.profitability": 25.0,
    "fundamental.cash_flow": 20.0,
    "fundamental.leverage": 15.0,
    "fundamental.tam_expansion": 15.0,
}


# ---------- Industry helpers ----------


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


# ---------- Fundamental helpers ----------


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


@dataclass
class MacroSnapshot:
    latest: Dict[str, Optional[float]]
    threshold_series: Dict[str, Optional[pd.Series]]
    rate_expectation_source: str


@dataclass
class MacroScoreResult:
    factor_scores: Dict[str, Optional[float]]
    net_liquidity_score: Optional[int]
    spread_score: Optional[int]
    hy_score: Optional[int]
    fed_score: Optional[int]
    rate_trend_score: Optional[float]
    global_demand_score: Optional[int]


_MACRO_SERIES_IDS = {
    "yield_curve": "T10Y2Y",
    "high_yield_spread": "BAMLH0A0HYM2",
    "fed_funds": "FEDFUNDS",
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
    rate_expectation_source = (
        "massive_inflation_expectations" if expectation_series_aligned is not None else "fred_fed_funds"
    )

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


# ---------- Factor calculators ----------


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
    volumes = df["volume"].astype(float)
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
    errors: List[str] = []

    async def _fetch_or_log(name: str, coro: Awaitable[Any], default: Any) -> Any:
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


# ---------- Orchestration helper ----------


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
    fred: Optional[FREDProvider] = None,
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
    fred_provider = fred or FREDProvider()

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


async def compute_soft_factor(
    ticker: str,
    factor_code: str,
    db: Optional[AsyncSession],
    massive: Optional[MassiveProvider],
    yfinance: Optional[YFinanceProvider] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> FactorResult:
    """
    Load latest soft factor score (e.g., F2 Industry, F7 Catalyst) from DB cache.
    If not available, attempt to generate using news context; otherwise return placeholder.
    """
    applied_weights, weight_denominator = _sanitize_weights(factor_weights)

    if db is None:
        return _soft_result(
            None,
            status="unavailable",
            summary="DB session required to load soft factor scores",
            key_evidence=[],
            sources=[],
            asof_date=None,
            confidence=None,
            errors=["DB session required to load soft factor scores"],
            weights=applied_weights,
            weight_denominator=weight_denominator,
            factor_scores=_factor_scores_from(None, applied_weights),
        )

    entry = await fetch_soft_factor_score(
        ticker=ticker,
        factor_code=factor_code,
        db=db,
        latest=True,
    )

    if not entry:
        texts: List[str] = []
        sources: List[Dict[str, Any]] = []
        news_articles: List[Dict[str, Any]] = []
        calendar_context: Dict[str, Any] = {}
        if massive:
            try:
                since = (date.today() - timedelta(days=30)).isoformat()
                news = await massive.fetch_news(ticker, limit=20, published_utc=since)
                news_articles = news or []
                for item in news_articles:
                    title = item.get("title") or ""
                    summary = item.get("description") or item.get("summary") or ""
                    combined = f"{title}. {summary}".strip(". ")
                    if combined:
                        texts.append(combined)
                    sources.append(
                        {
                            "title": title or summary or "",
                            "url": item.get("article_url") or item.get("url"),
                            "source": item.get("source"),
                            "published_utc": item.get("published_utc") or item.get("published_at"),
                        }
                    )
            except Exception as exc:  # pragma: no cover - network/third-party
                logger.warning("soft_factor_news_failed", extra={"factor": factor_code, "ticker": ticker, "error": str(exc)})

        if factor_code == "F7" and yfinance is not None:
            try:
                calendar_context = await yfinance.fetch_earnings_calendar(ticker)
            except ProviderError as exc:  # pragma: no cover - provider errors
                logger.warning("soft_factor_calendar_failed", extra={"factor": factor_code, "ticker": ticker, "error": str(exc)})
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.warning("soft_factor_calendar_failed", extra={"factor": factor_code, "ticker": ticker, "error": str(exc)})

        scoring_context: Optional[Dict[str, Any]] = None
        if factor_code == "F7":
            scoring_context = {
                "news_articles": news_articles,
                "sources": sources,
                "calendar": calendar_context,
            }

        should_attempt = bool(texts)
        if factor_code == "F7" and (calendar_context or news_articles):
            should_attempt = True

        if should_attempt:
            try:
                scored = await score_soft_factor(
                    ticker,
                    factor_code,
                    texts,
                    db=db,
                    context=scoring_context,
                )
                factor_scores_payload = scored.get("factor_scores")
                return _soft_result(
                    scored.get("score"),
                    status="ok" if scored.get("score") is not None else "static",
                    summary=scored.get("summary"),
                    key_evidence=scored.get("key_evidence", []),
                    sources=scored.get("sources", []) or sources,
                    asof_date=scored.get("asof_date"),
                    confidence=scored.get("confidence"),
                    errors=[],
                    weights=applied_weights,
                    weight_denominator=weight_denominator,
                    factor_scores=factor_scores_payload or _factor_scores_from(scored.get("score"), applied_weights),
                    extra_components=scored.get("components"),
                )
            except Exception as exc:  # pragma: no cover - LLM/DB errors
                logger.warning("soft_factor_score_failed", extra={"factor": factor_code, "ticker": ticker, "error": str(exc)})

        return _soft_result(
            None,
            status="static",
            summary="No soft factor score found",
            key_evidence=[],
            sources=sources,
            asof_date=None,
            confidence=None,
            errors=["No soft factor score found; run soft factor scoring pipeline"],
            weights=applied_weights,
            weight_denominator=weight_denominator,
            factor_scores=_factor_scores_from(None, applied_weights),
        )

    stored_factor_scores = None
    stored_components = None
    if isinstance(entry.reasons, dict):
        stored_factor_scores = entry.reasons.get("factor_scores")
        stored_components = entry.reasons.get("components")

    return _soft_result(
        entry.score,
        status="ok" if entry.score is not None else "static",
        summary=entry.reasons,
        key_evidence=entry.citations,
        sources=[],
        asof_date=entry.asof_date.isoformat(),
        confidence=entry.confidence,
        errors=["Soft factor score is null; check ingest/LLM pipeline"] if entry.score is None else [],
        weights=applied_weights,
        weight_denominator=weight_denominator,
        factor_scores=stored_factor_scores or _factor_scores_from(entry.score, applied_weights),
        extra_components=stored_components,
    )
