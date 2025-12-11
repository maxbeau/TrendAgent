from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from app.services.providers.base import ProviderError

from app.services.soft_factors import normalize_key_evidence, normalize_summary


def _soft_result(
    score: Optional[float],
    *,
    status: str,
    summary: Optional[str],
    key_evidence: Optional[List[str]],
    sources: Optional[List[Dict[str, Any]]] = None,
    asof_date: Optional[str] = None,
    confidence: Optional[float] = None,
    errors: List[str],
    weights: Optional[Dict[str, float]] = None,
    weight_denominator: float = 0.0,
    factor_scores: Optional[Dict[str, Optional[float]]] = None,
    extra_components: Optional[Dict[str, Any]] = None,
) -> "FactorResult":
    sanitized_summary = normalize_summary(summary)
    sanitized_key_evidence = normalize_key_evidence(key_evidence)
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


def _records_to_series(
    records: Sequence[Dict[str, Any]],
    *,
    value_keys: Sequence[str],
    date_keys: Sequence[str],
) -> pd.Series:
    data = []
    for row in records:
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
)
_HOLDER_SHARE_FIELDS: Sequence[str] = ("shares", "adj_holding", "shares_held")
_HOLDER_VALUE_FIELDS: Sequence[str] = ("value", "market_value", "marketValue")
_HOLDER_NAME_FIELDS: Sequence[str] = (
    "holder",
    "organization",
    "investorName",
    "label",
    "name",
)


def _coerce_holder_date(record: Mapping[str, Any]) -> Optional[pd.Timestamp]:
    for key in _HOLDER_DATE_FIELDS:
        if key in record:
            try:
                return pd.to_datetime(record.get(key))
            except Exception:
                return None
    return None


def _sanitize_holder_numeric(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        val = record.get(key)
        try:
            num = float(val)
        except Exception:
            continue
        if math.isfinite(num):
            return num
    return None


def _summarize_institutional_holders(records: Sequence[Mapping[str, Any]], *, source: str) -> Dict[str, Any]:
    if not records:
        return {
            "latest_holder_count": 0,
            "trend_metric": None,
            "source": source,
        }

    sorted_records = sorted(records, key=lambda r: _coerce_holder_date(r) or pd.Timestamp.min, reverse=True)
    latest = sorted_records[0]
    latest_count = _sanitize_holder_numeric(latest, ("count", "total", "holderCount"))

    if len(sorted_records) >= 2:
        previous = sorted_records[1]
        prev_count = _sanitize_holder_numeric(previous, ("count", "total", "holderCount"))
        trend = _relative_change(latest_count, prev_count) if latest_count and prev_count else None
    else:
        trend = None

    return {
        "latest_holder_count": int(latest_count) if latest_count is not None else None,
        "trend_metric": trend,
        "source": source,
    }


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
