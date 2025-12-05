"""Simple risk/reward ratio estimator from recent price extremes."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.providers import MassiveProvider
from app.services.providers.base import ProviderError


settings = get_settings()


def _to_date_str(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else value


async def summarize_risk_reward(
    ticker: str,
    *,
    lookback_days: int = 120,
    massive: Optional[MassiveProvider] = None,
) -> Dict[str, Any]:
    """
    Compute risk/reward using recent high/low vs latest close from Massive aggregates.
    """
    provider = massive or MassiveProvider()
    end = date.today()
    start = end - timedelta(days=lookback_days)

    try:
        bars = await provider.fetch_equity_aggregates(
            ticker, start=_to_date_str(start), end=_to_date_str(end), limit=5000
        )
    except ProviderError as exc:  # pragma: no cover - network
        raise ProviderError(f"Failed to fetch price history from Massive: {exc}") from exc

    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    latest_close: Optional[float] = None

    for row in bars:
        c = row.get("c")
        h = row.get("h")
        l = row.get("l")
        if c is None or h is None or l is None:
            continue
        closes.append(c)
        highs.append(h)
        lows.append(l)
        latest_close = c

    if not closes or latest_close is None:
        return {
            "ticker": ticker,
            "status": "unavailable",
            "message": "No OHLCV data available.",
            "generated_at": datetime.utcnow(),
        }

    recent_high = max(highs)
    recent_low = min(lows)
    reward = max(recent_high - latest_close, 0)
    risk = max(latest_close - recent_low, 0)
    ratio = (reward / risk) if risk else None

    return {
        "ticker": ticker,
        "status": "ok",
        "close": latest_close,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "reward": reward,
        "risk": risk,
        "ratio": ratio,
        "lookback_days": lookback_days,
        "generated_at": datetime.utcnow(),
    }
