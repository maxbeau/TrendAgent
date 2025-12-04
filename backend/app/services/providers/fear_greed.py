"""Lightweight adapter over CNN Fear & Greed data."""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from app.services.providers.base import DEFAULT_TIMEOUT, ProviderError


HEADERS: Dict[str, str] = {
    # Spoof minimal browser headers; X-Forwarded-For mitigates 418 bot block.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com",
    "X-Forwarded-For": "8.8.8.8",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


class FearGreedIndexProvider:
    """
    Fetches CNN Fear & Greed JSON endpoint with anti-bot headers.
    Returns text fields only.
    """

    name = "fear-greed-index"
    endpoint = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    async def _fetch_graphdata(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=HEADERS) as client:
            try:
                response = await client.get(self.endpoint)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                raise ProviderError(f"HTTP {exc.response.status_code}: {exc.response.text}") from exc
            except httpx.HTTPError as exc:  # pragma: no cover - network errors
                raise ProviderError(str(exc)) from exc

    async def fetch_index(self) -> Dict[str, Any]:
        data = await self._fetch_graphdata()

        fg = data.get("fear_and_greed") or {}
        score = _safe_float(fg.get("score"))
        rating = str(fg.get("rating") or "").lower()
        timestamp = fg.get("timestamp")

        indicator_map = {
            "market_momentum_sp500": "Market Momentum",
            "stock_price_strength": "Stock Price Strength",
            "stock_price_breadth": "Stock Price Breadth",
            "put_call_options": "Put & Call Options",
            "market_volatility_vix": "Market Volatility",
            "junk_bond_demand": "Junk Bond Demand",
            "safe_haven_demand": "Safe Haven Demand",
        }

        indicators: List[Dict[str, Any]] = []
        for key, label in indicator_map.items():
            entry = data.get(key) or {}
            indicators.append(
                {
                    "type": label,
                    "sentiment": str(entry.get("rating") or "").lower(),
                    "summary": f"Score {entry.get('score')}",
                    "last_sentiment": "",
                    "last_changed": "",
                    "updated_on": entry.get("timestamp") or "",
                }
            )

        summary_parts: List[str] = []
        if score is not None:
            summary_parts.append(f"Now: {score:.1f}")
        if rating:
            summary_parts.append(f"({rating})")
        if timestamp:
            summary_parts.append(f"as of {timestamp}")

        summary_text = " ".join(summary_parts).strip()

        return {
            "score": score,
            "rating": rating,
            "timestamp": timestamp,
            "summary": summary_text,
            "indicators": indicators,
        }
