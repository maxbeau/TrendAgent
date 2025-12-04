"""FMP data adapter for equities, fundamentals, and news."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.providers.base import HTTPProvider, ProviderError, _strip_none


settings = get_settings()


class FMPProvider(HTTPProvider):
    name = "FMP"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(base_url or settings.fmp_base_url)
        self.api_key = api_key or settings.fmp_api_key

    async def _get_authed(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = _strip_none(params)
        merged["apikey"] = self.api_key
        return await self._get(path, params=merged)

    async def fetch_equity_daily(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 250,
    ) -> List[Dict[str, Any]]:
        params_v4: Dict[str, Any] = {
            "symbol": ticker,
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
            "page": 0,
        }
        # Prefer v4; avoid hitting deprecated v3 unless v4 empty or errors.
        try:
            data_v4 = await self._get_authed("/v4/historical-price", params=params_v4)
        except ProviderError:
            data_v4 = {}

        results_v4 = data_v4.get("historicalStockList", []) if isinstance(data_v4, dict) else []
        if results_v4:
            return results_v4[:limit]

        params_v3: Dict[str, Any] = {
            "from": params_v4.get("from"),
            "to": params_v4.get("to"),
            "serietype": "line",
            "timeseries": limit if not start else None,
        }
        try:
            legacy = await self._get_authed(f"/v3/historical-price-full/{ticker}", params=params_v3)
            return legacy.get("historical", []) if isinstance(legacy, dict) else []
        except ProviderError:
            return []

    async def fetch_income_statement(
        self,
        ticker: str,
        *,
        period: str = "annual",
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        params = {"symbol": ticker, "period": period, "limit": limit}
        try:
            data = await self._get_authed("/v4/income-statement", params=params)
            return data if isinstance(data, list) else data.get("financials", [])
        except ProviderError:
            legacy = await self._get_authed("/v3/income-statement/", params=params)
            return legacy if isinstance(legacy, list) else legacy.get("financials", [])

    async def fetch_financial_growth(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        params = {"symbol": ticker, "limit": limit}
        try:
            # Preferred stable endpoint (docs/API/FMP_API.md)
            data = await self._get_authed("/stable/financial-growth", params=params)
        except ProviderError:
            data = await self._get_authed(f"/v3/financial-growth/{ticker}")
        return data if isinstance(data, list) else data.get("items", [])

    async def fetch_ratios_ttm(self, ticker: str) -> List[Dict[str, Any]]:
        params = {"symbol": ticker}
        try:
            # Preferred stable endpoint (docs/API/FMP_API.md)
            data = await self._get_authed("/stable/ratios-ttm", params=params)
        except ProviderError:
            data = await self._get_authed(f"/v3/ratios-ttm/{ticker}")
        return data if isinstance(data, list) else data.get("ratios", [])

    async def fetch_institutional_holders(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            data = await self._get_authed("/v4/institutional-holders", params={"symbol": ticker})
            return data if isinstance(data, list) else data.get("items", [])
        except ProviderError:
            legacy = await self._get_authed(f"/v3/institutional-holder/{ticker}")
            return legacy if isinstance(legacy, list) else legacy.get("items", [])

    async def fetch_news(
        self,
        ticker: str,
        *,
        limit: int = 20,
        page: int = 0,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        params = {
            "limit": limit,
            "page": page,
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
            "tickers": ticker,
        }
        try:
            # Preferred stable endpoint per latest docs.
            data = await self._get_authed("/stable/news/stock-latest", params=params)
        except ProviderError:
            data = await self._get_authed("/v3/stock_news", params=_strip_none(params))
        return data if isinstance(data, list) else data.get("news", [])
