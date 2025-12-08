"""Massive data adapter for equities and options snapshots."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.providers.base import HTTPProvider, ProviderError, _strip_none


settings = get_settings()


def _to_date_str(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else value


class MassiveProvider(HTTPProvider):
    name = "Massive"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(base_url or settings.massive_base_url)
        self.api_key = api_key or settings.massive_api_key

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def fetch_equity_daily(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        multiplier: int = 1,
        timespan: str = "day",
        limit: int = 5000,
        adjusted: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Adapter over aggregates to align with BaseProvider equity_daily shape.
        """
        records = await self.fetch_equity_aggregates(
            ticker,
            start=start,
            end=end,
            multiplier=multiplier,
            timespan=timespan,
            limit=limit,
            adjusted=adjusted,
        )
        normalized: List[Dict[str, Any]] = []
        for row in records:
            if "t" not in row:
                continue
            normalized.append(
                {
                    "ticker": row.get("T") or ticker,
                    "date": datetime.utcfromtimestamp(row["t"] / 1000).date().isoformat(),
                    "open": row.get("o"),
                    "high": row.get("h"),
                    "low": row.get("l"),
                    "close": row.get("c"),
                    "volume": row.get("v"),
                }
            )
        return normalized

    async def fetch_equity_aggregates(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        multiplier: int = 1,
        timespan: str = "day",
        limit: int = 5000,
        adjusted: bool = True,
    ) -> List[Dict[str, Any]]:
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{_to_date_str(start)}/{_to_date_str(end)}"
        params = {"limit": limit, "adjusted": str(adjusted).lower()}
        data = await self._get(path, params=params, headers=self._auth_headers())
        return data.get("results", []) if isinstance(data, dict) else []

    async def fetch_option_chain_snapshot(
        self,
        underlying: str,
        *,
        limit: int = 100,
        expiration_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        strike_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        According to docs/API/massive/rest/options/snapshots/option-chain-snapshot.txt,
        this endpoint defaults to limit=10. For validation we want a larger slice
        (up to API max 250) and we need to respect optional filters.
        """
        params: Dict[str, Any] = {
            "limit": min(limit, 250),
            "expiration_date": expiration_date,
            "contract_type": contract_type,
            "strike_price": strike_price,
        }
        results: List[Dict[str, Any]] = []
        next_url: Optional[str] = f"/v3/snapshot/options/{underlying}"

        while next_url and len(results) < limit:
            data = await self._get(
                next_url,
                params=_strip_none(params) if next_url.startswith("/") else None,
                headers=self._auth_headers(),
            )
            page_results = data.get("results", []) if isinstance(data, dict) else []
            results.extend(page_results)

            remaining = limit - len(results)
            if not isinstance(data, dict):
                break

            next_url = data.get("next_url")
            if next_url and remaining < 250:
                # When we continue pagination, ask only for the remaining records.
                params = {"limit": remaining}
            else:
                params = {}

        return results[:limit]

    async def fetch_unified_snapshot(self, symbol: str) -> Dict[str, Any]:
        return await self._get(
            f"/v3/snapshot/{symbol}",
            headers=self._auth_headers(),
        )

    async def fetch_financial_ratios(
        self, ticker: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        params = {"ticker": ticker, "limit": min(limit, 1000)}
        data = await self._get(
            "/stocks/financials/v1/ratios",
            params=_strip_none(params),
            headers=self._auth_headers(),
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        return results[:limit]

    async def fetch_news(
        self, ticker: str, *, limit: int = 20, published_utc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "ticker": ticker,
            "limit": min(limit, 1000),
            "published_utc.gte": published_utc,
        }
        data = await self._get(
            "/v2/reference/news",
            params=_strip_none(params),
            headers=self._auth_headers(),
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        return results[:limit]

    async def fetch_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Massive plan currently exposes ratios; return them within a standard envelope so callers can fallback to others.
        """
        try:
            ratios = await self.fetch_financial_ratios(ticker, limit=100)
        except ProviderError:
            ratios = []
        return {
            "income_statement": [],
            "balance_sheet": [],
            "cash_flow": [],
            "quarterly_income_statement": [],
            "quarterly_balance_sheet": [],
            "quarterly_cash_flow": [],
            "ratios": ratios,
        }

    async def fetch_holders(self, ticker: str) -> List[Dict[str, Any]]:
        # Not available on current Massive plan; return empty for fallback to kick in.
        return []

    async def fetch_inflation_expectations(
        self,
        *,
        horizon: Optional[str] = None,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        """
        Cleveland Fed / FedWatch equivalent time series.
        """
        params: Dict[str, Any] = {
            "limit": min(limit, 1000),
            "horizon": horizon,
        }
        data = await self._get(
            "/fed/v1/inflation-expectations",
            params=_strip_none(params),
            headers=self._auth_headers(),
        )
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data  # pragma: no cover - API variant
        else:
            results = []
        return results[:limit]

    async def fetch_treasury_yields(
        self,
        *,
        maturity: Optional[str] = None,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "limit": min(limit, 1000),
            "maturity": maturity,
        }
        data = await self._get(
            "/fed/v1/treasury-yields",
            params=_strip_none(params),
            headers=self._auth_headers(),
        )
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data  # pragma: no cover - API variant
        else:
            results = []
        return results[:limit]
