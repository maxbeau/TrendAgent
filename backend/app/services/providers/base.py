"""Common helpers for external data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_TIMEOUT = 20.0


class ProviderError(Exception):
    """Wrap upstream errors so callers can handle them uniformly."""


def _strip_none(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if v is not None}


class BaseProvider(ABC):
    """
    Minimal abstract interface for market-data style providers.
    Non-HTTP providers (e.g., yfinance) can still conform to this contract.
    """

    name: str

    @abstractmethod
    async def fetch_equity_daily(
        self,
        ticker: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def fetch_option_chain_snapshot(
        self,
        underlying: str,
        *,
        limit: int = 200,
        expiration_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        strike_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def fetch_financials(self, ticker: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def fetch_holders(self, ticker: str) -> List[Dict[str, Any]]:
        ...


class HTTPProvider(BaseProvider):
    """Lightweight HTTP client wrapper shared by all providers."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            try:
                response = await client.get(path, params=_strip_none(params), headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                raise ProviderError(f"HTTP {exc.response.status_code} for {path}: {exc.response.text}") from exc
            except httpx.HTTPError as exc:  # pragma: no cover - network errors
                raise ProviderError(f"HTTP error for {path}: {exc}") from exc

        data: Dict[str, Any] = response.json() if response.content else {}
        # Some providers wrap errors inside JSON without HTTP status.
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError(f"Upstream error for {path}: {data['error']}")
        return data

    async def fetch_equity_daily(
        self,
        ticker: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        raise ProviderError(f"{self.__class__.__name__} does not implement fetch_equity_daily")

    async def fetch_option_chain_snapshot(
        self,
        underlying: str,
        *,
        limit: int = 200,
        expiration_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        strike_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        raise ProviderError(f"{self.__class__.__name__} does not implement fetch_option_chain_snapshot")

    async def fetch_financials(self, ticker: str) -> Dict[str, Any]:
        raise ProviderError(f"{self.__class__.__name__} does not implement fetch_financials")

    async def fetch_holders(self, ticker: str) -> List[Dict[str, Any]]:
        raise ProviderError(f"{self.__class__.__name__} does not implement fetch_holders")
