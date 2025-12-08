"""Common helpers for external data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_TIMEOUT = 20.0
MAX_RETRIES = 2
RETRY_STATUS = {429, 503}
logger = logging.getLogger(__name__)


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
        def _log_rate_event(
            *,
            status: int,
            detail: str = "",
            retry_after: Optional[str] = None,
            attempt: int = 1,
            will_retry: bool = False,
            url: Optional[str] = None,
        ) -> None:
            # Avoid泄漏敏感信息，仅记录元数据。
            logger.warning(
                "provider_rate_event",
                extra={
                    "provider": getattr(self, "name", self.__class__.__name__),
                    "status": status,
                    "path": path,
                    "url": url,
                    "retry_after": retry_after,
                    "param_keys": sorted(list((params or {}).keys())),
                    "auth_header": bool(headers and any(k.lower() == "authorization" for k in headers)),
                    "detail": detail[:200],
                    "attempt": attempt,
                    "will_retry": will_retry,
                },
            )

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.get(path, params=_strip_none(params), headers=headers)
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                    status = exc.response.status_code
                    detail = exc.response.text
                    retry_after = exc.response.headers.get("Retry-After")
                    if status in RETRY_STATUS and attempt < MAX_RETRIES:
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.5 * (2**attempt)
                        _log_rate_event(
                            status=status,
                            detail=detail,
                            retry_after=retry_after,
                            attempt=attempt + 1,
                            will_retry=True,
                            url=str(exc.request.url) if exc.request else None,
                        )
                        await asyncio.sleep(delay)
                        continue
                    if status >= 429:
                        _log_rate_event(
                            status=status,
                            detail=detail,
                            retry_after=retry_after,
                            attempt=attempt + 1,
                            will_retry=False,
                            url=str(exc.request.url) if exc.request else None,
                        )
                    else:
                        _log_rate_event(status=status, detail=str(exc))
                    raise ProviderError(f"HTTP {exc.response.status_code} for {path}: {exc.response.text}") from exc
                except httpx.HTTPError as exc:  # pragma: no cover - network errors
                    _log_rate_event(status=0, detail=str(exc))
                    raise ProviderError(f"HTTP error for {path}: {exc}") from exc

        data: Dict[str, Any] = response.json() if response.content else {}
        # Some providers wrap errors inside JSON without HTTP status.
        if isinstance(data, dict):
            msg = data.get("error") or data.get("message")
            status_field = data.get("status")
            if msg and isinstance(msg, str) and ("too many" in msg.lower() or "rate" in msg.lower()):
                logger.warning(
                    "provider_rate_event",
                    extra={
                        "provider": getattr(self, "name", self.__class__.__name__),
                        "status": status_field or 200,
                        "path": path,
                        "url": str(response.url),
                        "retry_after": response.headers.get("Retry-After"),
                        "param_keys": sorted(list((params or {}).keys())),
                        "auth_header": bool(headers and any(k.lower() == "authorization" for k in headers)),
                        "detail": msg[:200],
                    },
                )
                raise ProviderError(f"Upstream rate limit for {path}: {msg}")
            if msg:
                raise ProviderError(f"Upstream error for {path}: {msg}")
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
