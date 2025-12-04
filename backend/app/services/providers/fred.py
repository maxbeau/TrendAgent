"""FRED data adapter for macro time series."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.providers.base import HTTPProvider, _strip_none


settings = get_settings()


class FREDProvider(HTTPProvider):
    name = "FRED"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(base_url or settings.fred_base_url)
        self.api_key = api_key or settings.fred_api_key

    async def fetch_series(
        self,
        series_id: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat() if start else None,
            "observation_end": end.isoformat() if end else None,
        }
        data = await self._get("/fred/series/observations", params=_strip_none(params))
        return data.get("observations", []) if isinstance(data, dict) else []
