"""
Smoke test for CNN Fear & Greed Index adapter.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.services.providers import FearGreedIndexProvider
from app.services.providers.base import ProviderError


async def main() -> None:
    provider = FearGreedIndexProvider()
    try:
        index = await provider.fetch_index()
        summary = index.get("summary", "")
        indicators = index.get("indicators") or []
        print(f"ok - summary preview: {summary[:80]}...")
        print(f"ok - indicators: {len(indicators)} entries")
    except ProviderError as exc:
        print(f"failed - {exc}")


if __name__ == "__main__":
    asyncio.run(main())
