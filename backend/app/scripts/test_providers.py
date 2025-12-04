"""Quick connectivity check for FMP, Massive, and FRED."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.services.providers import FMPProvider, FREDProvider, MassiveProvider, YFinanceProvider
from app.services.providers.base import ProviderError


async def main() -> None:
    ticker = "AAPL"
    start = date.today() - timedelta(days=10)
    end = date.today() - timedelta(days=1)

    massive = MassiveProvider()
    fmp = FMPProvider()
    fred = FREDProvider()
    yfinance = YFinanceProvider()

    print("=== Massive: aggregates ===")
    try:
        agg = await massive.fetch_equity_aggregates(ticker, start, end, limit=3)
        print(f"ok - got {len(agg)} bars")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== yfinance: aggregates ===")
    try:
        agg = await yfinance.fetch_equity_daily(ticker, start=start, end=end)
        print(f"ok - got {len(agg)} bars")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== yfinance: financials ===")
    try:
        fin = await yfinance.fetch_financials(ticker)
        income = fin.get("income_statement") or []
        print(f"ok - income_statement rows: {len(income)}")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== Massive: option chain snapshot ===")
    try:
        chain = await massive.fetch_option_chain_snapshot(ticker)
        print(f"ok - got {len(chain)} option legs")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== yfinance: option chain snapshot ===")
    try:
        chain = await yfinance.fetch_option_chain_snapshot(ticker)
        print(f"ok - got {len(chain)} option legs")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== yfinance: holders ===")
    try:
        holders = await yfinance.fetch_holders(ticker)
        print(f"ok - got {len(holders)} holders")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== FMP: historical price ===")
    try:
        prices = await fmp.fetch_equity_daily(ticker, start=start, end=end, limit=5)
        print(f"ok - got {len(prices)} rows")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\\n=== FRED: yield spread ===")
    try:
        obs = await fred.fetch_series("T10Y2Y", start=start - timedelta(days=30), end=end)
        print(f"ok - got {len(obs)} points")
    except ProviderError as exc:
        print(f"failed - {exc}")


if __name__ == "__main__":
    asyncio.run(main())
