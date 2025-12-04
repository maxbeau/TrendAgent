"""
Smoke test for YFinanceProvider without relying on Massive limits.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.config import get_settings
from app.services.providers import YFinanceProvider
from app.services.providers.base import ProviderError


async def main() -> None:
    ticker = "AAPL"
    start = date.today() - timedelta(days=30)
    end = date.today()
    settings = get_settings()

    provider = YFinanceProvider(proxy=settings.yfinance_proxy)

    print("=== yfinance: daily bars ===")
    try:
        bars = await provider.fetch_equity_daily(ticker, start=start, end=end)
        print(f"ok - got {len(bars)} bars; first: {bars[:1]}")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: option chain snapshot ===")
    try:
        chain = await provider.fetch_option_chain_snapshot(ticker, limit=10)
        print(f"ok - got {len(chain)} option legs; sample: {chain[:2]}")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: financials ===")
    try:
        fin = await provider.fetch_financials(ticker)
        income = fin.get("income_statement") or []
        print(f"ok - income_statement rows: {len(income)}")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: holders ===")
    try:
        holders = await provider.fetch_holders(ticker)
        print(f"ok - got {len(holders)} holders")
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: put/call ratio (SPY) ===")
    try:
        pcr = await provider.fetch_put_call_ratio("SPY")
        ratio = pcr.get("put_call_ratio")
        print(
            "ok - ratio: {ratio}, call_vol: {call_vol}, put_vol: {put_vol}, exp: {exp}".format(
                ratio=ratio,
                call_vol=pcr.get("call_volume"),
                put_vol=pcr.get("put_volume"),
                exp=pcr.get("expiration"),
            )
        )
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: gamma exposure (SPY) ===")
    try:
        gex = await provider.fetch_gamma_exposure("SPY")
        total = gex.get("total_gamma_exposure")
        call = gex.get("call_gamma_exposure")
        put = gex.get("put_gamma_exposure")
        exp = gex.get("expiration")
        spot = gex.get("spot")

        if None in (total, call, put, spot):
            print(
                "partial - total: {total}, call: {call}, put: {put}, exp: {exp}, spot: {spot}".format(
                    total=total, call=call, put=put, exp=exp, spot=spot
                )
            )
        else:
            print(
                "ok - total_gex: {total:.2e}, call_gex: {call:.2e}, put_gex: {put:.2e}, exp: {exp}, spot: {spot}".format(
                    total=total, call=call, put=put, exp=exp, spot=spot
                )
            )
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: IV skew 25d (SPY) ===")
    try:
        skew = await provider.fetch_vol_skew("SPY")
        print(
            "skew: {skew}, put25: {put25}, call25: {call25}, atm: {atm}, exp: {exp}, spot: {spot}".format(
                skew=skew.get("skew_25d"),
                put25=skew.get("put_25d_iv"),
                call25=skew.get("call_25d_iv"),
                atm=skew.get("atm_iv"),
                exp=skew.get("expiration"),
                spot=skew.get("spot"),
            )
        )
    except ProviderError as exc:
        print(f"failed - {exc}")

    print("\n=== yfinance: IV vs HV (SPY) ===")
    try:
        ivhv = await provider.fetch_iv_hv("SPY")
        print(
            "atm_iv: {atm}, hv: {hv}, diff: {diff}, exp: {exp}, spot: {spot}".format(
                atm=ivhv.get("atm_iv"),
                hv=ivhv.get("hv"),
                diff=ivhv.get("iv_vs_hv"),
                exp=ivhv.get("expiration"),
                spot=ivhv.get("spot"),
            )
        )
    except ProviderError as exc:
        print(f"failed - {exc}")


if __name__ == "__main__":
    asyncio.run(main())
