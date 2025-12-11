"""
AION Data Source Validator
--------------------------
Validates reachability and data integrity for all 8 AION factors:
1. Macro (FRED)
2. Industry (Massive + LLM)
3. Fundamental (FMP)
4. Technical (Massive)
5. Flow (Massive + FMP)
6. Sentiment (Massive + LLM)
7. Catalyst (Massive/FMP News + LLM)
8. Volatility (Massive Options)
"""

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
# Prefer backend/.env (where keys are stored) but also allow root .env without overriding.
load_dotenv(BACKEND_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=False)

from app.config import get_settings
from app.services.providers import (
    FMPProvider,
    FREDProvider,
    FearGreedIndexProvider,
    MassiveProvider,
    YFinanceProvider,
)
from app.services.providers.base import ProviderError

settings = get_settings()


class ValidationReport:
    def __init__(self):
        self.results: Dict[str, Dict[str, str]] = {}

    def log(self, factor: str, item: str, provider: str, status: str, message: str = ""):
        key = f"{factor}::{item}::{provider}"
        self.results[key] = {"status": status, "message": message}
        icon = "✅" if status == "PASS" else "⚠️" if status == "SKIPPED" else "❌"
        print(f"{icon} [{factor}][{item}][{provider}] {status}: {message}")

    def print_summary(self):
        print("\n=== AION Data Source Validation Summary ===")
        passed = sum(1 for r in self.results.values() if r["status"] == "PASS")
        total = len(self.results)
        print(f"Total checks: {total}, Passed: {passed}, Failed/Skipped: {total - passed}")
        if passed == total:
            print("🚀 All systems go! Ready for AION Algorithm implementation.")
        else:
            print("⚠️ Some data sources are unreachable or missing. Review failures above.")

async def validate_macro(fred: FREDProvider, report: ValidationReport):
    print("\n--- Validating F1: Macro (FRED) ---")
    macro_codes = [
        ("WALCL", "Real Liquidity: Fed BS"),
        ("RRPONTSYD", "Real Liquidity: RRP"),
        ("WTREGEN", "Real Liquidity: TGA"),
        ("BAMLH0A0HYM2", "Credit Spread: HY"),
        ("T10Y2Y", "Rate Trend: 10Y-2Y"),
        ("DFEDTARU", "Rate Trend: FFR"),
        ("IPMAN", "Global Demand Proxy: Industrial Production Manufacturing"),
        ("DGORDER", "Global Demand Proxy: Durable Goods Orders"),
    ]
    for code, label in macro_codes:
        try:
            series = await fred.fetch_series(code, start=date.today() - timedelta(days=180))
            status = "PASS" if series else "FAIL"
            report.log("Macro", label, "FRED", status, f"{len(series)} points")
        except Exception as exc:
            report.log("Macro", label, "FRED", "FAIL", str(exc))


async def validate_industry(fmp: FMPProvider, yfinance: YFinanceProvider, ticker: str, report: ValidationReport):
    """
    Industry: prefer yfinance statements; FMP optional fallback (many v3 endpoints now legacy/paid).
    """
    print(f"\n--- Validating F2: Industry (yfinance first, FMP fallback) for {ticker} ---")
    try:
        y_financials = await yfinance.fetch_financials(ticker)
    except Exception as exc:  # pragma: no cover - network
        y_financials = {}
        report.log("Industry", "prefetch", "yfinance", "FAIL", str(exc))

    try:
        fmp_income = await fmp.fetch_income_statement(ticker, limit=4)
    except ProviderError as exc:  # pragma: no cover - network
        fmp_income = []
        if "403" in str(exc) or "Legacy" in str(exc):
            report.log("Industry", "prefetch", "FMP", "SKIPPED", "Plan limit / legacy endpoint")
        else:
            report.log("Industry", "prefetch", "FMP", "FAIL", str(exc))
    except Exception as exc:  # pragma: no cover - network
        fmp_income = []
        report.log("Industry", "prefetch", "FMP", "FAIL", str(exc))

    industry_items = [
        ("Revenue YoY", "income_statement"),
        ("CapEx Cycle", "cash_flow"),
        ("Margin Trend", "income_statement"),
        ("Policy Tailwind", "LLM"),
    ]

    income = y_financials.get("income_statement") or []
    cashflow = y_financials.get("cash_flow") or []

    for item, kind in industry_items:
        # yfinance primary
        if kind == "LLM":
            report.log("Industry", item, "yfinance", "SKIPPED", "Qualitative via LLM")
        elif (kind == "income_statement" and income) or (kind == "cash_flow" and cashflow):
            msg = f"income rows: {len(income)}, cashflow rows: {len(cashflow)}"
            report.log("Industry", item, "yfinance", "PASS", msg)
        else:
            report.log("Industry", item, "yfinance", "FAIL", "No financial statements")

        # FMP fallback
        if kind == "LLM":
            report.log("Industry", item, "FMP", "SKIPPED", "Qualitative via LLM")
        elif fmp_income:
            report.log("Industry", item, "FMP", "PASS", f"income rows: {len(fmp_income)} (legacy-prone)")
        else:
            report.log("Industry", item, "FMP", "SKIPPED", "Plan limit / legacy endpoint likely")


async def validate_fundamental(fmp: FMPProvider, yfinance: YFinanceProvider, ticker: str, report: ValidationReport):
    """
    Fundamental: prefer yfinance statements; compute ratios locally; FMP treated as optional/legacy.
    """
    print(f"\n--- Validating F3: Fundamental (yfinance first, FMP optional) for {ticker} ---")
    items = [
        ("Revenue Growth", "income_statement"),
        ("Margins", "income_statement"),
        ("Cash Flow Quality", "cash_flow"),
        ("NetDebt/EBITDA", "income_statement"),
        ("ROIC", "income_statement"),
        ("TAM", "LLM"),
    ]

    try:
        y_financials = await yfinance.fetch_financials(ticker)
    except Exception as exc:
        y_financials = {}
        report.log("Fundamental", "prefetch", "yfinance", "FAIL", str(exc))

    try:
        fmp_income = await fmp.fetch_income_statement(ticker, limit=8)
    except ProviderError as exc:
        fmp_income = []
        if "403" in str(exc) or "Legacy" in str(exc):
            report.log("Fundamental", "prefetch", "FMP", "SKIPPED", "Plan limit / legacy endpoint")
        else:
            report.log("Fundamental", "prefetch", "FMP", "FAIL", str(exc))
    except Exception as exc:
        fmp_income = []
        report.log("Fundamental", "prefetch", "FMP", "FAIL", str(exc))

    income = y_financials.get("income_statement") or []
    cashflow = y_financials.get("cash_flow") or []

    for item, kind in items:
        # yfinance primary
        if kind == "LLM":
            report.log("Fundamental", item, "yfinance", "SKIPPED", "Qualitative via LLM")
        elif kind == "cash_flow":
            if cashflow:
                report.log("Fundamental", item, "yfinance", "PASS", f"cashflow rows: {len(cashflow)}")
            else:
                report.log("Fundamental", item, "yfinance", "FAIL", "No cash_flow data")
        else:
            if income:
                report.log("Fundamental", item, "yfinance", "PASS", f"income rows: {len(income)} (local calc)")
            else:
                report.log("Fundamental", item, "yfinance", "FAIL", "No income_statement data")

        # FMP optional
        if kind == "LLM":
            report.log("Fundamental", item, "FMP", "SKIPPED", "Qualitative via LLM")
        elif kind == "cash_flow":
            report.log("Fundamental", item, "FMP", "SKIPPED", "Use local calc if statements available")
        elif fmp_income:
            report.log("Fundamental", item, "FMP", "PASS", f"income rows: {len(fmp_income)} (legacy-prone)")
        else:
            report.log("Fundamental", item, "FMP", "SKIPPED", "Plan limit / legacy endpoint likely")


async def validate_technical(massive: MassiveProvider, yfinance: YFinanceProvider, ticker: str, report: ValidationReport):
    print(f"\n--- Validating F4: Technical (Massive/yfinance) for {ticker} ---")
    start = date.today() - timedelta(days=400)
    end = date.today()

    def _bars_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        rename_map = {}
        if "c" in df.columns and "close" not in df.columns:
            rename_map["c"] = "close"
        if "v" in df.columns and "volume" not in df.columns:
            rename_map["v"] = "volume"
        df = df.rename(columns=rename_map)
        if "close" not in df.columns or "volume" not in df.columns:
            return pd.DataFrame()
        return df.dropna(subset=["close", "volume"])

    async def _fetch_df(symbol: str) -> pd.DataFrame:
        try:
            bars = await massive.fetch_equity_aggregates(symbol, start=start, end=end)
        except Exception:
            bars = []
        if not bars:
            try:
                bars = await yfinance.fetch_equity_daily(symbol, start=start, end=end)
            except Exception:
                bars = []
        return _bars_to_df(bars)

    def _volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict[str, float] | None:
        if df.empty:
            return None
        prices = df["close"].astype(float)
        vols = df["volume"].astype(float)
        clean = pd.DataFrame({"close": prices, "volume": vols}).dropna()
        if clean.empty:
            return None
        price_min, price_max = clean["close"].min(), clean["close"].max()
        if price_max == price_min:
            return {
                "top_volume": float(clean["volume"].sum()),
                "top_price": float(price_min),
                "bins": 1,
            }
        try:
            clean["bucket"] = pd.cut(clean["close"], bins=bins, include_lowest=True)
            vp = clean.groupby("bucket", observed=False)["volume"].sum()
            if vp.empty:
                return None
            top_bucket = vp.idxmax()
            return {
                "top_volume": float(vp.max()),
                "top_price": float(top_bucket.mid) if top_bucket else None,
                "bins": bins,
            }
        except Exception:
            return None

    def _rs_rating(df_symbol: pd.DataFrame, df_bench: pd.DataFrame) -> Dict[str, float] | None:
        if df_symbol.empty or df_bench.empty:
            return None
        s = df_symbol.sort_index()
        b = df_bench.sort_index()
        if s.empty or b.empty:
            return None
        try:
            ret_sym = float(s["close"].iloc[-1]) / float(s["close"].iloc[0]) - 1.0
            ret_bench = float(b["close"].iloc[-1]) / float(b["close"].iloc[0]) - 1.0
            rs = ret_sym - ret_bench
            ratio = (1 + ret_sym) / (1 + ret_bench) - 1.0 if (1 + ret_bench) != 0 else None
            return {"return": ret_sym, "bench_return": ret_bench, "rs": rs, "rs_ratio": ratio}
        except Exception:
            return None

    async def _check_bars(provider_name: str, fetch_fn, label: str):
        try:
            bars = await fetch_fn()
            status = "PASS" if len(bars) >= 200 else "FAIL"
            report.log("Technical", label, provider_name, status, f"bars: {len(bars)}")
        except ProviderError as exc:
            msg = str(exc)
            if provider_name.lower() == "massive" and any(token in msg.lower() for token in ["403", "401", "not authorized", "legacy"]):
                report.log("Technical", label, provider_name, "SKIPPED", msg)
            else:
                report.log("Technical", label, provider_name, "FAIL", msg)
        except Exception as exc:  # pragma: no cover - network
            report.log("Technical", label, provider_name, "FAIL", str(exc))

    await _check_bars(
        "Massive",
        lambda: massive.fetch_equity_aggregates(ticker, start=start, end=end),
        "MA20/50/200",
    )
    await _check_bars(
        "yfinance",
        lambda: yfinance.fetch_equity_daily(ticker, start=start, end=end),
        "MA20/50/200",
    )
    report.log("Technical", "IV Rank/Skew", "Massive", "SKIPPED", "Requires options history, not computed here")
    report.log("Technical", "IV Rank/Skew", "yfinance", "SKIPPED", "Requires options history, not computed here")

    price_df = await _fetch_df(ticker)
    bench_df = await _fetch_df("SPY")

    rs = _rs_rating(price_df, bench_df)
    if rs is None:
        report.log("Technical", "RS Rating", "Massive", "FAIL", "Insufficient data for RS calculation")
    else:
        msg = f"ret {rs['return']:.3f} vs SPY {rs['bench_return']:.3f} (rs {rs['rs']:.3f})"
        report.log("Technical", "RS Rating", "Massive", "PASS", msg)

    vp = _volume_profile(price_df)
    if vp is None or vp.get("top_price") is None:
        report.log("Technical", "Volume Profile", "Massive", "FAIL", "Could not compute volume profile")
    else:
        msg = f"POC {vp['top_price']:.2f} vol {vp['top_volume']:.0f} (bins {vp['bins']})"
        report.log("Technical", "Volume Profile", "Massive", "PASS", msg)


async def validate_flow(massive: MassiveProvider, fmp: FMPProvider, yfinance: YFinanceProvider, ticker: str, report: ValidationReport):
    print(f"\n--- Validating F5: Flow (Massive/FMP/yfinance) for {ticker} ---")
    try:
        holders = await fmp.fetch_institutional_holders(ticker)
        status = "PASS" if holders else "FAIL"
        report.log("Flow", "13F Holders", "FMP", status, f"rows: {len(holders)}")
    except ProviderError as exc:
        if "403" in str(exc) or "Legacy" in str(exc):
            report.log("Flow", "13F Holders", "FMP", "SKIPPED", "Plan limit / legacy endpoint")
        else:
            report.log("Flow", "13F Holders", "FMP", "FAIL", str(exc))
    except Exception as exc:
        report.log("Flow", "13F Holders", "FMP", "FAIL", str(exc))

    try:
        holders = await yfinance.fetch_holders(ticker)
        status = "PASS" if holders else "FAIL"
        report.log("Flow", "13F Holders", "yfinance", status, f"rows: {len(holders)}")
    except Exception as exc:
        report.log("Flow", "13F Holders", "yfinance", "FAIL", str(exc))

    report.log("Flow", "13F Holders", "Massive", "SKIPPED", "Endpoint not available on current Massive plan")
    report.log("Flow", "ETF Weights", "Massive", "SKIPPED", "Endpoint not available in docs")
    report.log("Flow", "ETF Weights", "FMP", "SKIPPED", "Endpoint not implemented")
    report.log("Flow", "ETF Weights", "yfinance", "SKIPPED", "Not provided by yfinance API")

    try:
        options = await massive.fetch_option_chain_snapshot(ticker, limit=150)
        status = "PASS" if options else "FAIL"
        report.log("Flow", "Option Flow", "Massive", status, f"legs: {len(options)}")
    except ProviderError as exc:
        if "403" in str(exc).lower() or "not authorized" in str(exc).lower():
            report.log("Flow", "Option Flow", "Massive", "SKIPPED", "Plan limit / not authorized")
        else:
            report.log("Flow", "Option Flow", "Massive", "FAIL", str(exc))
    except Exception as exc:
        report.log("Flow", "Option Flow", "Massive", "FAIL", str(exc))

    try:
        options = await yfinance.fetch_option_chain_snapshot(ticker, limit=150)
        status = "PASS" if options else "FAIL"
        report.log("Flow", "Option Flow", "yfinance", status, f"legs: {len(options)}")
    except ProviderError as exc:
        if "403" in str(exc).lower() or "not authorized" in str(exc).lower():
            report.log("Flow", "Option Flow", "yfinance", "SKIPPED", "Plan limit / not authorized")
        else:
            report.log("Flow", "Option Flow", "yfinance", "FAIL", str(exc))
    except Exception as exc:
        report.log("Flow", "Option Flow", "yfinance", "FAIL", str(exc))

    report.log("Flow", "Option Flow", "FMP", "SKIPPED", "Options not available on FMP")
    report.log("Flow", "Gamma Exposure", "Massive", "SKIPPED", "Needs option Greeks aggregation")
    try:
        gex = await yfinance.fetch_gamma_exposure(ticker)
        total = gex.get("total_gamma_exposure")
        if total is None:
            report.log("Flow", "Gamma Exposure", "yfinance", "FAIL", "No gamma exposure computed")
        else:
            msg = f"GEX {total:.2e} (exp {gex.get('expiration')}, spot {gex.get('spot')})"
            report.log("Flow", "Gamma Exposure", "yfinance", "PASS", msg)
    except Exception as exc:
        report.log("Flow", "Gamma Exposure", "yfinance", "FAIL", str(exc))
    report.log("Flow", "Gamma Exposure", "FMP", "SKIPPED", "Options not available on FMP")


async def validate_sentiment(
    yfinance: YFinanceProvider, fear_greed: FearGreedIndexProvider, report: ValidationReport
):
    print("\n--- Validating F6: Sentiment (yfinance + fear-greed-index) ---")
    try:
        vix = await yfinance.fetch_equity_daily("^VIX", start=date.today() - timedelta(days=30), end=date.today())
        status = "PASS" if vix else "FAIL"
        report.log("Sentiment", "VIX", "yfinance", status, f"rows: {len(vix)}")
    except Exception as exc:
        report.log("Sentiment", "VIX", "yfinance", "FAIL", str(exc))

    try:
        pcr = await yfinance.fetch_put_call_ratio("SPY")
        ratio = pcr.get("put_call_ratio")
        if ratio is None:
            report.log("Sentiment", "Put/Call Ratio", "yfinance", "FAIL", "No option volume data")
        else:
            msg = f"vol PCR {ratio:.3f} (exp {pcr.get('expiration')})"
            report.log("Sentiment", "Put/Call Ratio", "yfinance", "PASS", msg)
    except Exception as exc:
        report.log("Sentiment", "Put/Call Ratio", "yfinance", "FAIL", str(exc))
    try:
        index = await fear_greed.fetch_index()
        summary = index.get("summary", "")
        indicators = index.get("indicators") or []
        status = "PASS" if summary or indicators else "FAIL"
        report.log("Sentiment", "CNN Fear & Greed", "fear-greed-index", status, f"indicators: {len(indicators)}")
    except ProviderError as exc:
        report.log("Sentiment", "CNN Fear & Greed", "fear-greed-index", "FAIL", str(exc))
    except Exception as exc:  # pragma: no cover - network
        report.log("Sentiment", "CNN Fear & Greed", "fear-greed-index", "FAIL", str(exc))


async def validate_catalyst(fmp: FMPProvider, massive: MassiveProvider, yfinance: YFinanceProvider, report: ValidationReport):
    print("\n--- Validating F7: Catalyst (Massive/FMP) ---")
    try:
        news = await fmp.fetch_news("AAPL", limit=10)
        status = "PASS" if news else "FAIL"
        report.log("Catalyst", "News", "FMP", status, f"articles: {len(news)}")
    except Exception as exc:
        report.log("Catalyst", "News", "FMP", "FAIL", str(exc))

    try:
        news = await massive.fetch_news("AAPL", limit=10)
        status = "PASS" if news else "FAIL"
        report.log("Catalyst", "News", "Massive", status, f"articles: {len(news)}")
    except Exception as exc:
        report.log("Catalyst", "News", "Massive", "FAIL", str(exc))

    report.log("Catalyst", "Earnings Calendar", "Massive", "SKIPPED", "Endpoint unavailable on plan")
    report.log("Catalyst", "Earnings Calendar", "FMP", "SKIPPED", "Endpoint not implemented in adapter")

    try:
        cal = await yfinance.fetch_earnings_calendar("AAPL")
        next_date = cal.get("next_earnings_date")
        if next_date:
            report.log("Catalyst", "Earnings Calendar", "yfinance", "PASS", f"next {next_date}")
        else:
            report.log("Catalyst", "Earnings Calendar", "yfinance", "FAIL", "No earnings date")
    except Exception as exc:
        report.log("Catalyst", "Earnings Calendar", "yfinance", "FAIL", str(exc))
    report.log("Catalyst", "Policy Events", "LLM", "SKIPPED", "LLM pipeline not executed")

    if not settings.openai_api_key or "placeholder" in settings.openai_api_key:
        report.log("LLM", "LLM Connectivity", "OpenAI", "SKIPPED", "No OpenAI API Key configured")
    else:
        try:
            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model_name,
                base_url=settings.openai_base_url if settings.openai_base_url else None,
            )
            response = await llm.ainvoke([HumanMessage(content="Hello, are you ready for financial analysis?")])
            report.log("LLM", "LLM Connectivity", "OpenAI", "PASS", response.content[:60])
        except Exception as exc:
            report.log("LLM", "LLM Connectivity", "OpenAI", "FAIL", str(exc))


async def validate_volatility(massive: MassiveProvider, yfinance: YFinanceProvider, ticker: str, report: ValidationReport):
    print(f"\n--- Validating F8: Volatility (Massive/yfinance) for {ticker} ---")
    try:
        options = await massive.fetch_option_chain_snapshot(ticker, limit=150)
        status = "PASS" if options else "FAIL"
        report.log("Volatility", "Option Chain", "Massive", status, f"legs: {len(options)}")
    except ProviderError as exc:
        if "403" in str(exc).lower() or "not authorized" in str(exc).lower():
            report.log("Volatility", "Option Chain", "Massive", "SKIPPED", "Plan limit / not authorized")
        else:
            report.log("Volatility", "Option Chain", "Massive", "FAIL", str(exc))
    except Exception as exc:
        report.log("Volatility", "Option Chain", "Massive", "FAIL", str(exc))

    try:
        options = await yfinance.fetch_option_chain_snapshot(ticker, limit=150)
        status = "PASS" if options else "FAIL"
        report.log("Volatility", "Option Chain", "yfinance", status, f"legs: {len(options)}")
    except ProviderError as exc:
        if "403" in str(exc).lower() or "not authorized" in str(exc).lower():
            report.log("Volatility", "Option Chain", "yfinance", "SKIPPED", "Plan limit / not authorized")
        else:
            report.log("Volatility", "Option Chain", "yfinance", "FAIL", str(exc))
    except Exception as exc:
        report.log("Volatility", "Option Chain", "yfinance", "FAIL", str(exc))

    report.log("Volatility", "Option Chain", "FMP", "SKIPPED", "Options not available on FMP")
    report.log("Volatility", "IV Rank/Skew", "Massive", "SKIPPED", "Needs historical IV computation")

    try:
        skew = await yfinance.fetch_vol_skew(ticker)
        skew_value = skew.get("skew_25d")
        if skew_value is None:
            report.log("Volatility", "IV Skew 25d", "yfinance", "FAIL", "Could not compute skew")
        else:
            msg = (
                f"skew {skew_value:.3f} (put25 {skew.get('put_25d_iv'):.3f}, "
                f"call25 {skew.get('call_25d_iv'):.3f}, atm {skew.get('atm_iv'):.3f})"
            )
            report.log("Volatility", "IV Skew 25d", "yfinance", "PASS", msg)
    except Exception as exc:
        report.log("Volatility", "IV Skew 25d", "yfinance", "FAIL", str(exc))

    try:
        ivhv = await yfinance.fetch_iv_hv(ticker)
        atm_iv = ivhv.get("atm_iv")
        hv = ivhv.get("hv")
        if atm_iv is None or hv is None:
            report.log("Volatility", "IV vs HV", "yfinance", "FAIL", "Missing atm_iv or hv")
        else:
            msg = f"atm_iv {atm_iv:.3f} vs hv {hv:.3f} (diff {ivhv.get('iv_vs_hv'):.3f})"
            report.log("Volatility", "IV vs HV", "yfinance", "PASS", msg)
    except Exception as exc:
        report.log("Volatility", "IV vs HV", "yfinance", "FAIL", str(exc))

    try:
        gex = await yfinance.fetch_gamma_exposure(ticker)
        total = gex.get("total_gamma_exposure")
        if total is None:
            report.log("Volatility", "Gamma Exposure", "yfinance", "FAIL", "No gamma exposure computed")
        else:
            msg = f"GEX {total:.2e} (exp {gex.get('expiration')}, spot {gex.get('spot')})"
            report.log("Volatility", "Gamma Exposure", "yfinance", "PASS", msg)
    except Exception as exc:
        report.log("Volatility", "Gamma Exposure", "yfinance", "FAIL", str(exc))

async def main():
    report = ValidationReport()
    
    # Initialize Providers
    fmp = FMPProvider()
    massive = MassiveProvider()
    fred = FREDProvider()
    fear_greed = FearGreedIndexProvider()
    yfinance = YFinanceProvider()

    TICKER = "AAPL"  # Use a highly liquid ticker for validation
    
    await validate_macro(fred, report)
    await validate_industry(fmp, yfinance, TICKER, report)                 # F2
    await validate_fundamental(fmp, yfinance, TICKER, report)              # F3
    await validate_technical(massive, yfinance, TICKER, report)            # F4
    await validate_flow(massive, fmp, yfinance, TICKER, report)            # F5
    await validate_sentiment(yfinance, fear_greed, report)                 # F6
    await validate_catalyst(fmp, massive, yfinance, report)                # F7
    await validate_volatility(massive, yfinance, TICKER, report)           # F8
    
    report.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
