"""Async yfinance adapter kept minimal now that the upstream client is stable."""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from app.services.providers.base import BaseProvider, ProviderError


def _df_records(df: pd.DataFrame, date_label: str = "date") -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    renamed = df.copy()
    renamed.index.name = date_label
    return renamed.reset_index().to_dict(orient="records")


def _std_norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _infer_spot(ticker: yf.Ticker, provided_spot: Optional[float]) -> Optional[float]:
    if provided_spot is not None:
        return float(provided_spot)

    fast_info = getattr(ticker, "fast_info", {}) or {}
    for key in ("last_price", "lastPrice", "last_trade_price", "last_trade", "last"):
        if key in fast_info and fast_info[key] is not None:
            return float(fast_info[key])

    try:
        recent = ticker.history(period="1d", interval="1m")
        if recent is not None and not recent.empty:
            return float(recent["Close"].iloc[-1])
    except Exception:
        return None

    return None


def _resolve_expiration(ticker: yf.Ticker, preferred: Optional[str]) -> Optional[str]:
    expirations = getattr(ticker, "options", None) or []
    if preferred:
        return preferred
    return expirations[0] if expirations else None


def _option_chain(
    ticker: yf.Ticker, preferred: Optional[str]
) -> Tuple[Optional[Any], Optional[str]]:
    target = _resolve_expiration(ticker, preferred)
    if not target:
        return None, None
    chain = ticker.option_chain(target)
    return chain, target


def _closest_iv_to_spot(frame: Optional[pd.DataFrame], spot: Optional[float]) -> Optional[float]:
    if frame is None or frame.empty or spot is None:
        return None
    best_iv: Optional[float] = None
    best_diff = float("inf")
    for _, row in frame.iterrows():
        iv = row.get("impliedVolatility")
        strike = row.get("strike")
        if iv is None or strike is None:
            continue
        try:
            iv_f = float(iv)
            strike_f = float(strike)
        except (TypeError, ValueError):
            continue
        if iv_f <= 0 or not math.isfinite(iv_f) or not math.isfinite(strike_f):
            continue
        diff = abs(strike_f - float(spot))
        if diff < best_diff:
            best_diff = diff
            best_iv = iv_f
    return best_iv


def _atm_iv_from_chain(chain: Optional[Any], spot: Optional[float]) -> Optional[float]:
    if chain is None or spot is None:
        return None
    values = []
    for frame in (getattr(chain, "calls", None), getattr(chain, "puts", None)):
        iv = _closest_iv_to_spot(frame, spot)
        if iv is not None:
            values.append(iv)
    return sum(values) / len(values) if values else None


class YFinanceProvider(BaseProvider):
    """Minimal async wrapper around yfinance to align with BaseProvider."""

    name = "yfinance"

    def __init__(self, proxy: Optional[str] = None) -> None:
        self.proxy = proxy

    def _make_ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol)

    async def _to_thread(self, func: Callable, *args, **kwargs):
        def _wrapper():
            if self.proxy:
                yf.set_config(proxy=self.proxy)
            return func(*args, **kwargs)

        try:
            return await asyncio.to_thread(_wrapper)
        except Exception as exc:  # pragma: no cover - network errors
            raise ProviderError(str(exc)) from exc

    async def fetch_equity_daily(
        self,
        ticker: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        def _history() -> pd.DataFrame:
            t = self._make_ticker(ticker)
            return t.history(start=start, end=end, interval=interval, auto_adjust=False, actions=False)

        df: pd.DataFrame = await self._to_thread(_history)
        if df is None or df.empty:
            return []

        renamed = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        renamed.index = pd.to_datetime(renamed.index).date
        renamed = renamed.reset_index().rename(columns={"index": "date"})
        renamed["ticker"] = ticker

        return renamed[["ticker", "date", "open", "high", "low", "close", "volume"]].to_dict("records")

    async def fetch_option_chain_snapshot(
        self,
        underlying: str,
        *,
        limit: int = 200,
        expiration_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        strike_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        def _load_chain() -> List[Dict[str, Any]]:
            ticker = self._make_ticker(underlying)
            chain, target_exp = _option_chain(ticker, expiration_date)
            if not target_exp or chain is None:
                return []
            frames = []
            if contract_type in (None, "call", "calls"):
                frames.append(chain.calls.assign(side="call"))
            if contract_type in (None, "put", "puts"):
                frames.append(chain.puts.assign(side="put"))
            if not frames:
                return []

            merged = pd.concat(frames, ignore_index=True)
            if strike_price is not None:
                merged = merged[merged["strike"] == strike_price]

            merged["expiration"] = target_exp
            return merged.to_dict(orient="records")[:limit]

        records = await self._to_thread(_load_chain)
        return [
            {
                "symbol": row.get("contractSymbol"),
                "underlying": underlying,
                "expiration": row.get("expiration"),
                "strike": row.get("strike"),
                "side": row.get("side"),
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "last": row.get("lastPrice"),
                "volume": row.get("volume"),
                "open_interest": row.get("openInterest"),
                "implied_vol": row.get("impliedVolatility"),
            }
            for row in records
        ]

    async def fetch_news(
        self, ticker: str, *, limit: int = 20, start: Optional[date | datetime | str] = None
    ) -> List[Dict[str, Any]]:
        def _load_news() -> List[Dict[str, Any]]:
            t = self._make_ticker(ticker)
            return getattr(t, "news", []) or []

        raw = await self._to_thread(_load_news)

        start_dt: Optional[datetime] = None
        if isinstance(start, datetime):
            start_dt = start
        elif isinstance(start, date):
            start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        elif isinstance(start, str):
            try:
                start_dt = datetime.fromisoformat(start)
            except ValueError:
                start_dt = None
            if start_dt and start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)

        normalized: List[Dict[str, Any]] = []
        for item in raw:
            published_raw = item.get("providerPublishTime") or item.get("providerPublishDate")
            published_at: Optional[datetime] = None
            if isinstance(published_raw, (int, float)):
                published_at = datetime.fromtimestamp(published_raw, tz=timezone.utc)
            elif isinstance(published_raw, str):
                try:
                    published_at = datetime.fromisoformat(published_raw)
                except ValueError:
                    published_at = None
                if published_at and published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)

            if start_dt and published_at and published_at < start_dt:
                continue

            normalized.append(
                {
                    "title": item.get("title") or "",
                    "summary": item.get("summary") or item.get("content") or "",
                    "url": item.get("link") or item.get("url"),
                    "source": item.get("publisher"),
                    "published_at": published_at.isoformat() if published_at else None,
                }
            )
            if len(normalized) >= limit:
                break

        return normalized

    async def fetch_put_call_ratio(
        self, underlying: str, *, expiration_date: Optional[str] = None
    ) -> Dict[str, Any]:
        def _compute_ratio() -> Dict[str, Any]:
            ticker = self._make_ticker(underlying)
            chain, target_exp = _option_chain(ticker, expiration_date)
            if not target_exp or chain is None:
                return {
                    "underlying": underlying,
                    "expiration": None,
                    "put_call_ratio": None,
                    "call_volume": 0.0,
                    "put_volume": 0.0,
                }

            def _sum_volume(frame: pd.DataFrame) -> float:
                if frame is None or frame.empty or "volume" not in frame:
                    return 0.0
                return float(frame["volume"].fillna(0).sum())

            call_volume = _sum_volume(chain.calls)
            put_volume = _sum_volume(chain.puts)
            ratio = put_volume / call_volume if call_volume > 0 else None

            return {
                "underlying": underlying,
                "expiration": target_exp,
                "put_call_ratio": ratio,
                "call_volume": call_volume,
                "put_volume": put_volume,
            }

        return await self._to_thread(_compute_ratio)

    async def fetch_gamma_exposure(
        self,
        underlying: str,
        *,
        expiration_date: Optional[str] = None,
        risk_free_rate: float = 0.0,
        spot_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        def _black_scholes_gamma(spot: float, strike: float, t_years: float, iv: float) -> Optional[float]:
            if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
                return None
            sqrt_t = math.sqrt(t_years)
            d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
            return _std_norm_pdf(d1) / (spot * iv * sqrt_t)

        def _compute_gex() -> Dict[str, Any]:
            ticker = self._make_ticker(underlying)
            spot = _infer_spot(ticker, spot_price)
            if spot is None:
                return {
                    "underlying": underlying,
                    "expiration": None,
                    "spot": None,
                    "risk_free_rate": risk_free_rate,
                    "total_gamma_exposure": None,
                    "call_gamma_exposure": None,
                    "put_gamma_exposure": None,
                }

            chain, target_exp = _option_chain(ticker, expiration_date)
            if not target_exp or chain is None:
                return {
                    "underlying": underlying,
                    "expiration": None,
                    "spot": spot,
                    "risk_free_rate": risk_free_rate,
                    "total_gamma_exposure": None,
                    "call_gamma_exposure": None,
                    "put_gamma_exposure": None,
                }

            try:
                exp_dt = datetime.fromisoformat(target_exp)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return {
                    "underlying": underlying,
                    "expiration": target_exp,
                    "spot": spot,
                    "risk_free_rate": risk_free_rate,
                    "total_gamma_exposure": None,
                    "call_gamma_exposure": None,
                    "put_gamma_exposure": None,
                }

            now = datetime.now(timezone.utc)
            t_years = max((exp_dt - now).total_seconds(), 0.0) / (365 * 24 * 3600)
            if t_years <= 0:
                return {
                    "underlying": underlying,
                    "expiration": target_exp,
                    "spot": spot,
                    "risk_free_rate": risk_free_rate,
                    "total_gamma_exposure": None,
                    "call_gamma_exposure": None,
                    "put_gamma_exposure": None,
                }

            def _side_gex(frame: pd.DataFrame, sign: float) -> float:
                if frame is None or frame.empty:
                    return 0.0
                total = 0.0
                for _, row in frame.iterrows():
                    iv = row.get("impliedVolatility")
                    strike = row.get("strike")
                    oi = row.get("openInterest")
                    if iv is None or strike is None or oi is None:
                        continue
                    if not math.isfinite(iv) or iv <= 0:
                        continue
                    if not math.isfinite(strike) or strike <= 0:
                        continue
                    if not math.isfinite(oi) or oi <= 0:
                        continue

                    gamma = _black_scholes_gamma(float(spot), float(strike), t_years, float(iv))
                    if gamma is None:
                        continue
                    total += sign * gamma * float(oi) * float(spot) * 100.0
                return total

            call_gex = _side_gex(chain.calls, sign=1.0)
            put_gex = _side_gex(chain.puts, sign=-1.0)
            total_gex = call_gex + put_gex

            return {
                "underlying": underlying,
                "expiration": target_exp,
                "spot": spot,
                "risk_free_rate": risk_free_rate,
                "total_gamma_exposure": total_gex,
                "call_gamma_exposure": call_gex,
                "put_gamma_exposure": put_gex,
            }

        return await self._to_thread(_compute_gex)

    async def fetch_vol_skew(
        self,
        underlying: str,
        *,
        expiration_date: Optional[str] = None,
        risk_free_rate: float = 0.0,
        spot_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        def _compute_skew() -> Dict[str, Any]:
            ticker = self._make_ticker(underlying)
            spot = _infer_spot(ticker, spot_price)
            if spot is None:
                return {
                    "underlying": underlying,
                    "expiration": None,
                    "spot": None,
                    "risk_free_rate": risk_free_rate,
                    "atm_iv": None,
                    "call_25d_iv": None,
                    "put_25d_iv": None,
                    "skew_25d": None,
                }

            chain, target_exp = _option_chain(ticker, expiration_date)
            if not target_exp or chain is None:
                return {
                    "underlying": underlying,
                    "expiration": None,
                    "spot": spot,
                    "risk_free_rate": risk_free_rate,
                    "atm_iv": None,
                    "call_25d_iv": None,
                    "put_25d_iv": None,
                    "skew_25d": None,
                }

            try:
                exp_dt = datetime.fromisoformat(target_exp)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return {
                    "underlying": underlying,
                    "expiration": target_exp,
                    "spot": spot,
                    "risk_free_rate": risk_free_rate,
                    "atm_iv": None,
                    "call_25d_iv": None,
                    "put_25d_iv": None,
                    "skew_25d": None,
                }

            now = datetime.now(timezone.utc)
            t_years = max((exp_dt - now).total_seconds(), 0.0) / (365 * 24 * 3600)
            if t_years <= 0:
                return {
                    "underlying": underlying,
                    "expiration": target_exp,
                    "spot": spot,
                    "risk_free_rate": risk_free_rate,
                    "atm_iv": None,
                    "call_25d_iv": None,
                    "put_25d_iv": None,
                    "skew_25d": None,
                }

            def _calc_d1(strike: float, iv: float) -> Optional[float]:
                if strike <= 0 or iv <= 0:
                    return None
                sqrt_t = math.sqrt(t_years)
                return (math.log(spot / strike) + (risk_free_rate + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)

            def _nearest_iv_by_delta(frame: pd.DataFrame, target_delta: float, is_call: bool) -> Optional[float]:
                if frame is None or frame.empty:
                    return None
                best_iv: Optional[float] = None
                best_diff = float("inf")
                for _, row in frame.iterrows():
                    iv = row.get("impliedVolatility")
                    strike = row.get("strike")
                    if iv is None or strike is None:
                        continue
                    if not math.isfinite(iv) or iv <= 0:
                        continue
                    if not math.isfinite(strike) or strike <= 0:
                        continue
                    d1 = _calc_d1(float(strike), float(iv))
                    if d1 is None:
                        continue
                    delta = _std_norm_cdf(d1) if is_call else _std_norm_cdf(d1) - 1.0
                    diff = abs(delta - target_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best_iv = float(iv)
                return best_iv

            call_25 = _nearest_iv_by_delta(chain.calls, target_delta=0.25, is_call=True)
            put_25 = _nearest_iv_by_delta(chain.puts, target_delta=-0.25, is_call=False)
            atm_iv = _atm_iv_from_chain(chain, spot)

            skew = None
            if atm_iv and call_25 is not None and put_25 is not None:
                skew = (put_25 - call_25) / atm_iv

            return {
                "underlying": underlying,
                "expiration": target_exp,
                "spot": spot,
                "risk_free_rate": risk_free_rate,
                "atm_iv": atm_iv,
                "call_25d_iv": call_25,
                "put_25d_iv": put_25,
                "skew_25d": skew,
            }

        return await self._to_thread(_compute_skew)

    async def fetch_iv_hv(
        self,
        underlying: str,
        *,
        expiration_date: Optional[str] = None,
        spot_price: Optional[float] = None,
        window: int = 20,
    ) -> Dict[str, Any]:
        def _compute_iv_hv() -> Dict[str, Any]:
            ticker = self._make_ticker(underlying)
            spot = _infer_spot(ticker, spot_price)
            if spot is None:
                return {
                    "underlying": underlying,
                    "expiration": None,
                    "spot": None,
                    "atm_iv": None,
                    "hv": None,
                    "iv_vs_hv": None,
                }

            target_exp = _resolve_expiration(ticker, expiration_date)
            atm_iv = None
            if target_exp:
                try:
                    chain, _ = _option_chain(ticker, target_exp)
                except Exception:
                    chain = None
                if chain is not None:
                    atm_iv = _atm_iv_from_chain(chain, spot)

            hv = None
            try:
                hist = ticker.history(period="1y", interval="1d")
                if hist is not None and not hist.empty and len(hist.index) > window:
                    returns = hist["Close"].pct_change().dropna()
                    if not returns.empty:
                        hv = float(returns.tail(window).std() * math.sqrt(252))
            except Exception:
                hv = None

            iv_vs_hv = atm_iv - hv if atm_iv is not None and hv is not None else None

            return {
                "underlying": underlying,
                "expiration": target_exp,
                "spot": spot,
                "atm_iv": atm_iv,
                "hv": hv,
                "iv_vs_hv": iv_vs_hv,
            }

        return await self._to_thread(_compute_iv_hv)

    async def fetch_earnings_calendar(self, ticker: str) -> Dict[str, Any]:
        def _normalize(val: Any) -> Any:
            if isinstance(val, pd.Timestamp):
                return val.date().isoformat()
            if isinstance(val, datetime):
                return val.date().isoformat()
            if isinstance(val, str):
                return val
            return val

        def _load_calendar() -> Dict[str, Any]:
            t = self._make_ticker(ticker)
            cal = getattr(t, "calendar", None)
            if cal is None:
                return {"next_earnings_date": None, "raw": {}}

            data: Dict[str, Any] = {}
            try:
                if isinstance(cal, pd.DataFrame):
                    if cal.empty:
                        return {"next_earnings_date": None, "raw": {}}
                    series = cal.iloc[:, 0]
                    data = {str(k): _normalize(v) for k, v in series.to_dict().items()}
                elif isinstance(cal, dict):
                    data = {str(k): _normalize(v) for k, v in cal.items()}
            except Exception:
                data = {}

            next_date = None
            for key in ("Earnings Date", "EarningsDate", "Earnings Report Date"):
                if key in data and data[key]:
                    next_date = data[key]
                    break

            return {"next_earnings_date": next_date, "raw": data}

        return await self._to_thread(_load_calendar)

    async def fetch_peer_tickers(self, ticker: str, *, limit: int = 10) -> List[str]:
        def _load_peers() -> List[str]:
            peers: List[str] = []
            t = self._make_ticker(ticker)
            get_peers = getattr(t, "get_peers", None)
            if callable(get_peers):
                try:
                    raw = get_peers()
                except Exception:
                    raw = []
                if isinstance(raw, (list, tuple, set)):
                    peers.extend(str(item).upper() for item in raw if isinstance(item, str))
            return peers[:limit]

        try:
            return await self._to_thread(_load_peers)
        except ProviderError as exc:
            raise ProviderError(f"Failed to fetch peers for {ticker}: {exc}") from exc

    async def fetch_financials(self, ticker: str) -> Dict[str, Any]:
        def _load_financials() -> Dict[str, Any]:
            t = self._make_ticker(ticker)

            return {
                "income_statement": _df_records(getattr(t, "income_stmt", None)),
                "balance_sheet": _df_records(getattr(t, "balance_sheet", None)),
                "cash_flow": _df_records(getattr(t, "cashflow", None)),
                "quarterly_income_statement": _df_records(getattr(t, "quarterly_income_stmt", None)),
                "quarterly_balance_sheet": _df_records(getattr(t, "quarterly_balance_sheet", None)),
                "quarterly_cash_flow": _df_records(getattr(t, "quarterly_cashflow", None)),
                "ratios": [],
            }

        return await self._to_thread(_load_financials)

    async def fetch_holders(self, ticker: str) -> List[Dict[str, Any]]:
        def _load_holders() -> List[Dict[str, Any]]:
            t = self._make_ticker(ticker)
            return _df_records(getattr(t, "institutional_holders", None), date_label="as_of")

        return await self._to_thread(_load_holders)
