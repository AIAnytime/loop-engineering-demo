"""Finnhub tools -- the loop's connection to ground truth.

Every tool returns plain dicts and records what it fetched. That record is the
*evidence table*: the verifier checks the memo against it, so a number that
never came out of a tool call is by definition a hallucination.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import requests

from .config import FINNHUB_BASE, FINNHUB_API_KEY


class ToolError(RuntimeError):
    pass


def _get(path: str, **params: Any) -> Any:
    if not FINNHUB_API_KEY:
        raise ToolError("FINHUB_API_KEY is not set (check .env)")
    params["token"] = FINNHUB_API_KEY
    resp = requests.get(f"{FINNHUB_BASE}/{path}", params=params, timeout=30)
    if resp.status_code == 429:
        raise ToolError("Finnhub rate limit hit (free tier: 60 calls/min)")
    resp.raise_for_status()
    return resp.json()


def get_quote(ticker: str) -> dict:
    q = _get("quote", symbol=ticker)
    if not q or q.get("c") in (None, 0):
        raise ToolError(f"no quote for {ticker}")
    return {
        "current": q["c"],
        "change": q.get("d"),
        "percent_change": q.get("dp"),
        "high": q.get("h"),
        "low": q.get("l"),
        "open": q.get("o"),
        "prev_close": q.get("pc"),
    }


def get_profile(ticker: str) -> dict:
    p = _get("stock/profile2", symbol=ticker)
    return {
        "name": p.get("name"),
        "exchange": p.get("exchange"),
        "industry": p.get("finnhubIndustry"),
        "market_cap_musd": p.get("marketCapitalization"),
        "shares_outstanding_m": p.get("shareOutstanding"),
        "ipo": p.get("ipo"),
    }


_METRICS = {
    "peTTM": "pe_ttm",
    "psTTM": "ps_ttm",
    "pbAnnual": "pb_annual",
    "roeTTM": "roe_ttm",
    "roaTTM": "roa_ttm",
    "grossMarginTTM": "gross_margin_ttm",
    "netProfitMarginTTM": "net_margin_ttm",
    "currentRatioQuarterly": "current_ratio",
    "totalDebt/totalEquityQuarterly": "debt_to_equity",
    "52WeekHigh": "week52_high",
    "52WeekLow": "week52_low",
    "beta": "beta",
    "dividendYieldIndicatedAnnual": "dividend_yield",
    "revenueGrowthTTMYoy": "revenue_growth_yoy",
    "epsGrowthTTMYoy": "eps_growth_yoy",
}


def get_metrics(ticker: str) -> dict:
    raw = (_get("stock/metric", symbol=ticker, metric="all") or {}).get("metric", {})
    return {friendly: raw.get(key) for key, friendly in _METRICS.items() if raw.get(key) is not None}


def get_recommendations(ticker: str) -> dict:
    rows = _get("stock/recommendation", symbol=ticker) or []
    if not rows:
        return {}
    latest = rows[0]
    return {
        "period": latest.get("period"),
        "strong_buy": latest.get("strongBuy"),
        "buy": latest.get("buy"),
        "hold": latest.get("hold"),
        "sell": latest.get("sell"),
        "strong_sell": latest.get("strongSell"),
    }


def get_news(ticker: str, days: int = 14, limit: int = 6) -> list[dict]:
    today = dt.date.today()
    rows = _get(
        "company-news",
        symbol=ticker,
        **{"from": str(today - dt.timedelta(days=days)), "to": str(today)},
    ) or []
    out = []
    for item in rows[:limit]:
        out.append(
            {
                "headline": item.get("headline"),
                "source": item.get("source"),
                "date": dt.datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d"),
                "url": item.get("url"),
            }
        )
    return out


def collect_evidence(ticker: str) -> dict:
    """One call the loop makes per attempt: everything the memo may cite.

    Tools that fail degrade to an explicit error string instead of vanishing --
    the verifier needs to know the difference between 'no data' and 'not fetched'.
    """
    evidence: dict[str, Any] = {"ticker": ticker.upper(), "fetched_at": dt.datetime.now().isoformat(timespec="seconds")}
    for key, fn in (
        ("quote", get_quote),
        ("profile", get_profile),
        ("metrics", get_metrics),
        ("analyst_consensus", get_recommendations),
        ("news", get_news),
    ):
        try:
            evidence[key] = fn(ticker)
        except Exception as exc:  # noqa: BLE001 - degrade, never silently drop
            evidence[key] = {"error": str(exc)[:160]}
    return evidence
