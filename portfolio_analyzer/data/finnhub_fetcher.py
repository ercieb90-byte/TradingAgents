"""
Fetches news, sentiment, earnings surprises, insider transactions, and analyst
consensus data from the Finnhub API (free tier).

Free tier limit: 60 requests / minute → we sleep 0.3 s between calls.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import traceback

import requests


_BASE_URL = "https://finnhub.io/api/v1"
_SLEEP    = 0.3   # seconds between API calls (free tier: 60 req/min)


def fetch(ticker: str, finnhub_api_key: str) -> Dict[str, Any]:
    """
    Fetch Finnhub data for *ticker*.

    Returns a nested dict with sections:
        - news          : last 7 days, up to 10 articles
        - metrics       : 50+ fundamental / technical metrics from Finnhub
        - earnings      : last 4 quarterly earnings surprises
        - insiders      : buy/sell counts for last 90 days
        - recommendation: latest analyst consensus

    If *finnhub_api_key* is empty, returns ``{"available": False}``.
    """
    if not finnhub_api_key:
        return {"available": False, "reason": "No Finnhub API key configured. Get a free key at https://finnhub.io/"}

    result: Dict[str, Any] = {
        "available": True,
        "ticker": ticker.upper(),
        "news": [],
        "metrics": {},
        "earnings": [],
        "insiders": {},
        "recommendation": {},
        "error": None,
    }

    try:
        sym = ticker.upper()

        # ---------------------------------------------------------------- news
        result["news"] = _fetch_news(sym, finnhub_api_key)
        time.sleep(_SLEEP)

        # ---------------------------------------------------------------- metrics
        result["metrics"] = _fetch_metrics(sym, finnhub_api_key)
        time.sleep(_SLEEP)

        # ---------------------------------------------------------------- earnings
        result["earnings"] = _fetch_earnings(sym, finnhub_api_key)
        time.sleep(_SLEEP)

        # ---------------------------------------------------------------- insiders
        result["insiders"] = _fetch_insiders(sym, finnhub_api_key)
        time.sleep(_SLEEP)

        # ---------------------------------------------------------------- recommendation
        result["recommendation"] = _fetch_recommendation(sym, finnhub_api_key)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["_traceback"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# Section fetchers
# ---------------------------------------------------------------------------

def _fetch_news(ticker: str, api_key: str) -> List[Dict[str, Any]]:
    """Company news for the last 7 days, up to 10 articles."""
    today   = date.today()
    week_ago = today - timedelta(days=7)
    params = {
        "symbol": ticker,
        "from":   str(week_ago),
        "to":     str(today),
        "token":  api_key,
    }
    resp = _get("/company-news", params)
    articles = resp if isinstance(resp, list) else []

    cleaned = []
    for art in articles[:10]:
        cleaned.append({
            "headline":  art.get("headline", ""),
            "summary":   art.get("summary", "")[:300] if art.get("summary") else "",
            "source":    art.get("source", ""),
            "datetime":  art.get("datetime"),
            "url":       art.get("url", ""),
            "sentiment": art.get("sentiment"),  # may be None on free tier
        })
    return cleaned


def _fetch_metrics(ticker: str, api_key: str) -> Dict[str, Any]:
    """All fundamental / technical metrics from Finnhub."""
    params = {"symbol": ticker, "metric": "all", "token": api_key}
    resp = _get("/stock/metric", params)
    if not isinstance(resp, dict):
        return {}

    metric_data = resp.get("metric", {})
    if not isinstance(metric_data, dict):
        return {}

    # Pick the most useful subset and rename for clarity
    useful_keys = [
        "52WeekHigh", "52WeekLow", "52WeekPriceReturnDaily",
        "10DayAverageTradingVolume", "3MonthAverageTradingVolume",
        "rsi14D", "revenueGrowthTTMYoy", "grossMarginTTM",
        "netProfitMarginTTM", "roeTTM", "roaTTM", "currentRatioQuarterly",
        "totalDebtToEquityQuarterly", "epsNormalizedAnnual",
        "revenuePerShareTTM", "bookValuePerShareQuarterly",
        "priceRelativeToS&P50013Week", "priceRelativeToS&P50026Week",
        "priceRelativeToS&P500YTD",
        "payoutRatioTTM", "dividendYieldIndicatedAnnual",
        "beta", "ltDebtToEquityQuarterly",
        "earningsPerShareTTM", "peNormalizedAnnual",
        "tangibleBookValuePerShareQuarterly",
    ]

    out = {}
    for k in useful_keys:
        v = metric_data.get(k)
        if v is not None:
            out[k] = v

    # Also include any remaining keys that weren't in our list
    for k, v in metric_data.items():
        if k not in out and v is not None:
            out[k] = v

    return out


def _fetch_earnings(ticker: str, api_key: str) -> List[Dict[str, Any]]:
    """Last 4 quarterly earnings with surprise calculations."""
    params = {"symbol": ticker, "token": api_key}
    resp = _get("/stock/earnings", params)
    if not isinstance(resp, list):
        return []

    results = []
    for q in resp[:4]:
        actual   = q.get("actual")
        estimate = q.get("estimate")
        surprise_pct: Optional[float] = None
        if actual is not None and estimate is not None and estimate != 0:
            try:
                surprise_pct = round((float(actual) - float(estimate)) / abs(float(estimate)) * 100, 2)
            except (TypeError, ZeroDivisionError):
                surprise_pct = None

        results.append({
            "period":       q.get("period"),
            "actual_eps":   actual,
            "estimate_eps": estimate,
            "surprise_pct": surprise_pct,
            "year":         q.get("year"),
            "quarter":      q.get("quarter"),
        })
    return results


def _fetch_insiders(ticker: str, api_key: str) -> Dict[str, Any]:
    """Insider transactions for the last 90 days."""
    today    = date.today()
    past_90  = today - timedelta(days=90)
    params = {
        "symbol": ticker,
        "from":   str(past_90),
        "to":     str(today),
        "token":  api_key,
    }
    resp = _get("/stock/insider-transactions", params)
    if not isinstance(resp, dict):
        return {}

    transactions = resp.get("data", [])
    if not isinstance(transactions, list):
        return {}

    buys  = [t for t in transactions if _is_buy(t)]
    sells = [t for t in transactions if _is_sell(t)]

    total_buy_shares  = sum(int(t.get("share", 0) or 0) for t in buys)
    total_sell_shares = sum(int(t.get("share", 0) or 0) for t in sells)

    net_shares = total_buy_shares - total_sell_shares
    sentiment: str
    if net_shares > 0:
        sentiment = "Bullish - Net buying"
    elif net_shares < 0:
        sentiment = "Bearish - Net selling"
    else:
        sentiment = "Neutral"

    return {
        "buy_count":         len(buys),
        "sell_count":        len(sells),
        "total_buy_shares":  total_buy_shares,
        "total_sell_shares": total_sell_shares,
        "net_shares":        net_shares,
        "insider_sentiment": sentiment,
        "recent_transactions": [
            {
                "name":       t.get("name", ""),
                "title":      t.get("officerTitle", ""),
                "transaction_type": t.get("transactionCode", ""),
                "shares":     t.get("share"),
                "price":      t.get("transactionPrice"),
                "date":       t.get("filingDate") or t.get("date"),
            }
            for t in transactions[:10]
        ],
    }


def _fetch_recommendation(ticker: str, api_key: str) -> Dict[str, Any]:
    """Latest analyst recommendation consensus."""
    params = {"symbol": ticker, "token": api_key}
    resp = _get("/stock/recommendation", params)
    if not isinstance(resp, list) or not resp:
        return {}

    latest = resp[0]  # most recent month first
    strong_buy  = int(latest.get("strongBuy",  0) or 0)
    buy         = int(latest.get("buy",        0) or 0)
    hold        = int(latest.get("hold",       0) or 0)
    sell        = int(latest.get("sell",       0) or 0)
    strong_sell = int(latest.get("strongSell", 0) or 0)
    total       = strong_buy + buy + hold + sell + strong_sell

    # Compute weighted consensus score: 1=Strong Buy … 5=Strong Sell
    consensus_score: Optional[float] = None
    if total > 0:
        consensus_score = round(
            (1 * strong_buy + 2 * buy + 3 * hold + 4 * sell + 5 * strong_sell) / total, 2
        )

    consensus_label = _score_to_label(consensus_score)

    return {
        "period":          latest.get("period"),
        "strong_buy":      strong_buy,
        "buy":             buy,
        "hold":            hold,
        "sell":            sell,
        "strong_sell":     strong_sell,
        "total_analysts":  total,
        "consensus_score": consensus_score,   # 1=Strong Buy, 5=Strong Sell
        "consensus_label": consensus_label,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict) -> Any:
    """Make a GET request to the Finnhub API and return parsed JSON."""
    url  = _BASE_URL + endpoint
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _is_buy(transaction: dict) -> bool:
    """Return True for open-market purchase transactions."""
    code = str(transaction.get("transactionCode", "")).upper()
    return code in ("P", "A")  # P = Purchase, A = Award/Grant treated as buy


def _is_sell(transaction: dict) -> bool:
    """Return True for open-market sale transactions."""
    code = str(transaction.get("transactionCode", "")).upper()
    return code in ("S", "D")  # S = Sale, D = Disposition


def _score_to_label(score: Optional[float]) -> str:
    if score is None:
        return "Unknown"
    if score <= 1.5:
        return "Strong Buy"
    if score <= 2.5:
        return "Buy"
    if score <= 3.5:
        return "Hold"
    if score <= 4.5:
        return "Sell"
    return "Strong Sell"
