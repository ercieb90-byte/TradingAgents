"""
Fetches macroeconomic data from the FRED (Federal Reserve Economic Data) API.
Uses the REST API directly via requests - no special library needed.

All series are fetched with the most recent observations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import traceback

import requests


# FRED REST endpoint
_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series definitions: id -> human label
_SERIES = {
    "FEDFUNDS":         "Fed Funds Rate (%)",
    "CPIAUCSL":         "CPI (All Items, seasonally adj.)",
    "UNRATE":           "Unemployment Rate (%)",
    "A191RL1Q225SBEA":  "Real GDP Growth (annualized %)",
    "GS10":             "10-Year Treasury Yield (%)",
    "GS2":              "2-Year Treasury Yield (%)",
    "VIXCLS":           "VIX Fear Index",
    "UMCSENT":          "UMich Consumer Sentiment",
    "T10YIE":           "10-Year Breakeven Inflation Rate (%)",
    "BAMLH0A0HYM2":     "High Yield OAS Spread (bps)",
}


def fetch(fred_api_key: str) -> Dict[str, Any]:
    """
    Fetch macro-economic indicators from FRED.

    Args:
        fred_api_key: FRED API key. If empty, returns a stub with
                      ``{"available": False, "reason": "..."}``.

    Returns:
        Nested dict with keys matching ``_SERIES`` plus derived metrics
        (``cpi_yoy``, ``yield_curve_spread``, ``yield_curve_inverted``).
    """
    if not fred_api_key:
        return {
            "available": False,
            "reason": "No FRED API key configured. Get a free key at https://fred.stlouisfed.org/",
        }

    result: Dict[str, Any] = {"available": True, "series": {}, "derived": {}, "error": None}

    raw: Dict[str, list] = {}

    for series_id, label in _SERIES.items():
        try:
            observations = _fetch_series(series_id, fred_api_key, limit=14)
            raw[series_id] = observations
            latest = _latest_value(observations)
            result["series"][series_id] = {
                "label": label,
                "latest": latest,
                "recent_observations": observations[:6],  # last 6 for trend
            }
        except Exception as exc:
            result["series"][series_id] = {
                "label": label,
                "latest": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    # CPI year-over-year (need 12 months of data)
    try:
        cpi_obs = raw.get("CPIAUCSL", [])
        if len(cpi_obs) >= 13:
            cpi_latest = _value_at(cpi_obs, 0)
            cpi_12m_ago = _value_at(cpi_obs, 12)
            if cpi_latest is not None and cpi_12m_ago is not None and cpi_12m_ago != 0:
                cpi_yoy = round((cpi_latest / cpi_12m_ago - 1) * 100, 2)
                result["derived"]["cpi_yoy_pct"] = cpi_yoy
    except Exception:
        pass

    # Yield curve: 10Y - 2Y spread
    try:
        gs10 = result["series"].get("GS10", {}).get("latest")
        gs2  = result["series"].get("GS2", {}).get("latest")
        if gs10 is not None and gs2 is not None:
            spread = round(gs10 - gs2, 3)
            result["derived"]["yield_curve_spread_10y_2y"] = spread
            result["derived"]["yield_curve_inverted"] = bool(spread < 0)
    except Exception:
        pass

    # VIX regime
    try:
        vix = result["series"].get("VIXCLS", {}).get("latest")
        if vix is not None:
            if vix < 15:
                regime = "Low Volatility / Complacent"
            elif vix < 20:
                regime = "Normal Volatility"
            elif vix < 30:
                regime = "Elevated Volatility / Caution"
            else:
                regime = "High Fear / Crisis"
            result["derived"]["vix_regime"] = regime
    except Exception:
        pass

    # Fed funds vs. inflation (real rate)
    try:
        fedfunds = result["series"].get("FEDFUNDS", {}).get("latest")
        cpi_yoy  = result["derived"].get("cpi_yoy_pct")
        if fedfunds is not None and cpi_yoy is not None:
            result["derived"]["real_fed_funds_rate"] = round(fedfunds - cpi_yoy, 2)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_series(series_id: str, api_key: str, limit: int = 14) -> list:
    """
    Fetch FRED observations for *series_id*.

    Returns a list of dicts ``[{"date": "YYYY-MM-DD", "value": float}, ...]``
    sorted newest first. Missing/empty values are skipped.
    """
    params = {
        "series_id":   series_id,
        "api_key":     api_key,
        "file_type":   "json",
        "sort_order":  "desc",
        "limit":       limit,
    }
    resp = requests.get(_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    observations = []
    for obs in data.get("observations", []):
        raw_val = obs.get("value", ".")
        if raw_val in (".", "", None):
            continue
        try:
            observations.append({
                "date": obs["date"],
                "value": float(raw_val),
            })
        except (ValueError, KeyError):
            continue

    return observations


def _latest_value(observations: list) -> Optional[float]:
    """Return the value from the most recent observation."""
    if observations:
        return observations[0]["value"]
    return None


def _value_at(observations: list, index: int) -> Optional[float]:
    """Return the value at a specific index (0 = newest)."""
    try:
        return observations[index]["value"]
    except (IndexError, KeyError, TypeError):
        return None
