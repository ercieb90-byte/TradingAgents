"""
Fetches price history, technical indicators, fundamental data, and analyst
targets for a given ticker via yfinance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import traceback

import numpy as np
import pandas as pd


def fetch(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive stock data for *ticker*.

    Returns a nested dict with sections:
        - price          : current price, 52-week stats, returns
        - technicals     : RSI, MACD, SMAs, Bollinger Bands, ATR, volume ratio, signals
        - fundamentals   : valuation ratios, margins, growth, quality metrics
        - financials     : quarterly revenue / net income / FCF / debt
        - analyst        : consensus recommendations + price targets
        - meta           : ticker, company name, sector, industry
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    result: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "price": {},
        "technicals": {},
        "fundamentals": {},
        "financials": {},
        "analyst": {},
        "meta": {},
        "error": None,
    }

    try:
        t = yf.Ticker(ticker)
        info = _safe_info(t)

        # ------------------------------------------------------------------ meta
        result["meta"] = {
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "exchange": info.get("exchange"),
            "currency": info.get("currency", "USD"),
        }

        # ------------------------------------------------------------ price history
        hist_1y = _safe_history(t, period="1y")
        hist_3m = _safe_history(t, period="3mo")
        hist_6m = _safe_history(t, period="6mo")
        hist_1m = _safe_history(t, period="1mo")
        hist_1w = _safe_history(t, period="5d")

        current_price = _last_close(hist_1y) or info.get("currentPrice") or info.get("regularMarketPrice")

        result["price"] = {
            "current": _round(current_price),
            "52w_high": _round(info.get("fiftyTwoWeekHigh") or _series_max(hist_1y)),
            "52w_low": _round(info.get("fiftyTwoWeekLow") or _series_min(hist_1y)),
            "return_1w": _pct_return(hist_1w),
            "return_1m": _pct_return(hist_1m),
            "return_3m": _pct_return(hist_3m),
            "return_6m": _pct_return(hist_6m),
            "return_1y": _pct_return(hist_1y),
        }

        # ------------------------------------------------------- technical indicators
        if hist_1y is not None and not hist_1y.empty:
            close = hist_1y["Close"].dropna()
            high  = hist_1y["High"].dropna()
            low   = hist_1y["Low"].dropna()
            volume = hist_1y["Volume"].dropna()

            rsi    = _compute_rsi(close, period=14)
            macd_line, signal_line, histogram = _compute_macd(close)
            sma20  = _sma(close, 20)
            sma50  = _sma(close, 50)
            sma200 = _sma(close, 200)
            bb_upper, bb_mid, bb_lower = _bollinger(close, 20, 2)
            atr    = _compute_atr(high, low, close, period=14)
            vol_ratio = _volume_ratio(volume, 20)

            price = float(close.iloc[-1])

            result["technicals"] = {
                "rsi_14": _round(rsi),
                "macd_line": _round(macd_line),
                "macd_signal": _round(signal_line),
                "macd_histogram": _round(histogram),
                "sma_20": _round(sma20),
                "sma_50": _round(sma50),
                "sma_200": _round(sma200),
                "bb_upper": _round(bb_upper),
                "bb_mid": _round(bb_mid),
                "bb_lower": _round(bb_lower),
                "atr_14": _round(atr),
                "volume_ratio_20d": _round(vol_ratio),
                # price distance from each SMA in %
                "pct_above_sma20": _round(_pct_dist(price, sma20)),
                "pct_above_sma50": _round(_pct_dist(price, sma50)),
                "pct_above_sma200": _round(_pct_dist(price, sma200)),
                # cross signals
                "golden_cross": _golden_cross(sma50, sma200),   # True = SMA50 > SMA200
                "death_cross": _death_cross(sma50, sma200),
                "rsi_signal": _rsi_signal(rsi),
                "bb_position": _bb_position(price, bb_upper, bb_lower),
            }
        else:
            result["technicals"] = {"error": "Insufficient price history"}

        # ------------------------------------------------------- fundamentals
        result["fundamentals"] = {
            "market_cap": info.get("marketCap"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "profit_margin": _pct_fmt(info.get("profitMargins")),
            "roe": _pct_fmt(info.get("returnOnEquity")),
            "roa": _pct_fmt(info.get("returnOnAssets")),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "dividend_yield": _pct_fmt(info.get("dividendYield")),
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "revenue_growth": _pct_fmt(info.get("revenueGrowth")),
            "earnings_growth": _pct_fmt(info.get("earningsGrowth")),
            "beta": info.get("beta"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
        }

        # ------------------------------------------------------- financial statements
        result["financials"] = _fetch_financials(t)

        # ------------------------------------------------------- analyst data
        result["analyst"] = _fetch_analyst(t, info)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["_traceback"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# Financial statements helper
# ---------------------------------------------------------------------------

def _fetch_financials(t: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        qfin = t.quarterly_financials
        if qfin is not None and not qfin.empty:
            # columns are dates (newest first in some versions), rows are line items
            rev_row = _find_row(qfin, ["Total Revenue", "totalRevenue"])
            ni_row  = _find_row(qfin, ["Net Income", "netIncome"])
            if rev_row is not None:
                vals = rev_row.dropna().head(4).tolist()
                out["revenue_quarterly"] = [_round(v) for v in vals]
                if len(vals) >= 2:
                    out["revenue_qoq_growth_pct"] = _round((vals[0] / vals[1] - 1) * 100)
            if ni_row is not None:
                vals = ni_row.dropna().head(4).tolist()
                out["net_income_quarterly"] = [_round(v) for v in vals]
    except Exception:
        pass

    try:
        qcf = t.quarterly_cashflow
        if qcf is not None and not qcf.empty:
            fcf_row = _find_row(qcf, ["Free Cash Flow", "freeCashFlow"])
            if fcf_row is not None:
                out["fcf_latest_quarter"] = _round(fcf_row.dropna().iloc[0])
    except Exception:
        pass

    try:
        bs = t.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            debt_row = _find_row(bs, ["Total Debt", "Long Term Debt", "totalDebt", "longTermDebt"])
            if debt_row is not None:
                out["total_debt_latest"] = _round(debt_row.dropna().iloc[0])
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Analyst data helper
# ---------------------------------------------------------------------------

def _fetch_analyst(t: Any, info: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Recommendations
    try:
        recs = t.recommendations
        if recs is not None and not recs.empty:
            # Newer yfinance returns columns: period, strongBuy, buy, hold, sell, strongSell
            if "period" in recs.columns:
                latest = recs[recs["period"] == "0m"]
                if latest.empty:
                    latest = recs.head(1)
                if not latest.empty:
                    row = latest.iloc[0]
                    out["rec_strong_buy"] = int(row.get("strongBuy", 0))
                    out["rec_buy"] = int(row.get("buy", 0))
                    out["rec_hold"] = int(row.get("hold", 0))
                    out["rec_sell"] = int(row.get("sell", 0))
                    out["rec_strong_sell"] = int(row.get("strongSell", 0))
                    out["rec_total"] = sum([
                        out["rec_strong_buy"], out["rec_buy"],
                        out["rec_hold"],
                        out["rec_sell"], out["rec_strong_sell"],
                    ])
            else:
                # Older yfinance: index = date, columns include "To Grade"
                latest = recs.tail(30)
                buy_grades = ["Buy", "Strong Buy", "Outperform", "Overweight", "Accumulate"]
                sell_grades = ["Sell", "Strong Sell", "Underperform", "Underweight"]
                hold_grades = ["Hold", "Neutral", "Market Perform", "Equal-Weight"]
                graded = latest.get("To Grade", latest.get("toGrade", pd.Series(dtype=str)))
                out["rec_buy"] = int(graded.isin(buy_grades).sum())
                out["rec_hold"] = int(graded.isin(hold_grades).sum())
                out["rec_sell"] = int(graded.isin(sell_grades).sum())
    except Exception:
        pass

    # Analyst price targets
    try:
        apt = t.analyst_price_targets
        if apt is not None:
            if isinstance(apt, dict):
                out["target_mean"] = _round(apt.get("mean"))
                out["target_high"] = _round(apt.get("high"))
                out["target_low"] = _round(apt.get("low"))
            elif hasattr(apt, "mean"):
                out["target_mean"] = _round(float(apt["mean"].iloc[0]) if hasattr(apt["mean"], "iloc") else apt["mean"])
    except Exception:
        pass

    # Also from info dict as fallback
    if "target_mean" not in out:
        out["target_mean"] = _round(info.get("targetMeanPrice"))
        out["target_high"] = _round(info.get("targetHighPrice"))
        out["target_low"] = _round(info.get("targetLowPrice"))

    out["recommendation_key"] = info.get("recommendationKey")
    out["number_of_analyst_opinions"] = info.get("numberOfAnalystOpinions")

    return out


# ---------------------------------------------------------------------------
# Technical indicator computations
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    """Compute RSI via Wilder's smoothed average."""
    try:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return None


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute MACD line, signal line, and histogram."""
    try:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])
    except Exception:
        return None, None, None


def _sma(close: pd.Series, period: int) -> Optional[float]:
    try:
        if len(close) < period:
            return None
        return float(close.rolling(period).mean().iloc[-1])
    except Exception:
        return None


def _bollinger(
    close: pd.Series, period: int = 20, num_std: int = 2
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])
    except Exception:
        return None, None, None


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Optional[float]:
    """Average True Range."""
    try:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return None


def _volume_ratio(volume: pd.Series, period: int = 20) -> Optional[float]:
    try:
        avg = volume.rolling(period).mean().iloc[-1]
        current = volume.iloc[-1]
        return float(current / avg) if avg else None
    except Exception:
        return None


def _golden_cross(sma50: Optional[float], sma200: Optional[float]) -> Optional[bool]:
    if sma50 is None or sma200 is None:
        return None
    return bool(sma50 > sma200)


def _death_cross(sma50: Optional[float], sma200: Optional[float]) -> Optional[bool]:
    if sma50 is None or sma200 is None:
        return None
    return bool(sma50 < sma200)


def _rsi_signal(rsi: Optional[float]) -> Optional[str]:
    if rsi is None:
        return None
    if rsi >= 70:
        return "Overbought"
    if rsi <= 30:
        return "Oversold"
    return "Neutral"


def _bb_position(price: float, upper: Optional[float], lower: Optional[float]) -> Optional[str]:
    if upper is None or lower is None:
        return None
    if price > upper:
        return "Above Upper Band"
    if price < lower:
        return "Below Lower Band"
    return "Within Bands"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_info(t: Any) -> dict:
    try:
        info = t.info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _safe_history(t: Any, period: str) -> Optional[pd.DataFrame]:
    try:
        h = t.history(period=period, auto_adjust=True)
        return h if h is not None and not h.empty else None
    except Exception:
        return None


def _last_close(hist: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        if hist is not None and not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def _series_max(hist: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        return float(hist["High"].max()) if hist is not None else None
    except Exception:
        return None


def _series_min(hist: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        return float(hist["Low"].min()) if hist is not None else None
    except Exception:
        return None


def _pct_return(hist: Optional[pd.DataFrame]) -> Optional[float]:
    """Return percentage gain from first to last close."""
    try:
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None
        return _round((float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100)
    except Exception:
        return None


def _pct_fmt(val: Any) -> Optional[float]:
    """Convert a ratio (e.g. 0.12) to percentage (12.0)."""
    try:
        return _round(float(val) * 100) if val is not None else None
    except Exception:
        return None


def _pct_dist(price: float, sma: Optional[float]) -> Optional[float]:
    """(price / sma - 1) * 100"""
    if sma is None or sma == 0:
        return None
    return (price / sma - 1) * 100


def _round(val: Any, digits: int = 2) -> Optional[float]:
    try:
        return round(float(val), digits) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
    except Exception:
        return None


def _find_row(df: pd.DataFrame, candidate_names: list) -> Optional[pd.Series]:
    """Find a row in a financial DataFrame by trying several possible index labels."""
    for name in candidate_names:
        if name in df.index:
            return df.loc[name]
    # Case-insensitive fallback
    lower_index = {str(i).lower(): i for i in df.index}
    for name in candidate_names:
        key = name.lower()
        if key in lower_index:
            return df.loc[lower_index[key]]
    return None
