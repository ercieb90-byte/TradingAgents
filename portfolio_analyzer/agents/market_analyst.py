"""
Market / Technical Analyst Agent.

Analyzes technical indicators from yfinance data and produces a
structured technical analysis report.
"""

from __future__ import annotations

from typing import Any, Dict


def analyze(ticker: str, data_bundle: Dict[str, Any], llm_client: Any) -> str:
    """
    Perform technical analysis for *ticker*.

    Args:
        ticker:      Stock symbol.
        data_bundle: Full data dict from the orchestrator
                     (contains "yfinance", "fred", "finnhub" keys).
        llm_client:  LLMClient instance.

    Returns:
        A string report with 5-7 key technical findings.
    """
    yf_data = data_bundle.get("yfinance", {})
    tech    = yf_data.get("technicals", {})
    price   = yf_data.get("price", {})
    meta    = yf_data.get("meta", {})

    company_name = meta.get("name", ticker)
    language     = data_bundle.get("_language", "German")

    # Build a readable technical data block
    tech_summary = _format_technicals(ticker, company_name, price, tech)

    system_prompt = f"""Du bist ein erfahrener technischer Analyst mit 20 Jahren Börsenerfahrung.
Du spezialisierst dich auf Chart-Analyse, Momentum-Strategien und Risikomanagement.
Antworte ausschließlich auf {language}.

Deine Aufgabe ist es, die technischen Indikatoren einer Aktie zu analysieren und
5-7 präzise, handlungsrelevante Erkenntnisse zu liefern. Sei konkret und nutze die
tatsächlichen Zahlenwerte in deiner Analyse.""" if language == "German" else f"""You are an experienced technical analyst with 20 years of market experience.
You specialize in chart analysis, momentum strategies, and risk management.
Always respond in {language}.

Your task is to analyze technical indicators for a stock and deliver
5-7 precise, actionable findings. Be specific and use the actual numeric values in your analysis."""

    user_prompt = f"""Analysiere die folgenden technischen Indikatoren für {ticker} ({company_name}):

{tech_summary}

Erstelle einen technischen Analysebericht mit:
1. **Trendanalyse**: Primärer Trend (Bull/Bear/Seitwärts), Stärke des Trends
2. **Momentum**: RSI-Bewertung, MACD-Signal, was sagen diese Indikatoren?
3. **Gleitende Durchschnitte**: Position des Kurses relativ zu SMA20/50/200, Golden/Death Cross
4. **Bollinger Bands**: Position im Band, Volatilität
5. **Unterstützung & Widerstand**: Wichtige Preisniveaus basierend auf den SMAs und Bollinger Bands
6. **Volumen**: Interpretation des Volumens relativ zum Durchschnitt
7. **Risiko/Reward**: ATR-basierte Volatilitätseinschätzung, mögliche Stop-Loss-Level

Fasse am Ende in 2-3 Sätzen die technische Gesamteinschätzung zusammen (Bullish/Neutral/Bearish).
""" if language == "German" else f"""Analyze the following technical indicators for {ticker} ({company_name}):

{tech_summary}

Create a technical analysis report covering:
1. **Trend Analysis**: Primary trend (Bull/Bear/Sideways), trend strength
2. **Momentum**: RSI assessment, MACD signal interpretation
3. **Moving Averages**: Price position relative to SMA20/50/200, Golden/Death Cross
4. **Bollinger Bands**: Band position, volatility assessment
5. **Support & Resistance**: Key price levels based on SMAs and Bollinger Bands
6. **Volume**: Volume interpretation relative to 20-day average
7. **Risk/Reward**: ATR-based volatility assessment, suggested stop-loss levels

Conclude with a 2-3 sentence overall technical assessment (Bullish/Neutral/Bearish).
"""

    return llm_client.complete(system_prompt, user_prompt, deep=False)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_technicals(ticker: str, name: str, price: dict, tech: dict) -> str:
    """Format price and technical data into a readable text block."""
    lines = []

    # Price overview
    current = price.get("current")
    w52_high = price.get("52w_high")
    w52_low  = price.get("52w_low")

    if current:
        lines.append(f"Aktueller Kurs: ${current}")
    if w52_high and w52_low:
        lines.append(f"52-Wochen-Hoch: ${w52_high} | 52-Wochen-Tief: ${w52_low}")

    # Returns
    ret_parts = []
    for label, key in [("1W", "return_1w"), ("1M", "return_1m"), ("3M", "return_3m"),
                        ("6M", "return_6m"), ("1J", "return_1y")]:
        val = price.get(key)
        if val is not None:
            sign = "+" if val >= 0 else ""
            ret_parts.append(f"{label}: {sign}{val}%")
    if ret_parts:
        lines.append("Performance: " + " | ".join(ret_parts))

    lines.append("")

    # Momentum indicators
    rsi = tech.get("rsi_14")
    if rsi is not None:
        signal = tech.get("rsi_signal", "")
        lines.append(f"RSI(14): {rsi} → {signal}")

    macd = tech.get("macd_line")
    macd_sig = tech.get("macd_signal")
    macd_hist = tech.get("macd_histogram")
    if macd is not None:
        direction = "Bullish" if (macd_hist or 0) > 0 else "Bearish"
        lines.append(f"MACD: Linie={macd}, Signal={macd_sig}, Histogram={macd_hist} → {direction}")

    lines.append("")

    # Moving averages
    sma20  = tech.get("sma_20")
    sma50  = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    p20    = tech.get("pct_above_sma20")
    p50    = tech.get("pct_above_sma50")
    p200   = tech.get("pct_above_sma200")
    if sma20:
        lines.append(f"SMA20: ${sma20} (Kurs {_dist_str(p20)} vom SMA20)")
    if sma50:
        lines.append(f"SMA50: ${sma50} (Kurs {_dist_str(p50)} vom SMA50)")
    if sma200:
        lines.append(f"SMA200: ${sma200} (Kurs {_dist_str(p200)} vom SMA200)")

    golden = tech.get("golden_cross")
    death  = tech.get("death_cross")
    if golden is True:
        lines.append("✓ Golden Cross aktiv (SMA50 > SMA200) – Bullisches Signal")
    elif death is True:
        lines.append("✗ Death Cross aktiv (SMA50 < SMA200) – Bearishes Signal")

    lines.append("")

    # Bollinger Bands
    bb_u = tech.get("bb_upper")
    bb_m = tech.get("bb_mid")
    bb_l = tech.get("bb_lower")
    bb_pos = tech.get("bb_position")
    if bb_u and bb_l:
        lines.append(f"Bollinger Bands (20,2): Oberes Band=${bb_u}, Mitte=${bb_m}, Unteres Band=${bb_l}")
        lines.append(f"  → Kursposition: {bb_pos}")

    lines.append("")

    # ATR & volume
    atr = tech.get("atr_14")
    vol = tech.get("volume_ratio_20d")
    if atr:
        lines.append(f"ATR(14): ${atr} (Tagesvolatilität / Stop-Loss-Referenz)")
    if vol:
        lines.append(f"Volumen-Ratio (heute vs. 20d-Durchschnitt): {vol}x")

    return "\n".join(lines)


def _dist_str(pct: float | None) -> str:
    if pct is None:
        return "k.A."
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"
