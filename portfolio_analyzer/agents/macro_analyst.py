"""
Macro / FRED Analyst Agent.

Analyzes macroeconomic conditions from FRED data and assesses their
impact on the target stock / sector.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def analyze(ticker: str, data_bundle: Dict[str, Any], llm_client: Any) -> str:
    """
    Analyze the macroeconomic environment and its impact on *ticker*.

    Args:
        ticker:      Stock symbol.
        data_bundle: Full data dict (contains "yfinance", "fred", "finnhub").
        llm_client:  LLMClient instance.

    Returns:
        A string report on macro environment and sector/stock impact.
    """
    fred_data = data_bundle.get("fred", {})
    yf_data   = data_bundle.get("yfinance", {})
    meta      = yf_data.get("meta", {})
    language  = data_bundle.get("_language", "German")

    company_name = meta.get("name", ticker)
    sector       = meta.get("sector", "Unbekannt")
    beta         = yf_data.get("fundamentals", {}).get("beta")

    if not fred_data.get("available", False):
        reason = fred_data.get("reason", "FRED-Daten nicht verfügbar.")
        return (
            f"## Makroökonomische Analyse – {ticker}\n\n"
            f"**Hinweis:** {reason}\n\n"
            "Ohne FRED-Daten kann keine vollständige Makroanalyse durchgeführt werden. "
            "Bitte FRED_API_KEY in der .env Datei konfigurieren (kostenlos auf fred.stlouisfed.org)."
        )

    macro_summary = _format_fred_data(fred_data, beta)

    system_prompt = (
        f"Du bist ein erfahrener Makroökonom und Investmentstratege. "
        f"Du analysierst gesamtwirtschaftliche Daten und leitest daraus Auswirkungen auf einzelne Sektoren und Aktien ab. "
        f"Antworte ausschließlich auf {language}."
    ) if language == "German" else (
        f"You are an experienced macro economist and investment strategist. "
        f"You analyze macroeconomic data and derive implications for individual sectors and stocks. "
        f"Always respond in {language}."
    )

    user_prompt = f"""Analysiere das folgende makroökonomische Umfeld und seine Auswirkungen auf {ticker} ({company_name}),
das im Sektor **{sector}** tätig ist (Beta: {beta if beta else 'k.A.'}).

{macro_summary}

Erstelle einen Makroanalyse-Bericht mit:
1. **Zinsumfeld**: Fed Funds Rate-Bewertung, Zinserwartungen, Auswirkung auf Bewertungen
2. **Inflationstrend**: CPI-Trend, reale Zinsen, was bedeutet das für diesen Sektor?
3. **Wirtschaftszyklus**: BIP-Wachstum, Arbeitslosigkeit, in welcher Zyklusphase befinden wir uns?
4. **Risikosentiment**: VIX-Level, Verbrauchervertrauen, Credit Spreads
5. **Zinskurve**: Invertierung? Was signalisiert das für die Wirtschaft?
6. **Sektorspezifische Auswirkung**: Wie wirkt das Makroumfeld konkret auf {sector} und {ticker}?
7. **Gesamteinschätzung**: Ist das Makroumfeld für diese Aktie rückenwind oder gegenwind?

Sei präzise und nutze die konkreten Datenpunkte in der Analyse.
""" if language == "German" else f"""Analyze the following macroeconomic environment and its impact on {ticker} ({company_name}),
operating in the **{sector}** sector (Beta: {beta if beta else 'N/A'}).

{macro_summary}

Create a macro analysis report covering:
1. **Interest Rate Environment**: Fed Funds assessment, rate expectations, impact on valuations
2. **Inflation Trend**: CPI trend, real rates, implications for this sector
3. **Economic Cycle**: GDP growth, unemployment, current cycle phase
4. **Risk Sentiment**: VIX level, consumer confidence, credit spreads
5. **Yield Curve**: Inversion? What does it signal for the economy?
6. **Sector-Specific Impact**: How does the macro environment specifically affect {sector} and {ticker}?
7. **Overall Assessment**: Is the macro environment a tailwind or headwind for this stock?

Be precise and use the concrete data points in your analysis.
"""

    return llm_client.complete(system_prompt, user_prompt, deep=False)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_fred_data(fred_data: dict, beta: Optional[float]) -> str:
    """Format FRED series and derived metrics into a readable text block."""
    lines = ["=== FRED Makrodaten ===", ""]

    series = fred_data.get("series", {})
    derived = fred_data.get("derived", {})

    # Key rate indicators
    _add_series_line(lines, series, "FEDFUNDS", "Fed Funds Rate")
    _add_series_line(lines, series, "GS10",     "10-Jahres-Treasury-Rendite")
    _add_series_line(lines, series, "GS2",      "2-Jahres-Treasury-Rendite")

    spread = derived.get("yield_curve_spread_10y_2y")
    inverted = derived.get("yield_curve_inverted")
    if spread is not None:
        inv_flag = " ⚠️ INVERTIERT" if inverted else ""
        lines.append(f"Zinskurve (10J-2J): {spread:+.3f}%{inv_flag}")

    real_rate = derived.get("real_fed_funds_rate")
    if real_rate is not None:
        lines.append(f"Realer Leitzins (Fed - CPI YoY): {real_rate:+.2f}%")

    lines.append("")

    # Inflation
    cpi_series = series.get("CPIAUCSL", {})
    cpi_latest = cpi_series.get("latest")
    cpi_yoy    = derived.get("cpi_yoy_pct")
    if cpi_latest or cpi_yoy:
        parts = []
        if cpi_latest:
            parts.append(f"Index={cpi_latest}")
        if cpi_yoy is not None:
            parts.append(f"YoY={cpi_yoy:+.2f}%")
        lines.append(f"CPI: {' | '.join(parts)}")

    _add_series_line(lines, series, "T10YIE", "10J Break-even Inflation (Markterwartung)")

    lines.append("")

    # Economic activity
    _add_series_line(lines, series, "A191RL1Q225SBEA", "Reales BIP-Wachstum (annualisiert %)")
    _add_series_line(lines, series, "UNRATE",          "Arbeitslosenquote (%)")

    lines.append("")

    # Risk sentiment
    vix_series = series.get("VIXCLS", {})
    vix = vix_series.get("latest")
    vix_regime = derived.get("vix_regime", "")
    if vix is not None:
        lines.append(f"VIX: {vix} → {vix_regime}")

    _add_series_line(lines, series, "UMCSENT",      "UMich Verbrauchervertrauen")
    _add_series_line(lines, series, "BAMLH0A0HYM2", "High-Yield Credit Spread (bps)")

    if beta is not None:
        lines.append("")
        lines.append(f"Aktien-Beta: {beta} (Sensitivität gegenüber Marktbewegungen)")

    return "\n".join(lines)


def _add_series_line(lines: list, series: dict, series_id: str, label: str) -> None:
    """Add a formatted line for a FRED series if data is available."""
    data = series.get(series_id, {})
    val  = data.get("latest")
    if val is not None:
        lines.append(f"{label}: {val}")
    else:
        err = data.get("error")
        if err:
            lines.append(f"{label}: Nicht verfügbar ({err})")
